import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel, \
    DebertaForMaskedLM, DebertaTokenizer, BertForMaskedLM, BertTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
from rlprompt.rewards import BaseReward
import openai
import time
import concurrent.futures
import math

SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


class PromptedClassificationReward(BaseReward):
    def __init__(
            self,
            task_lm: str,
            is_mask_lm: Optional[bool],
            compute_zscore: bool,
            incorrect_coeff: float,
            correct_coeff: float,
            num_classes: int,
            verbalizers: List[str],
            template: Optional[str],
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.task_lm = task_lm
        if is_mask_lm is None:
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm
        print('Task LM:', self.task_lm)
        if "deberta" in self.task_lm:
            self._tokenizer = DebertaTokenizer.from_pretrained('lsanochkin/deberta-large-feedback')
            self._generator = (DebertaForMaskedLM
                               .from_pretrained('lsanochkin/deberta-large-feedback')
                               .to(self.device))
        elif self.task_lm == 'bert-large-cased':
            self._tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            self._generator = (BertForMaskedLM
                               .from_pretrained('bert-large-cased')
                               .to(self.device))
        elif self.task_lm == 'gpt-j':
            self._tokenizer = AutoTokenizer.from_pretrained(
                'EleutherAI/gpt-j-6B', pad_token='<|endoftext|>',
                revision="float16", torch_dtype=torch.float16
            )
            self._generator = (AutoModelForCausalLM.from_pretrained(
                'EleutherAI/gpt-j-6B', revision="float16", torch_dtype=torch.float16,
            ).to(self.device))
        elif self.task_lm in ["gpt3.5", "gpt3"]:
            openai.api_key = "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"
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2", pad_token='<|endoftext|>')
            self._generator = (GPT2LMHeadModel.from_pretrained("gpt2").to(self.device))
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
        elif self.task_lm == "llama-2-7b":
            self._tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            self._tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self._generator = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(self.device)
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id

            self._generator.resize_token_embeddings(len(self._tokenizer))
        elif self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(self.task_lm)
                               .to(self.device))
        else:
            assert self.task_lm in SUPPORTED_LEFT_TO_RIGHT_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.task_lm, pad_token='<|endoftext|>')
            self._generator = (GPT2LMHeadModel
                               .from_pretrained(self.task_lm)
                               .to(self.device))
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id

        self.compute_zscore = compute_zscore
self.incorrect_coeff = incorrect_coeff
self.correct_coeff = correct_coeff
self.num_classes = num_classes
self.verbalizers = verbalizers
print('Verbalizers:', self.verbalizers)
if self.task_lm not in ["gpt3.5", "gpt4"]:
    self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v) for v in self.verbalizers]
self._counter = 0

def load_default_template(self) -> Tuple[str, Any]:
    if self.is_mask_lm:
        mask_token = self._tokenizer.mask_token
        template = f"{{sentence}} {{clean_prompt}} {mask_token} ."
    else:
        template = "{sentence} {clean_prompt}"  # Template for left-to-right LMs like GPT-2
    return template

def forward(
        self,
        source_texts: List[str],
        clean_prompt: str,
        target_label: int,
        class_labels: List[int],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str,
) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any], Dict[Tuple, Tuple], Dict[Tuple, Tuple]:
    assert mode in ["train", "infer"]

    if mode == "train":
        self._counter += 1

    prompt_tokens = output_tokens
    prompt_strings = self._convert_tokens_to_string(prompt_tokens)
    batch_size = len(source_texts)

    rewards: List[torch.Tensor] = []
    input_rewards: Dict[str, List[float]] = defaultdict(list)
    quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for i, prompt in enumerate(prompt_strings):
        current_prompts = [prompt for _ in source_texts]
        formatted_templates = self._format_prompts(source_texts, clean_prompt)
        all_logits = self._get_logits(formatted_templates)

        if self.task_lm in ["gpt3.5", "gpt4"]:
            class_probs = all_logits
        else:
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)

        label_probs = class_probs[range(batch_size), class_labels]
        not_label_probs = torch.where(
            class_probs == label_probs.unsqueeze(1),
            torch.Tensor([-1]).to(self.device), class_probs)

        max_not_label_probs, _ = torch.max(not_label_probs, -1)

        gap = (label_probs - max_not_label_probs)
        correct = (gap > 0).long()
        reward = reward_paa.mean()

        sim = correct.float().mean()
        quantities_to_log['sim'] = sim
        quantities_to_log['paa'] = paa
        quantities_to_log['gap_reward'].append(reward.item())
        rewards.append(reward)

        input_rewards['z'] += [reward.item()]

        print_strs = [self._counter, '|', prompt, '\n']
        print_strs += ['similarity:', sim.item(), '|',
                       'paa:', paa.item(), '|',
                       'Reward:', round(reward.item(), 2)]
        print(*print_strs)
    rewards_tensor = torch.stack(rewards)

    if mode == 'train' and self.compute_zscore:
        input_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
        input_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
        idx_means = torch.tensor(input_reward_means['z']).float()
        idx_stds = torch.tensor(input_reward_stds['z']).float()
        rewards_tensor = (rewards_tensor - idx_means) / (idx_stds + 1e-4)
        for i in range(rewards_tensor.size(0)):
            quantities_to_log['resized_reward'].append(rewards_tensor[i].item())
    elif mode == 'infer':
        score = rewards_tensor.mean().item()
        print('Our Prompt:')
        print(prompt_strings, score)

    rewards_log = dict(
        (reward_key, torch.mean(torch.tensor(reward_vals))) for reward_key, reward_vals in quantities_to_log.items())

    if to_tensor is True:
        return rewards_tensor, rewards_log
    else:
        return rewards_tensor.tolist(), rewards_log

def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
    mask_token_index = torch.where(input_ids == self._tokenizer.mask_token_id)[1]
    return mask_token_index

def ensure_exactly_one_mask_token(
        self,
        model_inputs: Dict[str, torch.Tensor]
) -> None:
    for input_ids in model_inputs["input_ids"]:
        masked_index = self._get_mask_token_index(input_ids)
        numel = np.prod(masked_index.shape)
        assert numel == 1

@torch.no_grad()
def _get_logits(
        self,
        texts: List[str]
) -> torch.Tensor:
    batch_size = len(texts)
    encoded_inputs = self._tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True)

    if self.is_mask_lm:
        token_logits = self._generator(**encoded_inputs.to(self.device)).logits
        mask_token_indices = self._get_mask_token_index(encoded_inputs['input_ids'])
        out_logits = token_logits[range(batch_size), mask_token_indices, :]
    elif self.task_lm == "gpt3.5":
        # Implementation for GPT-3.5
    elif self.task_lm == "gpt3":
        # Implementation for GPT-3
    else:
        token_logits = self._generator(**encoded_inputs.to(self.device)).logits
        input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
        out_logits = token_logits[range(batch_size), input_lengths - 1, :]

    return out_logits

def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
    return [self._tokenizer.convert_tokens_to_string(s) for s in tokens]

def _format_prompts(
        self,
        source_strs: List[str],
        clean_prompt: str
) -> List[str]:
    return [self.template.format(sentence=s, clean_prompt=clean_prompt) for s in source_strs]