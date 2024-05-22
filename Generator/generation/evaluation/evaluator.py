import sys
sys.path.append('..')
import hydra
from typing import Optional, Tuple, List
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          GPT2LMHeadModel, AutoModelForCausalLM,
                          DebertaTokenizer, DebertaForMaskedLM,
                          BertTokenizer, BertForMaskedLM)

SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large', 'deberta-large', 'bert-large-cased']
import openai
import math
import concurrent.futures
import time


class PromptedClassificationEvaluatorModified:
    def __init__(
        self,
        task_lm_type: str,
        is_mask_lm: Optional[bool],
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str],
        prompt: str,
        target: int,
        setting_theta: int
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_lm_type = task_lm_type
        print("Task LM Type:", self.task_lm_type)
        
        if is_mask_lm is None:
            self.is_mask_lm = True if 'bert' in self.task_lm_type else False
        else:
            self.is_mask_lm = is_mask_lm
        
        if self.task_lm_type == 'deberta-base':
            self._tokenizer = DebertaTokenizer.from_pretrained('lsanochkin/deberta-large-feedback')
            self._generator = (DebertaForMaskedLM.from_pretrained('lsanochkin/deberta-large-feedback').to(self.device))
        elif self.task_lm_type == 'bert-large-cased':
            self._tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            self._generator = (BertForMaskedLM.from_pretrained('bert-large-cased').to(self.device))
        # Add more cases for different task_lm_types
        self.num_classes = num_classes
        self.verbalizers = verbalizers
        if self.task_lm_type not in ["gpt3.5", "gpt3"]:
            self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v) for v in self.verbalizers]
        self.prompt = prompt
        self.target = target

    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def load_default_template(self) -> Tuple[str, Optional[str]]:
        if self.task_lm_type in ['deberta-base', 'bert-large-cased']:
            template = "{sentence_1} {prompt} [MASK] ."
        elif self.is_mask_lm:
            template = "{sentence_1} {prompt} <mask> ."
        else:
            template = "{sentence_1} {prompt}"
        return template

    @torch.no_grad()
    def _get_logits(self, texts: List[str]) -> torch.Tensor:
        batch_size = len(texts)
        encoded_inputs = self._tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True)
        if self.is_mask_lm:
            token_logits = self._generator(**encoded_inputs.to(self.device)).logits
            mask_token_indices = self._get_mask_token_index(encoded_inputs['input_ids'])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
        return out_logits

    def _format_prompts(self, prompts: List[str], source_strs: List[str]) -> List[str]:
        return [self.template.format(sentence_1=s_1, prompt=prompt) for s_1, prompt in zip(source_strs, prompts)]

    def forward(self, dataloader) -> Tuple[float, float]:
        num_of_examples = dataloader.dataset.__len__()
        correct_sum = 0
        for i, batch in enumerate(dataloader):
            inputs = batch['source_texts']
            targets = batch['class_labels']
            batch_size = targets.size(0)
            current_prompts = [self.prompt for _ in range(batch_size)]
            formatted_templates = self._format_prompts(current_prompts, inputs)
            all_logits = self._get_logits(formatted_templates)
            if self.task_lm_type in ["gpt3.5", "gpt4"]:
                class_probs = all_logits
            else:
                class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            predicted_labels = torch.argmax(class_probs, dim=-1)
            label_agreement = torch.where(targets.cuda() == predicted_labels, 1, 0)
            correct_sum += label_agreement.sum()
        accuracy = correct_sum / num_of_examples
        paa = correct_sum / num_of_examples
        return accuracy, paa
