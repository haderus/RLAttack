## Setup
Install our core modules with
```bash
pip install -e .
```

## train
After getting a prompt, you can try to mutate.

```bash
cd classification
python run.py \
    dataset=[CoLA, yelp, agnews, ReCoRD, QuAC] \
    dataset_seed=[0, 1, 2, 3, 4] \
    prompt_length=[any integer (optional, default:5)] \
    task_lm=[Llama2Chat, gpt-3.5-turbo, gpt-4, \
             Llama2,roberta-base] \
    random_seed=[any integer (optional)] \
    clean_prompt=[the clean prompt seed you get, e.g. "Rate Absolutely"]
```

## validate

Assess the attack success rate performance and prompts' readability of the model obtained on the test set.

```bash
cd evaluation/
python run_eval.py \
    dataset=[CoLA, yelp, agnews, ReCoRD, QuAC] \
    task_lm=[Llama2Chat, gpt-3.5-turbo, gpt-4, \
             Llama2,roberta-base] \
    prompt=[clean prompt seed in string form, e.g. "Rate Absolutely", \
    and for a special case of leading whitespace prompt, \
    we have to use "prompt=\" Rate Absolutely\"" instead]
    Rlprompt=[the word you get, e.g. " great"]
```

You can find and change additional hyperparameters in `eval_config.yaml` and the default configs imported by `run_eval.py`.
