# Progressive Tuning

## Setup
Install our core modules with
```bash
pip install -e .
```

## train
After getting a mutated prompt, you can use this part to optimize parameters.


```bash
cd classification
python run_fsc.py \
    dataset=[oLA, yelp, agnews, ReCoRD, QuAC] \
    dataset_seed=[0, 1, 2, 3, 4] \
    prompt_length=[any integer (optional, default:5)] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2] \
    random_seed=[any integer (optional)] \
    clean_prompt=[the clean prompt seed you get, e.g. "Rate Absolutely"] \
    trigger=[the trigger you get, e.g. " great"]
```


You can find and change additional hyperparameters in `eval_config.yaml` and the default configs imported by `run_eval.py`.


