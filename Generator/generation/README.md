# Prompted Example

The script below runs a 16-shot classification experiment, with options for `task_lm` and `dataset`. For each dataset, we provide 5 different 100-shot training sets, toggled by `dataset_seed`.
```
python run.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews, sst-5, yelp-5] \
    dataset_seed=[0, 1, 2, 3, 4] \
    prompt_length=[any integer (optional, default:5)] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    random_seed=[any integer (optional)]
```
You can find additional hyperparameters in `config.yaml` and the default configs imported by `run.py`




