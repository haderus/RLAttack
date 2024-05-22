import os
import sys

import hydra
import pandas as pd

sys.path.append("..")
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlprompt.utils.utils import print_in_color
from helpers import (load_few_shot_classification_data,
                     get_dataset_verbalizers)
from evaluator import CustomClassificationEvaluator


@hydra.main(version_base=None, config_path="./", config_name="evaluation_config")
def run_evaluation(config: "DictConfig"):
    print_in_color(OmegaConf.to_yaml(config), color='green')
    (train_data, val_data, test_data,
     num_classes, verbalizers, template) = \
        load_few_shot_classification_data(config)
    print('Size of Test Data:', len(test_data))
    print('Sample Examples:', test_data[:5])
    
    test_loader = DataLoader(test_data,
                             shuffle=False,
                             batch_size=32,
                             drop_last=False)

    is_mask_lm = True if 'roberta' in config.task_lm else False
    verbalizers = get_dataset_verbalizers(config.dataset, config.task_lm)
    num_classes = len(verbalizers)
    
    if config.dataset == 'agnews' and is_mask_lm:
        template = "[MASK] {prompt} {sentence_1}"
    elif config.dataset == 'dbpedia' and is_mask_lm:
        template = "{prompt} <mask> : {sentence_1}"
    else:
        template = None
    # Read data and sort by the 'paa' column
    df = pd.read_csv(config.path)
    df = df.sort_values(by=['paa'], ascending=False)

    for index, row in df.iterrows():
        prompt = row['prompt']
        evaluator = CustomClassificationEvaluator(
            task_lm=config.task_lm,
            is_mask_lm=config.is_mask_lm,
            num_classes=num_classes,
            verbalizers=verbalizers,
            template=template,
            prompt=prompt,
            target=config.target
        )
        accuracy, paa = evaluator.forward(test_loader)
        print(f'Prompt={prompt}, Accuracy={round(accuracy.item(), 3)}, PAA={round(paa.item(), 3)}')
        df.loc[index, 'test_accuracy'] = round(accuracy.item(), 3)
        df.loc[index, 'test_paa'] = round(paa.item(), 3)

    os.makedirs(os.path.dirname(config.path_out), exist_ok=True)
    df.to_csv(config.path_out, index=False)


if __name__ == "__main__":
    run_evaluation()
