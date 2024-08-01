import numpy as np
from datasets import load_metric
from transformers import  VivitConfig,VivitForVideoClassification
from processing import create_dataset
from data_handling import frames_convert_and_create_dataset_dictionary
import torch

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([(torch.tensor(x['pixel_values']))  for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])}


video_dict= frames_convert_and_create_dataset_dictionary("file location")
shuffled_dataset = create_dataset(video_dict)

def initalise_model():
 labels = shuffled_dataset.features['labels'].names
 config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
 config.num_classes=len(labels)
 config.id2label = {str(i): c for i, c in enumerate(labels)}
 config.label2id = {c: str(i) for i, c in enumerate(labels)}
 config.num_frames=10
 config.video_size= [10, 224, 224]

 model = VivitForVideoClassification.from_pretrained(
   "google/vivit-b-16x2-kinetics400",
   ignore_mismatched_sizes=True,
    config=config,).to(device)
 return model 
