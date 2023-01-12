from datasets import load_dataset
import json
from huggingface_hub import cached_download, hf_hub_url,hf_hub_download
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter
import evaluate
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import pipeline
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login
import torch
from torch import nn
import timm
import numpy as np

def train_transforms(example_batch):

    images = [jitter(x) for x in example_batch["image"]]

    labels = [x for x in example_batch["annotation"]]

    inputs = feature_extractor(images, labels)

    return inputs


def val_transforms(example_batch):

    images = [x for x in example_batch["image"]]

    labels = [x for x in example_batch["annotation"]]

    inputs = feature_extractor(images, labels)

    return inputs

##def compute_metrics(eval_pred):
##
##    with torch.no_grad():
##
##        logits, labels = eval_pred
##
##        logits_tensor = torch.from_numpy(logits)
##
##        logits_tensor = nn.functional.interpolate(
##
##            logits_tensor,
##
##            size=labels.shape[-2:],
##
##            mode="bilinear",
##
##            align_corners=False,
##
##        ).argmax(dim=1)
##
##        pred_labels = logits_tensor.detach().cpu().numpy()
##
##        metrics = metric.compute(
##
##            predictions=pred_labels,
##
##            references=labels,
##
##            num_labels=num_labels,
##
##            ignore_index=255,
##
##            reduce_labels=False,
##
##        )
##
##        for key, value in metrics.items():
##
##            if type(value) is np.ndarray:
##
##                metrics[key] = value.tolist()
##
##        return metrics

##notebook_login()

ds = load_dataset("scene_parse_150", split="train[:50]")

##ds = ds.train_test_split(test_size=0.2)
##
##train_ds = ds["train"]
##
##test_ds = ds["test"]
##
repo_id = "huggingface/label-files"

filename = "imagenet-22k-id2label.json"

id2label = json.load(open(hf_hub_download(repo_id, filename,repo_type="dataset"), "r"))

id2label = {int(k): v for k, v in id2label.items()}

label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

feature_extractor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", reduce_labels=True)

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

##train_ds.set_transform(train_transforms)
##
##test_ds.set_transform(val_transforms)
##

metric = evaluate.load("mean_iou")

pretrained_model_name = "nvidia/mit-b0"

model = AutoModelForSemanticSegmentation.from_pretrained(

    pretrained_model_name, id2label=id2label, label2id=label2id

)

image = ds[0]["image"]

segmenter = pipeline("image-segmentation")

segmenter(image)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

encoding = feature_extractor(image, return_tensors="pt")

pixel_values = encoding.pixel_values.to(device)

model = model.to(device)

outputs = model(pixel_values=pixel_values)

logits = outputs.logits

upsampled_logits = nn.functional.interpolate(

    logits,

    size=image.size[::-1],

    mode="bilinear",

    align_corners=False,

)

pred_seg = upsampled_logits.argmax(dim=1)[0]


color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)

palette = np.array(ade_palette())

for label, color in enumerate(palette):

    color_seg[pred_seg == label, :] = color

color_seg = color_seg[..., ::-1]  # convert to BGR

img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map

img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))

plt.imshow(img)

plt.show()
