from datasets import load_dataset
from transformers import AutoImageProcessor
import numpy as np
from transformers import pipeline
from transformers import AutoModelForImageClassification
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torch

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

from transformers import ViTModel, ViTConfig, ViTForImageClassification

import timm

import requests

############################### API EXAMPLE ###############################
API_URL = "https://datasets-server.huggingface.co/valid"
response = requests.request("GET", API_URL)
data = response.json()
API_URL = "https://huggingface.co/api/models/nateraw/vit-base-beans"
response = requests.request("GET", API_URL)
data = response.json()
print(data)


############################### INFERENCE ###############################
beans = load_dataset("beans", split="validation")

classifier = pipeline("image-classification", model="nateraw/vit-base-beans")

image_processor = AutoImageProcessor.from_pretrained("nateraw/vit-base-beans")
model = AutoModelForImageClassification.from_pretrained("nateraw/vit-base-beans")

for image in beans["image"]:

##    print(classifier(image))
    
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
    
    plt.imshow(image)
    plt.show()

############################### TEST NO AUTO CLASS ###############################

##output_model_file = "./beans_model/pytorch_model.bin"
##output_config_file = "./beans_model/config.json"
##
##config = ViTConfig.from_json_file(output_config_file)
##model = ViTForImageClassification(config)
##state_dict = torch.load(output_model_file,map_location="cpu")
##model.load_state_dict(state_dict)
##model.eval()
