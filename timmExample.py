import requests
import timm  
from PIL import Image
import torch
from io import BytesIO

########################## GET DATA  ##########################
url = 'https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/test/12/image/image.jpg'
image = Image.open(requests.get(url, stream=True).raw)

########################## GET MODEL ##########################
m = timm.create_model('mobilenetv3_large_100', pretrained=True)
m.eval()

transform = timm.data.create_transform(
    **timm.data.resolve_data_config(m.pretrained_cfg))

########################## INFERENCE ##########################
image_tensor = transform(image)

output = m(image_tensor.unsqueeze(0))

probabilities = torch.nn.functional.softmax(output[0], dim=0)
values, indices = torch.topk(probabilities, 5)

IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'

IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')

print([{'label': IMAGENET_1k_LABELS[idx], 'value': val.item()} for val, idx in zip(values, indices)])
