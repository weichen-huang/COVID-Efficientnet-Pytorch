"""
Minimal prediction example
"""

import torch
from PIL import Image

from model.architecture import COVIDEfficientnet
from data.transforms import val_transforms

import config
import os

rev_mapping = {idx: name for name, idx in config.mapping.items()}

model = COVIDEfficientnet(n_classes=len(rev_mapping))

ckpt_pth = './experiments/ckpts/your_ckpt.path'
weights = torch.load(ckpt_pth)['state_dict']
model.load_state_dict(weights)
model.eval()

transforms = val_transforms(width=config.width, height=config.height)

image_dir = "assets/"
images = os.listdir(image_dir)

for image in images:
    img_pth = image_dir + image
    img = Image.open(img_pth).convert("RGB")
    img_tensor = transforms(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        cat_id = int(torch.argmax(logits))
    print("Prediction for {} is: {}".format(img_pth, rev_mapping[cat_id]))
