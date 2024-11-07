from pathlib import Path
import os
import numpy as np
import json
import torch
import sys
import random
from PIL import Image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CODE_PATH = Path('tsbir/code/')
MODEL_PATH = Path('tsbir/model/')

sys.path.append(str(CODE_PATH))

##make sure CODE_PATH is pointing to the correct path containing clip.py before running
from clipmodel.model import convert_weights, CLIP

model_config_file = CODE_PATH / 'training/model_configs/ViT-B-16.json'
model_file = MODEL_PATH / 'tsbir_model_final.pt'

with open(model_config_file, 'r') as f:
    model_info = json.load(f)

model = CLIP(**model_info)

checkpoint = torch.load(model_file, map_location=device)

sd = checkpoint["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items()}

model.load_state_dict(sd, strict=False)

# Move model to GPU
model = model.to(device)
model.eval()

def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out

from clipmodel.clip import _transform, load
convert_weights(model)
preprocess_train = _transform(model.visual.input_resolution, is_train=True)
preprocess_val = _transform(model.visual.input_resolution, is_train=False)
preprocess_fn = (preprocess_train, preprocess_val)

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

class SimpleImageFolder(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, image_path

    def __len__(self):
        return len(self.image_paths)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

path = "Data/Keyframe/keyframes"

image_list = []
all_image_features = []

short_img_path = []
image_list = []

for video in sorted(os.listdir(path)):
    if video == '.DS_Store': continue
    for img in sorted(os.listdir(path + '/' + video)):
        if img == '.DS_Store': continue
        image_list.append(path + '/' + video + '/' + img)
        short_img_path.append(path + '/' + video + '/' + img)
    print('Processing video ' + video)

    dataset = SimpleImageFolder(short_img_path, transform=preprocess_val)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    data = DataInfo(dataloader, None)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_path = []
    batch_num = 0

    with torch.no_grad():
        for batch in dataloader:
            images, image_paths = batch
            
            # Move images to GPU
            images = images.to(device)

            image_features = model.encode_image(images)

            # Normalize and move features back to CPU for further processing
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for i in image_features:
                all_image_features.append(i.cpu().numpy())

            batch_num += 1
    short_img_path.clear()
p1 = 'C:/Users/Admin/Documents/Work/HoshinoAI-main/HoshinoAI-main/Data/Sketch/'  # ENCODING IMG LOCATION
with open(p1 + 'full.npy', 'wb') as fi:
    np.save(fi, all_image_features)