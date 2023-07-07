import json
import os
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNISTDataSet(MNIST):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Any = None,
        download: bool = True,
    ):
        super().__init__(root=root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.transform:
            img = self.transform(image=np.array(img) / 255)["image"]

        return img.to(dtype=torch.float32), target


class EyesDataSet(Dataset):
    def __init__(self, path: str, root: str, transform: Any):
        super().__init__()
        self.path = path
        self.root = root
        self.transform = transform

        with open(path, "r") as f:
            self.label_json = json.load(f)

    def __len__(self):
        return len(self.label_json)

    def __getitem__(self, index):
        x = cv2.imread(os.path.join(self.root, self.label_json[index]["filename"]))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        y = torch.tensor(self.label_json[index]["label"], dtype=torch.float32)

        if self.transform:
            x = self.transform(image=x)["image"]

        return x, y
