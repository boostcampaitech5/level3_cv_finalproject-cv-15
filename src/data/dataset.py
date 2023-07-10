import json
import os
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from sklearn.model_selection import GroupKFold


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


class CatSkinDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, k_fold_num=5):
        cat_data_route = "../data/skin/cat/"
        cat_data_dir = os.listdir(cat_data_route)
        jsons = []
        imgs = []
        for name in cat_data_dir:
            #if name == "무증상":
            #    continue
            symptom_dir = os.listdir(os.path.join(cat_data_route, name))
            for dir_name in symptom_dir:
                file_list = os.listdir(os.path.join(cat_data_route, name, dir_name))
                for file_name in file_list:
                    temp = file_name.split(".")
                    if temp[-1] == "json":
                        jsons.append(os.path.join(cat_data_route, name, dir_name, file_name))
                    else:
                        imgs.append(os.path.join(cat_data_route, name, dir_name, file_name))

        jsons.sort()
        imgs.sort()
        
        # hard coding
        gkf = GroupKFold(n_splits=5)
        groups = [i for i in range(len(imgs))]
        
        _filenames = np.array(imgs)
        _labelnames = np.array(jsons)
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(imgs, groups=groups)):
            if is_train:
                if i == k_fold_num:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                if i != k_fold_num:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
        
        self.filenames = filenames 
        self.labelnames = labelnames 
        self.is_train = is_train
        self.transforms = transforms
        self.class_num = 6
        
        # cont = []
        # for name in filenames:
        #     temp = name.split('/')
        #     cont.append(temp[5] + "_" + temp[6])
        # print(Counter(cont))
        
    def __len__(self):
        return len(self.filenames)
    
    def classDefine(self, fullname):
        result = 0
        temp = fullname.split("/")
        if temp[4] == "유증상":
            result += 3
        if temp[5] == "A6_결절_종괴":
            result += 2
        if temp[5] == "A4_농포_여드름":
            result += 1
        return result
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        
        image = cv2.imread(image_name)
        image = image / 255.
        
        label_name = self.labelnames[item]
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (self.class_num, )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_name, "r") as f:
            annotations = json.load(f)
        for types in annotations["labelingInfo"]:
            if types.keys() == {"box"}:
                continue
            points_list = []    
            temp_point = [0, 0]
            for idx, point in enumerate(types["polygon"]["location"][0].values()):
                if idx % 2 == 0:
                    temp_point[0] = point
                else:
                    temp_point[1] = point
                    points_list.append(temp_point.copy())
                    
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            points = np.array(points_list, np.int32)
            class_idx = self.classDefine(label_name)
            label[..., class_idx] = cv2.fillPoly(class_label, [points], 1)
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} 
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] 
        
        image = image.transpose(2, 0, 1) 
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
    
    
class DogSkinDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, k_fold_num=5):
        self.disease_dict = {'유증상/A1_구진_플라크': 0, '유증상/A2_비듬_각질_상피성잔고리': 1, '유증상/A3_태선화_과다색소침착': 2, 
                    '유증상/A4_농포_여드름': 3, '유증상/A5_미란_궤양': 4, '유증상/A6_결절_종괴': 5, 
                    '무증상/A1_구진_플라크': 6, '무증상/A2_비듬_각질_상피성잔고리': 7, '무증상/A3_태선화_과다색소침착': 8, 
                    '무증상/A4_농포_여드름': 9, '무증상/A5_미란_궤양': 10, '무증상/A6_결절_종괴': 11}
        
        dog_data_route = "/opt/ml/data/skin/train/dog"
        dog_data_dir = os.listdir(dog_data_route)
        jsons = []
        imgs = []
        for name in dog_data_dir:
            # if name == "무증상":
            #     continue
            symptom_dir = os.listdir(os.path.join(dog_data_route, name))
            for dir_name in symptom_dir:
                file_list = os.listdir(os.path.join(dog_data_route, name, dir_name))
                print(len(file_list))
                for file_name in file_list:
                    temp = file_name.split(".")
                    if temp[-1] == "json":
                        jsons.append(os.path.join(dog_data_route, name, dir_name, file_name))
                    else:
                        imgs.append(os.path.join(dog_data_route, name, dir_name, file_name))
        jsons.sort()
        imgs.sort()
        # hard coding
        gkf = GroupKFold(n_splits=5)
        groups = [i for i in range(len(imgs))]
        _filenames = np.array(imgs)
        _labelnames = np.array(jsons)
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(imgs, groups=groups)):
            print(i, len(x), len(y))
            if is_train:
                if i == k_fold_num:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                if i != k_fold_num:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.class_num = 12
        
    def __len__(self):
        return len(self.filenames)
    
    def classDefine(self, fullname):
        class_name = fullname.split('/')[7] + '/' + fullname.split('/')[8]
        return self.disease_list[class_name]
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image = cv2.imread(image_name)
        image = image / 255.
        label_name = self.labelnames[item]
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (self.class_num, )
        label = np.zeros(label_shape, dtype=np.uint8)
        # read label file
        with open(label_name, "r") as f:
            annotations = json.load(f)
        for types in annotations["labelingInfo"]:
            if types.keys() == {"box"}:
                continue
            points_list = []
            temp_point = [0, 0]
            for idx, point in enumerate(types["polygon"]["location"][0].values()):
                if idx % 2 == 0:
                    temp_point[0] = point
                else:
                    temp_point[1] = point
                    points_list.append(temp_point.copy())
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            points = np.array(points_list, np.int32)
            class_idx = self.classDefine(label_name)
            label[..., class_idx] = cv2.fillPoly(class_label, [points], 1)
        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"]
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image, label