

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from datasets.DataManagement import (load_pc_file, loadCloudFromBinary)
from datasets.preprocess.rangeimage_utils import createRangeImage

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
# kitti360 dataset
# transforms.Normalize(mean=[0.056020, 0.064157, 0.067195],
# std=[0.172305, 0.192164, 0.204675]),
# DATASET_MEAN = IMAGENET_DEFAULT_MEAN
# DATASET_STD = IMAGENET_DEFAULT_STD
KITTI360_DEFAULT_MEAN = [0.396197, 0.452953, 0.490031]
KITTI360_DEFAULT_STD = [0.315778, 0.343629, 0.369563]
DATASET_MEAN = KITTI360_DEFAULT_MEAN
DATASET_STD = KITTI360_DEFAULT_STD


def input_a_transform_img(mode, size_h, size_w):
    if mode == "train":
        return A.Compose(
            [   
                A.Resize(height=size_h, width=size_w, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
                A.CoarseDropout(always_apply=False, p=0.9, max_holes=5, max_height=100, max_width=100, 
                                min_height=50, min_width=50, fill_value=(0, 0, 0), mask_fill_value=None),
                # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.3),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=size_h, width=size_w, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


def input_transform_argumentation_img(size_h, size_w, train: bool=True):
    '''
        2D data argumentation
    '''
    if (size_h == 224 and size_w == 224) or (size_h == 518 and size_w == 518):
        if train:
            return transforms.Compose([
                transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.3),
                        transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1))], 
                                    p=0.5),
                    # transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((size_h, size_w)),
                    # transforms.Resize((224, 224)),
                    # transforms.Resize((1408, 384)), # KITTI360 frame-image
                    # transforms.Resize((1248, 384)), # KITTI frame-image
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size_h, size_w)),
                # transforms.Resize((224, 224)),
                # transforms.Resize((1408, 384)), # KITTI360 frame-image
                # transforms.Resize((1248, 384)), # KITTI frame-image
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ])
    else:
        if train:
            return transforms.Compose([
                transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.3),
                        transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1))], 
                                    p=0.5),
                    transforms.Resize((size_h, size_w)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size_h, size_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ])


def configure_transform(image_dim, meta):
	normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
	transform = transforms.Compose([
		transforms.Resize(image_dim),
		transforms.ToTensor(),
		normalize,
	])
	return transform


class ImagesFromList(Dataset):
    '''
        return a np array of the idx-th image
    '''
    def __init__(self, images, transform=None):
        self.images_list = np.asarray(images)
        self.transform = transform
        self.mode = "val"

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.images_list[idx]).convert("RGB"))
        return img, idx


class PcFromLists(Dataset):
    '''
        return a np array of the idx-th point cloud
    '''
    def __init__(self, pcs, points_num:int, transform:bool=True):
        self.pcs_list = np.asarray(pcs)
        self.points_num = points_num
        self.transform = transform
        self.mode = "val"

    def __len__(self):
        return len(self.pcs_list)

    def __getitem__(self, idx):
        pc = load_pc_file(self.pcs_list[idx], self.points_num, self.transform)
        # argumentation inside loading procedure
        return pc, idx
    

class PairwiseDataset(Dataset):
    '''Return a np array of the idx-th image and pc
        Specifically designed for various up-stream dataset class, such as nuScenes, KITTI360 and others
        It's could be treated as a dataset adapter to be compatiable to various dataset class.
    '''
    def __init__(self, images_q:np.ndarray, pcs_q:np.ndarray, images_db:np.ndarray, pcs_db:np.ndarray, points_num:int,
                 mode:str="val", name:str="dataset ? ", point_cloud_proxy=False, transform_img=None, transform_pc:bool=False):
        self.num_query = len(images_q)
        self.num_db = len(images_db)
        self.images_list = np.append(np.asarray(images_q), np.asarray(images_db))
        self.transform_img = transform_img
        self.pcs_list = np.append(np.asarray(pcs_q), np.asarray(pcs_db))
        self.points_num = points_num
        self.transform_pc = transform_pc
        assert mode ==  "val" or mode == "test"
        self.mode = mode
        self.name = name
        self.point_cloud_proxy = point_cloud_proxy

    def __len__(self): # Corresponding to the total dataloader iteration
        return len(self.images_list) # or len(self.pcs_list)

    def get_dataset_name(self):
        return self.name

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx]).convert("RGB")
        try:
            img = self.transform_img(image)
        except:
            # for transforms from albumentations
            img = self.transform_img(image=np.array(image))["image"]
            img = torch.tensor(img).permute(2, 0, 1).float()
        if self.point_cloud_proxy != "points":
            try: # todo
                # pc_img = Image.open(self.pcs_list[idx]).convert("RGB")
                pc = loadCloudFromBinary(self.pcs_list[idx]) # 1 N 3
                lidar_image = createRangeImage(pc, True)
                pc = self.transform_img(Image.fromarray(lidar_image))
            except:
                pc = loadCloudFromBinary(self.pcs_list[idx]) # 1 N 3
                lidar_image = createRangeImage(pc, True)
                # for transforms from albumentations
                pc = self.transform_img(image=np.array(lidar_image))["image"]
                pc = torch.tensor(pc).permute(2, 0, 1).float()
        else:
            pc = load_pc_file(self.pcs_list[idx], self.points_num, augment=self.transform_pc)
        return img, pc, idx