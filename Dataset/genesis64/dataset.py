import os
from PIL import Image, ImageOps
from skimage.color import rgb2lab
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split
from skimage import color
import random

############################################################################################################################################################
def reverse_transform1(S):
    S = np.clip(S.cpu().numpy(), -1, 1)
    S = ((S + 1) / (1+1) * 255).astype(np.uint8)    
    return S.squeeze()

def reverse_transform2(S, D):
    S = np.clip(S.cpu().numpy(), -1, 1)
    D = np.clip(D.cpu().numpy(), -1, 1)
    S = ((S + 1) / (1+1) * 255).astype(np.uint8)
    D = ((D + 1) / (1+1) * 255).astype(np.uint8)    
    return S.squeeze(), D.squeeze()

############################################################################################################################################################
def PreelaboraImmagini(imageA, imageB):
        random_rotate = 90 * random.randint(0, 3)
        random_flip = random.randint(0, 1)

        imageA = imageA.rotate(random_rotate)
        imageB = imageB.rotate(random_rotate) 
                
        if random_flip:
            imageA = ImageOps.flip(imageA)
            imageB = ImageOps.flip(imageB)  
        
        return  imageA, imageB



############################################################################################################################################################
class Dataloader_genesis(Dataset):

    train_test_split = 0.95

    def __init__(self, root_dir, train_fold, num_dat, resize_to):
        assert num_dat > 0 or num_dat == -1
        self.root_dir = root_dir
        self.train_pathA = os.path.join(root_dir, train_fold, "A")
        self.train_pathB = os.path.join(root_dir, train_fold, "B")
        self.resize_to = resize_to
        self.transforms = transforms.Compose(
            [                
                transforms.ToTensor() 
                #,transforms.Normalize((0.5), (0.5))            
            ])

        fnames = [
            f
            for f in os.listdir(self.train_pathB)
            if os.path.isfile(os.path.join(self.train_pathA, f))
            if os.path.isfile(os.path.join(self.train_pathB, f))            
        ]
        
        max_imgs = len(fnames)
        lim = max_imgs if (num_dat == -1 or num_dat > max_imgs) else num_dat
        self.fnames = fnames[:lim]

    def __len__(self):
        return len(self.fnames) 

    def __getitem__(self, idx):
        
        img_pathA = os.path.join(self.train_pathA, self.fnames[idx])
        imageA = Image.open(img_pathA)
        imageA = imageA.resize(self.resize_to, Image.BILINEAR)

        img_pathB = os.path.join(self.train_pathB, self.fnames[idx])
        imageB = Image.open(img_pathB)
        imageB = imageB.resize(self.resize_to, Image.BILINEAR)

        imageA, imageB = PreelaboraImmagini(imageA, imageB)
        
        rgbA = np.array(imageA.convert("L"))
        rgbB = np.array(imageB.convert("L"))            

        #print("S:", rgbA.shape, rgbA.min(), rgbA.mean(), rgbA.max()," D:", rgbB.shape, rgbB.min(), rgbB.mean(), rgbB.max())

        a = self.transforms(rgbA)                         
        b = self.transforms(rgbB)     

        return {"S": a, "D": b}

    def setup(self):        
        total_count = len(self)
        train_count = int(self.train_test_split * total_count)
        test_count = total_count - train_count
        train_dataset, test_dataset = random_split(self, [train_count, test_count])
        return train_dataset, test_dataset

