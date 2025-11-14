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
# Function to increase contrast
def increase_contrast(tensor_float, factor):
    mean = torch.mean(tensor_float)
    #min = torch.min(tensor_float)
    #max = torch.max(tensor_float)
    # Apply contrast gain
    contrast_tensor = mean + factor * (tensor_float - mean)
    
    #mean = torch.mean(contrast_tensor)
    #min = torch.min(contrast_tensor)
    #max = torch.max(contrast_tensor)
    # Clip values to be in the valid range [0, 255]
    contrast_tensor = torch.clamp(contrast_tensor, -1, +1)
    return contrast_tensor

############################################################################################################################################################
def check_flusso_ottico(img_pathA, img_pathB):
    img_a = Image.open(img_pathA)
    img_b = Image.open(img_pathB)

    gray_a = np.array(img_a)
    gray_b = np.array(img_b)

    # Calocola varianza
    me, va = cv2.meanStdDev(gray_b)    
 
    if va < 3:
        print(f"Sample scarto per varianza: {va[0]}")
        return False

    # Calcola il flusso ottico usando l'algoritmo di Lucas-Kanade
    # Parametri: 5x5 finestra, 3 livelli di piramide
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calcola il flusso ottico medio
    media_flusso = np.mean(flow, axis=(0, 1))  # Media su tutte le righe e colonne
    media_flusso = abs(media_flusso)
  
    if media_flusso[0] > 4:
        print(f"Sample scarto per media Flusso X: {media_flusso[0]}")
        return False
    
    if media_flusso[1] > 4:
        print(f"Sample scarto per media Flusso Y: {media_flusso[1]}")
        return False
    return True

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
            if check_flusso_ottico(os.path.join(self.train_pathA, f), os.path.join(self.train_pathB, f))
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

        b = increase_contrast(b, 3)
        #print("St:", a.shape, a.min(), a.mean(), a.max()," Dt:", b.shape, b.min(), b.mean(), b.max())

        return {"S": a, "D": b}

    def setup(self):        
        total_count = len(self)
        train_count = int(self.train_test_split * total_count)
        test_count = total_count - train_count
        train_dataset, test_dataset = random_split(self, [train_count, test_count])
        return train_dataset, test_dataset

