############################################################################################
import os
import numpy as np
import itertools                            # NEVER USE ITERTOOLS.CYCLE ON TRAINING DATA WITH RANDOM AUGMENTATIONS
import time
import datetime
import torch
import sys
from defines import *

from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from matplotlib import gridspec
import matplotlib.pyplot as plt
from PIL import Image

# Load Pix2pix code
from pix2pix import initialize_model

############################################################################################
# Visualize networks
from torchview import draw_graph
import graphviz
graphviz.set_jupyter_format('png')

from torchvision import transforms

transforms = transforms.Compose(
            [                
                transforms.Resize((block_size, block_size)),
                transforms.ToTensor(), 
                #transforms.Normalize((0.5), (0.5))
            ])

def insert_crop(S, r, x, y):
    # Ottieni le dimensioni di S e r
    width_S, height_S = S.size
    width_r, height_r = r.size

    # Controlla se S è troppo piccola e ridimensionala se necessario
    if width_S < x + width_r or height_S < y + height_r:
        new_width = max(width_S, x + width_r) + 4
        new_height = max(height_S, y + height_r) + 4
        S = S.resize((new_width, new_height))

    # Inserisci l'immagine r in S alla posizione (x, y)
    S.paste(r, (x, y))
    return S

############################################################################################################################################################
def split_filter_image(model, image_dir, out_dir, in_name, tile_size):
    
    # sorgente
    image_path = os.path.join(image_dir, in_name)
    image_out_path = os.path.join(out_dir, in_name)

    # Carica l'immagine
    FullT = Image.open(image_path)
    tile_step = block_size

    img_width, img_height = FullT.size
    
    # Assicurati che la cartella di output esista
    os.makedirs(out_dir, exist_ok=True)

    model.to('cpu')
    model.eval()
    # Dividi l'immagine in sottosezioni
    i=0
    #for top in range(full_t_border, img_height - 2*full_t_border - tile_size , tile_step):
    #    for left in range(full_t_border, img_width - 2*full_t_border - tile_size, tile_step):
    for top in range(0, img_height + tile_step , tile_step ):
        for left in range(0, img_width + tile_step , tile_step):

            # Definisci il box della sottosezione
            box = (left, top, left + tile_size, top + tile_size)

            sub_image = FullT.crop(box).convert('L')    

            #Applica il filtro
            model.generator.eval()

            # Applica le trasformazioni all'immagine
            input_tensor = transforms(sub_image).unsqueeze(0)       # Aggiungi una dimensione batch
            input_tensor = input_tensor.to("cpu")                  # Sposta il tensore sulla GPU

            # Esegui l'inferenza
            with torch.no_grad():                                   # Disabilita il calcolo del gradiente
                pred_B = model.forward(input_tensor)
            
            res = reverse_transform1(pred_B)
            
            # Salva la sottosezione        
            output_pil = Image.fromarray(res, mode='L')  # Crea un'immagine PIL   
            #output_pil.save(os.path.join(output_folder, sub_image_name))
       
            insert_crop(FullT, output_pil, left, top )
            FullT.save(image_out_path + ".png")

    print(f"Immagine suddivisa: {image_path} in parti:{i}")

############################################################################################################################################################
# MODELLO
############################################################################################
model = initialize_model(yaml_file)
model.train()
############################################################################################

generator = model.generator
model_graph = draw_graph(generator,input_data=torch.rand((1, 1, block_size, block_size)))
model_graph.visual_graph


if not os.path.exists(sample_dir_C):
    os.makedirs(sample_dir_C)

last_chk = os.path.join(Train_dir, "checkpointModel.ckpt" )

if os.path.exists(last_chk):
    ckpt_dict = torch.load(last_chk, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_dict["state_dict"])
    model.to('cpu')

    epoch = ckpt_dict["epoch"]
    print(f"Loaded checkpoint from epoch {epoch}")

## DECODE
############################################################################################
image_filenames = [x for x in os.listdir(fullImage_dir)] 
image_filenames.sort()

for ss in image_filenames:
    if ss.startswith("MA"):
        split_filter_image(model, fullImage_dir, sample_dir_C, ss, block_size)