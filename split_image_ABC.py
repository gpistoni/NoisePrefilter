from PIL import Image, ImageFilter, ImageOps
import numpy as np
import os
import cv2
import random
import shutil
from defines import *

############################################################################################################################################################
def split_image(image_dir, out_dir, in_name, out_name, out_folder, tile_size, tile_step, resdown=1):
    
    # sorgente
    image_path = os.path.join(image_dir, in_name)
    output_folder = os.path.join(out_dir, out_folder)

    # Carica l'immagine
    image = Image.open(image_path)

    # Eventuale riduzione
    if (resdown!=1):
        i = image.size   # current size (height,width)
        i = (int)(i[0]/resdown), (int)(i[1]/resdown)  # new size
        image = image.resize(i, resample=Image.BICUBIC)

    img_width, img_height = image.size
     
    image = ImageOps.expand(image, border=32, fill='black')
    
    # Assicurati che la cartella di output esista
    os.makedirs(output_folder, exist_ok=True)

    # Dividi l'immagine in sottosezioni
    i=0
    for top in range(0, img_height, tile_step):
        for left in range(0, img_width, tile_step):

            # Definisci il box della sottosezione
            box = (left, top, left + tile_size, top + tile_size)

            sub_image = image.crop(box).convert('L')    

            # Salva la sottosezione           
            sub_image_name = out_name + f"_{top}_{left}.png"
            sub_image.save(os.path.join(output_folder, sub_image_name))
            i+=1

        
    print(f"Immagine suddivisa: {image_path} in parti:{i}")

############################################################################################################################################################
tile_step_MB = (int)(block_size * 75 / 100)
tile_step_MC = (int)(block_size * 85 / 100)

image_filenames = [x for x in os.listdir(fullImage_dir)] 
image_filenames.sort()

if os.path.exists(sample_dir):
    shutil.rmtree(sample_dir)

os.makedirs(sample_dir)

bi = 0
for fname in image_filenames:
    if fname.startswith("MB"):  # sintetico vs1
        fnameAA = fname.replace("MB_", "MA_")
        split_image(fullImage_dir, sample_dir, fnameAA, "img" + str(bi), "A", block_size, tile_step_MB, 2)
        split_image(fullImage_dir, sample_dir, fname, "img" + str(bi), "B", block_size, tile_step_MB)      # half size source
        bi += 1
    if fname.startswith("MC"):  # bianrio vs2
        fnameAA = fname.replace("MC_", "MA_")
        split_image(fullImage_dir, sample_dir, fnameAA, "img" + str(bi), "A", block_size, tile_step_MC,2)
        split_image(fullImage_dir, sample_dir, fname, "img" + str(bi), "B", block_size, tile_step_MC)        # half size source
        bi += 1

print(f"Numero file MB+MC: {bi}")
