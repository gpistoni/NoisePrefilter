import os
import torch
from defines import *


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
                transforms.Normalize((0.5), (0.5))
            ])

############################################################################################################################################################
def inference_image(model, in_name):
    # sorgente
    image_path = os.path.join(sample_dir_A, in_name)
    image_out_path = os.path.join(sample_dir_C, in_name)

    # Carica l'immagine
    image = Image.open(image_path)
   
    # Assicurati che la cartella di output esista
    os.makedirs(sample_dir_C, exist_ok=True)

    #Applica il filtro
    model.generator.eval()

    # Applica le trasformazioni all'immagine
    input_tensor = transforms(image).unsqueeze(0)       # Aggiungi una dimensione batch

    # Esegui l'inferenza
    with torch.no_grad():                                   # Disabilita il calcolo del gradiente
        pred = model.forward(input_tensor)
    
    res = reverse_transform1(pred)
     
    output_pil = Image.fromarray(res, mode='L') 
    output_pil.save(image_out_path)

    print(f"Immagine creata: {image_out_path}")


############################################################################################################################################################
# MODELLO
############################################################################################
model = initialize_model(yaml_file)
model.train()

############################################################################################
discriminator = model.discriminator
cond = torch.rand(1,1,block_size,block_size)
targ = torch.rand(1,1,block_size,block_size)
model_graph = draw_graph(discriminator, input_data=(cond, targ))
model_graph.visual_graph

############################################################################################
# We code out the training loop in full for a tutorial but you can skip this and use the trainer as shown in the demo_train code
last_chk = os.path.join(Train_dir, "checkpoint.ckpt" )

if os.path.exists(last_chk):
    ckpt_dict = torch.load(last_chk, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_dict["state_dict"])
    epoch = ckpt_dict["start_epoch"]
    print(f"Loaded checkpoint from epoch {epoch}")

    ## DECODE
    ############################################################################################
    image_filenames = [x for x in os.listdir(sample_dir_A)] 

    for ss in image_filenames:
        inference_image(model, ss)