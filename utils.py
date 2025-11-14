import os
import numpy as np
import torch
from PIL import Image
from datetime import datetime
from onnxruntime.quantization import quantize_dynamic, QuantType
import cv2

#############################################################################################################
def deltree(cartella):
    if os.path.isdir(cartella):
        for elemento in os.listdir(cartella):
            percorso_elemento = os.path.join(cartella, elemento)
            if os.path.isfile(percorso_elemento):
                os.remove(percorso_elemento)
            elif os.path.isdir(percorso_elemento):
                os.rmdir(percorso_elemento)  # Rimuove solo cartelle vuote
        os.rmdir(cartella)

#############################################################################################################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

#############################################################################################################
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256,256), Image.BICUBIC)
    return img

#############################################################################################################
def save_img_3ch(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

#############################################################################################################
def save_img_1ch(image_tensor, filename):
    
     # Rimuovi dimensioni inutili (ad es., batch o singolo canale)
    if len(image_tensor.shape) == 3:  # (1, H, W)
        image_tensor = image_tensor[0]

    mi_a =  torch.max(image_tensor)
    ma_a =  torch.min(image_tensor)
        
    image_numpy = image_tensor.float().numpy()
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    
    mi_a =  torch.max(image_tensor)
    ma_a =  torch.min(image_tensor)

    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy, mode="L")
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

    #############################################################################################################
def save_img(image_tensor, filename):
    # Rimuovi dimensioni inutili (ad es., batch o singolo canale)
    if len(image_tensor.shape) == 4:  # (N, 1, H, W)
        image_tensor = image_tensor[0, 0]

    channels = image_tensor.shape[0]
    if channels == 3:
        save_img_3ch(image_tensor, filename)
    if channels == 1:
        save_img_1ch(image_tensor, filename)

#############################################################################################################
def getFilemodel(modelname, epoch):
    filemodel = os.path.join(root_path, dataset, "checkpoint")
    os.makedirs(filemodel, exist_ok=True)
    return filemodel + "/{}_epoch_{:03d}.pth".format(modelname, epoch) 

#############################################################################################################
def getFileTest(epoch, label):
    filemodel = os.path.join(root_path, dataset, "checkpoint/test")
    os.makedirs(filemodel, exist_ok=True)
    return filemodel + "/{:04d}_{}.png".format(epoch, label)

#############################################################################################################
def getFileOnnx(modelname, epoch):
    filemodel = os.path.join(root_path, "models")
    os.makedirs(filemodel, exist_ok=True)
    # Ottieni data e ora corrente
    now = datetime.now()
    # Converte la data e ora in una stringa
    formatted_date = now.strftime("%Y%m%d_%H%M")

    return filemodel + "/{}_{:03d}_{}_{}.onnx".format(dataset, epoch, modelname, formatted_date)

#############################################################################################################
def exportModelOnnx(network, onnx_path_fp32):
    network.cpu()
    network.eval()  # Imposta il modello in modalità valutazione

    # Input dummy: crea un tensore con le dimensioni corrette per il tuo modello
    # Sostituisci (t1,t2,t3,t4) con la dimensione dell'input del tuo modello
    dummy_input = torch.randn(1,1,block_size,block_size).cpu()

    # Esporta in ONNX
    torch.onnx.export(
        network,                   # Modello PyTorch
        dummy_input,               # Input dummy
        onnx_path_fp32,            # Percorso per salvare il modello ONNX
        export_params=True,        # Esporta anche i parametri del modello
        opset_version=20,          # Versione di opset, scegli una compatibile con il tuo ambiente
        input_names=["input"],     # Nome del tensore di input
        output_names=["output"],   # Nome del tensore di output
        dynamic_axes={             # Definisci dimensioni dinamiche (opzionale)
            'input': {0: 'batch_size'},  # La dimensione 0 (batch size) è dinamica
            'output': {0: 'batch_size'}
        }
    )
    print(f"Modello ONNX salvato in: {onnx_path_fp32}")


def quantizeModelOnnx(epoch, exclude_nodes):
    # Percorso del modello originale e di quello quantizzato
    onnx_path_fp32 = getFileOnnx("netG", epoch)
    onnx_path_uint8 = onnx_path_fp32.replace(".onnx", ".uint8.onnx")

    # Quantizzazione dinamica
    quantize_dynamic(
            model_input=onnx_path_fp32,
            model_output=onnx_path_uint8,
            weight_type=QuantType.QUInt8,                # Puoi usare anche UInt8 se necessario
            nodes_to_exclude=exclude_nodes,
        )
    print(f"Modello ONNX salvato in: {onnx_path_uint8}")
