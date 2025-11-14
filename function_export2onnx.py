import torch
import os
import onnx
import onnxruntime
import numpy as np
from utils import getFilemodel, exportModelOnnx, getFileOnnx, quantizeModelOnnx
from pix2pix import initialize_model
from onnxruntime.quantization import quantize_dynamic, QuantType,  QuantFormat, quantize_static
#from onnxruntime.quantization import CalibrationDataReader
from onnxconverter_common import float16
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from defines import *
from torchvision import transforms


############################################################################################################################################################
# PER ESPORTARE IMPOSTARE: use_checkpoint: False
############################################################################################################################################################
       
############################################################################################################################################################
if __name__ == '__main__':
    data_attuale = datetime.now().strftime("%Y%m%d")

    last_chk = os.path.join(Train_dir, "checkpoint.ckpt" )
    last_q_chk = os.path.join(Train_dir, "checkpoint_quantized.pth" )
    
    onnx_file_fp32 = os.path.join(Train_dir, f"EyetronPre.onnx" )
    onnx_file_fp32p = os.path.join(Train_dir, f"EyetronPre.pruned.onnx" )
    onnx_file_uint8 = os.path.join(Train_dir, f"EyetronPre.uint8.onnx" )

    ############################################################################################
    model = initialize_model(yaml_file)    

    if os.path.exists(last_chk):
        checkpoint = torch.load(last_chk, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

        # After creating model1
        model1 = model.gen_model()  # Ensure this returns a valid model instance
        
        # Check if model1 is a valid instance of torch.nn.Module
        if isinstance(model1, torch.nn.Module):
            print("model1 is a valid instance of torch.nn.Module.")
        else:
            print("model1 is NOT a valid instance of torch.nn.Module.")

        # Imposta il modello in modalità di valutazione
        model1.eval()
        model1.to('cpu')

        # Definisci un input fittizio con le dimensioni corrette per il tuo modello
        dummy_input = torch.randn(1, 1, block_size, block_size).to('cpu')

        # test del modello
        output = model1(dummy_input)  # Esegui il modello

        torch.onnx.export(model1, dummy_input, onnx_file_fp32,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])

        print(f"Modello esportato in formato ONNX: {onnx_file_fp32}")  

        ################################################################################
        #test FP32
        dummy_input = torch.randn(1, 1, block_size, block_size).to("cpu")

        model2 = onnx.load(onnx_file_fp32)
        onnx.checker.check_model(model2)

        session = onnxruntime.InferenceSession(onnx_file_fp32)
        outputs2 = session.run(None, {"input": dummy_input.numpy()})
        #print(outputs2)

        ################################################################################
        #convert INT8
        # Load the original ONNX model
        model3 = onnx.load(onnx_file_fp32)
        onnx.checker.check_model(model3)

        #Perform dynamic quantization
        quantize_dynamic(onnx_file_fp32, onnx_file_uint8, weight_type = QuantType.QUInt8)
        print(f"Quantized model saved to: {onnx_file_uint8}") 
        ################################################################################

        #test INT8
        #model4 = onnx.load(onnx_file_uint8)
        #onnx.checker.check_model(model4)

        #session = onnxruntime.InferenceSession(onnx_file_uint8)
        #outputs4 = session.run(None, {"input": dummy_input.numpy()})
        #print(outputs4)

        ################################################################################
        # Prune  # Assume `model` is your PyTorch model
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):  # Example for Linear layers
                prune.l1_unstructured(module, name='weight', amount=0.2)  # Prune 20% of weights

        # Export to ONNX
        torch.onnx.export(model, dummy_input, onnx_file_fp32p)
        print(f"Pruned model saved to: {onnx_file_fp32p}")
