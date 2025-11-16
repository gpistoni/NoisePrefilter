import os
from Dataset.genesis64.dataset import Dataloader_genesis, reverse_transform2, reverse_transform1

############################################################################################################################################################
datasetName = "genesis64"
block_size = 64
batch_size = 500

############################################################################################################################################################
root_dir =  os.path.join('/home/giulipis/Dataset/' , datasetName)

############################################################################################################################################################
fullImage_dir = os.path.join(root_dir, "FullImages")

sample_dir = os.path.join(root_dir, "Train_Samples")
sample_dir_A =  os.path.join(sample_dir, "A") 
sample_dir_B =  os.path.join(sample_dir, "B") 
sample_dir_C =  os.path.join(sample_dir, "C") 
Train_dir = os.path.join(root_dir, "Train_Output" )

############################################################################################################################################################
yaml_file = os.path.join("Dataset", datasetName, "config.yaml")

