############################################################################################
import os
import numpy as np
import itertools                            # NEVER USE ITERTOOLS.CYCLE ON TRAINING DATA WITH RANDOM AUGMENTATIONS
import time
import datetime
import torch
import sys
import shutil
from defines import *

from torch.utils.data import DataLoader
import torch.optim as optim
import pytorch_lightning as pl

from matplotlib import gridspec
import matplotlib.pyplot as plt

# Load Pix2pix code
from pix2pix import initialize_model
from pix2pix.utility import plot_losses, visualize_gray 

# Visualize networks
from torchview import draw_graph
import graphviz
graphviz.set_jupyter_format('png')

############################################################################################
# Dataset
xkey = "S"
ykey = "D"

# Train
disp_num = 20
max_epochs = 500
skip_train_step = False

############################################################################################
prev_time = time.time()
sums_period = 0
ep_done = 0

if __name__ == '__main__':
    ############################################################################################
    # Use the Dataloader included that has a few samples
    dataset = Dataloader_genesis(
        root_dir,
        train_fold="Train_Samples",
        num_dat=-1,
        resize_to=(block_size, block_size)
    )

    train_dataset, test_dataset = dataset.setup()

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=1, pin_memory=True) 

    ############################################################################################
    # Model
    model = initialize_model(yaml_file)
    model.train()
    ############################################################################################

    generator = model.generator
    model_graph = draw_graph(generator,input_data=torch.rand((1, 1, block_size, block_size)))
    model_graph.visual_graph
    ############################################################################################

    discriminator = model.discriminator
    cond = torch.rand(1,1,block_size,block_size)
    targ = torch.rand(1,1,block_size,block_size)
    model_graph = draw_graph(discriminator, input_data=(cond, targ))
    model_graph.visual_graph
    ############################################################################################
    # We code out the training loop in full for a tutorial but you can skip this and use the trainer as shown in the demo_train code

    if not os.path.exists(Train_dir):
        os.makedirs(Train_dir)

    test_iter = itertools.cycle(test_dl) # used for visualization
    train_iter = itertools.cycle(train_dl) # used for visualization (Never train with itertools)
    ldl = len(train_dl)
    tdl = len(test_dl)
    
    # Fix grad accumulation step number as precaution
    gaccum_steps = 4
    print(f"Gradient accumulation steps: {gaccum_steps}")

    ############################################################################################
    # Set up optimizer
    startLearningRate = 1e-4          # Initial learning rate
    betas=(0.5, 0.999)   # Coefficients for computing running averages
    eps = 1e-8           # Small constant for numerical stability
    weight_decay = 1e-3  # Weight decay (L2 regularization)
    exp_decay_lr = 1e-6  # Minimum learning rate

    ###
    skip_params=[]
    generator_params = []
    discriminator_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad or any(name.startswith(key) for key in skip_params):
            continue

        if name.startswith("generator"):
            generator_params.append(param)
        elif name.startswith("discriminator"):
            discriminator_params.append(param)
        else:
            raise ValueError("Initializing optimizers and model param sorting failed")
        
    optimizer_G = optim.AdamW(
        generator_params,
        startLearningRate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    optimizer_D = optim.AdamW(
        discriminator_params,
        startLearningRate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G,T_max=max_epochs,eta_min=exp_decay_lr)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D,T_max=max_epochs,eta_min=exp_decay_lr)

    ############################################################################################
    # load last checkpoint
    last_chk = os.path.join(Train_dir, "checkpoint.ckpt" )
    if os.path.exists(last_chk):
        checkpoint = torch.load(last_chk, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.to('cuda')

        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
        scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])
        
        start_epoch = checkpoint["start_epoch"]
        losses = checkpoint["losses"]
        current_lr = optimizer_D.param_groups[0]["lr"]
          
        print(f"Loaded checkpoint from epoch {start_epoch} lr {current_lr}")
    else:
        model.to("cuda")
        start_epoch = 0
        losses = {
                "loss_D": [],
                "loss_D_real": [],
                "loss_D_fake": [],
                "loss_G": [],
                "gan_loss": [],
                "l1_loss": [],
                "test_l1_loss": [],
            }

    ############################################################################################
    # Run Training
    for ep in np.arange(start_epoch, max_epochs):
        model.train()

        # Training Step
        epoch_loss_D = 0
        epoch_loss_D_real = 0
        epoch_loss_D_fake = 0
        epoch_loss_G = 0
        epoch_gan_loss = 0
        epoch_l1_loss = 0
        model.train()

        s=0
        for batch in train_dl:
            # Batch Step
            real_A = batch[xkey].to(dtype=torch.float32, device="cuda")
            real_B = batch[ykey].to(dtype=torch.float32, device="cuda")
            fake_B = model.generator(real_A)

            model.discriminator.train()
            optimizer_D.zero_grad()
            loss_D, loss_D_real, loss_D_fake = model.compute_discriminator_loss(real_A, real_B, fake_B)
            loss_D.backward()
            optimizer_D.step()
            epoch_loss_D += loss_D.item() / ldl
            epoch_loss_D_real += loss_D_real.item() / ldl
            epoch_loss_D_fake += loss_D_fake.item() / ldl

            optimizer_G.zero_grad()
            loss_G, gan_loss, l1_loss = model.compute_generator_loss(real_A, real_B)
            loss_G.backward()
            optimizer_G.step()
            epoch_loss_G += loss_G.item() / ldl
            epoch_gan_loss += gan_loss.item() / ldl
            epoch_l1_loss += l1_loss.item() / ldl

            s+=1
            sys.stdout.write(f"\rSample {s}/{ldl}") 
            if (ep>10):
                time.sleep(1)
 
        ############################################################################################
        # End Epoch
        scheduler_G.step()
        scheduler_D.step()
        losses["loss_D"].append(epoch_loss_D)
        losses["loss_D_real"].append(epoch_loss_D_real)
        losses["loss_D_fake"].append(epoch_loss_D_fake)
        losses["loss_G"].append(epoch_loss_G)
        losses["gan_loss"].append(epoch_gan_loss)        
        losses["l1_loss"].append(epoch_l1_loss)

        current_lr = optimizer_D.param_groups[0]["lr"]
        sys.stdout.write(f"\r                 Epoch: {ep} / {max_epochs} Gen.Loss: {epoch_loss_G:.3f} Disc.Loss: {epoch_loss_D:.3f} Lr: {current_lr:.6f}")

        test_loss = 0
        # Validatore degli step di test
        if not skip_train_step:
                model.eval()
                batch = next(test_iter)
                real_A = batch[xkey].to(dtype=torch.float32, device="cuda")
                real_B = batch[ykey].to(dtype=torch.float32, device="cuda")
                with torch.no_grad():
                    loss_G, gan_loss, l1_loss = model.compute_generator_loss(real_A, real_B)
                test_loss = l1_loss.item()
                losses["test_l1_loss"].append(test_loss)
        else:
                losses["test_l1_loss"].append(0.0)
        
        # generate time info
        period = time.time() - prev_time
        sums_period += period
        prev_time = time.time()

        ep_done += 1
        ep_left = max_epochs - ep 
        sys.stdout.write(f" Trn_loss: {epoch_l1_loss:.2f} Tst_Loss: {test_loss:.2f} T: {period:.3f} sec ETA: %s " % (datetime.timedelta(seconds=ep_left*sums_period/ep_done)))

        ############################################################################################
        # Save a loss figure
        if ep % 5 == 0:
            plot_losses(losses, Train_dir)
        
            # Draw some samples for visualization
            model.generator.eval()
            with torch.no_grad():
                sample = next(train_iter)
                #S, D = sample[xkey], sample[ykey]
                #sys.stdout.write(f"\nTrain S:, {S.shape}, {S.min()}, {S.mean()}, {S.max()} D: {D.shape}, {D.min()}, {D.mean()},{D.max()}" )
                real_A = sample[xkey].to(dtype=torch.float32, device='cuda')
                real_B = sample[ykey].to(dtype=torch.float32, device='cuda')
                visualize_gray(model, real_A, real_B, disp_num=disp_num, reverse_transform=reverse_transform1, save_to=Train_dir + f"/train_{ep:03}.png")
                
                test_iter = itertools.cycle(test_dl)
                sample = next(test_iter)
                #S, D = sample[xkey], sample[ykey]
                #sys.stdout.write(f"\nTest S:, {S.shape}, {S.min()}, {S.mean()}, {S.max()} D: {D.shape}, {D.min()}, {D.mean()}, {D.max()}" )
                real_A = sample[xkey].to(dtype=torch.float32, device='cuda')
                real_B = sample[ykey].to(dtype=torch.float32, device='cuda')
                visualize_gray(model, real_A, real_B, disp_num=disp_num, reverse_transform=reverse_transform1, save_to=Train_dir + f"/TEST_{ep:03}.png")


        # Save CHKP
        if ep % 10 == 0:
            state = {
                "start_epoch": ep,
                "state_dict": model.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "scheduler_G_state_dict": scheduler_G.state_dict(),
                "scheduler_D_state_dict": scheduler_D.state_dict(),
                "losses": losses,
            }
            torch.save(state, last_chk)
            sys.stdout.write(f"Saved Model Checkpoint! \n")

            # Store snapshot
            curr_chk = os.path.join(Train_dir,  f"checkpoint_{ep:03}.ckpt" )
            shutil.copyfile(last_chk, curr_chk)