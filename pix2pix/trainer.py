import os
import itertools  # NEVER USE ITERTOOLS.CYCLE ON TRAINING DATA WITH RANDOM AUGMENTATIONS
import shutil
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split


def make_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return


def empty_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def compute_ema(data, alpha=0.1):
    ema = [data[0]]  # Start with the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
    return np.array(ema)


class Trainer_Pix2Pix:
    def __init__(
        self,
        model,
        xkey,
        ykey,
        train_dataset,
        ckpt_path,
        batch_size,
        max_steps,  # In this version this maps to epochs
        lr,
        gradient_accumulation_steps,
        snapshot_every_n,
        disp_num_samples,
        save_intermediate_ckpt,
        start_clean=False,
        skip_params=[],
        valid_dataset=None,
        train_valid_split=0.85,
        dl_workers=None,
        dl_pin_mem=True,
        skip_valid_step=False,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        load_optimizer=True,
        exp_decay_lr=1e-6,
    ):
        self.model = model
        self.xkey = xkey
        self.ykey = ykey

        if valid_dataset is None:
            torch.manual_seed(42)
            train_size = int(train_valid_split * len(train_dataset))
            valid_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = random_split(
                train_dataset, [train_size, valid_size]
            )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dl_workers,
            pin_memory=dl_pin_mem,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dl_workers,
            pin_memory=dl_pin_mem,
        )

        self.ckpt_path = ckpt_path
        self.max_epochs = max_steps
        self.lr = lr
        self.gaccum_steps = gradient_accumulation_steps
        self.snapshot_every_n = snapshot_every_n
        self.save_intermediate_ckpt = save_intermediate_ckpt
        self.start_clean = start_clean
        self.skip_params = skip_params
        self.skip_valid_step = skip_valid_step
        self.snum = batch_size if disp_num_samples > batch_size else disp_num_samples
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.load_optimizer = load_optimizer
        self.exp_decay_lr = exp_decay_lr
        self.__init_dir()

    def __init_dir(self, overwrite=False):
        ckpt_path = self.ckpt_path
        directories = [
            ckpt_path,
            os.path.join(ckpt_path, "model_snapshots/"),
            os.path.join(ckpt_path, "train_logs/"),
        ]
        for directory in directories:
            if os.path.exists(directory) and overwrite:
                empty_directory(directory)
            make_directory_if_not_exists(directory)

        return

    def fit(self):
        model = self.model
        train_dl = self.train_dataloader
        valid_dl = self.valid_dataloader
        valid_iter = itertools.cycle(valid_dl)
        train_iter = itertools.cycle(train_dl)  # Used only for visualization data
        ldl = len(train_dl)

        ckpt_folder = self.ckpt_path + "model_snapshots/"
        log_folder = self.ckpt_path + "train_logs/"
        last_ckpt_path = ckpt_folder + "ckpt_last.ckpt"
        if self.start_clean:
            self.__init_dir(overwrite=True)

        # Fix grad accumulation step number as precaution
        gaccum_steps = ldl if self.gaccum_steps > ldl else self.gaccum_steps
        print(f"Gradient accumulation steps: {gaccum_steps}")

        # Set up optimizer
        generator_params = []
        discriminator_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad or any(
                name.startswith(key) for key in self.skip_params
            ):
                continue

            if name.startswith("generator"):
                generator_params.append(param)
            elif name.startswith("discriminator"):
                discriminator_params.append(param)
            else:
                raise ValueError(
                    "Initializing optimizers and model param sorting failed"
                )
        optimizer_G = AdamW(
            generator_params,
            self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        optimizer_D = AdamW(
            discriminator_params,
            self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_G,
            T_max=self.max_epochs,
            eta_min=self.exp_decay_lr,
        )
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_D,
            T_max=self.max_epochs,
            eta_min=self.exp_decay_lr,
        )

        # Load the last checkpoint to resume paused training
        if os.path.exists(last_ckpt_path):
            checkpoint = torch.load(last_ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            model.to("cuda")

            start_epoch = checkpoint["start_epoch"]
            losses = checkpoint["losses"]

            if self.load_optimizer:
                optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
                optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
                scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
                scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])
            else:
                scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_G,
                    T_max=self.max_epochs - start_epoch,
                    eta_min=self.exp_decay_lr,
                )
                scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_D,
                    T_max=self.max_epochs - start_epoch,
                    eta_min=self.exp_decay_lr,
                )
            print(f"Loaded checkpoint from epoch {start_epoch}")
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

        # Run Training
        for epoch in range(start_epoch, self.max_epochs):

            # Training Step
            epoch_loss_D = 0
            epoch_loss_D_real = 0
            epoch_loss_D_fake = 0
            epoch_loss_G = 0
            epoch_gan_loss = 0
            epoch_l1_loss = 0
            model.train()
            for batch in train_dl:
                real_A = batch[self.xkey].to(dtype=torch.float32, device="cuda")
                real_B = batch[self.ykey].to(dtype=torch.float32, device="cuda")
                fake_B = model.generator(real_A)

                model.discriminator.train()
                optimizer_D.zero_grad()
                loss_D, loss_D_real, loss_D_fake = model.compute_discriminator_loss(
                    real_A, real_B, fake_B
                )
                loss_D.backward()
                optimizer_D.step()
                epoch_loss_D += loss_D.item() / ldl
                epoch_loss_D_real += loss_D_real.item() / ldl
                epoch_loss_D_fake += loss_D_fake.item() / ldl

                model.generator.train()
                optimizer_G.zero_grad()
                loss_G, gan_loss, l1_loss = model.compute_generator_loss(real_A, real_B)
                loss_G.backward()
                optimizer_G.step()
                epoch_loss_G += loss_G.item() / ldl
                epoch_gan_loss += gan_loss.item() / ldl
                epoch_l1_loss += l1_loss.item() / ldl

            scheduler_G.step()
            scheduler_D.step()
            losses["loss_D"].append(epoch_loss_D)
            losses["loss_D_real"].append(epoch_loss_D_real)
            losses["loss_D_fake"].append(epoch_loss_D_fake)
            losses["loss_G"].append(epoch_loss_G)
            losses["gan_loss"].append(epoch_gan_loss)
            losses["l1_loss"].append(epoch_l1_loss)

            # Valid step
            if not self.skip_valid_step:
                model.eval()
                batch = next(valid_iter)
                real_A = batch[self.xkey].to(dtype=torch.float32, device="cuda")
                real_B = batch[self.ykey].to(dtype=torch.float32, device="cuda")
                with torch.no_grad():
                    loss_G, gan_loss, l1_loss = model.compute_generator_loss(real_A, real_B)
                test_loss = l1_loss.item()
                losses["test_l1_loss"].append(test_loss)
            else:
                losses["test_l1_loss"].append(0.0)

            current_lr = optimizer_D.param_groups[0]["lr"]

            print(f"\nEpoch {epoch+1} Finished | lr: {current_lr} Train_loss: {epoch_l1_loss:.2e} Test_Loss: {test_loss:.2e}")

            if epoch % self.snapshot_every_n == 0 or epoch == self.max_epochs - 1:
                # Save a loss figure
                self.plot_losses(losses, log_folder + "training_loss.png")

                # Draw some samples for visualization
                self.plot_drawn_samples(next(train_iter), f"epoch_{epoch}_train.png")
                self.plot_drawn_samples(next(valid_iter), f"epoch_{epoch}_test.png")

                state = {
                    "start_epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "scheduler_G_state_dict": scheduler_G.state_dict(),
                    "scheduler_D_state_dict": scheduler_D.state_dict(),
                    "losses": losses,
                }
                torch.save(state, ckpt_folder + "ckpt_last.ckpt")
                if self.save_intermediate_ckpt:
                    torch.save(state, ckpt_folder + f"ckpt_{epoch}.ckpt")
                print("Saved Model Checkpoint!")
                time.sleep(3)

        return

    def plot_drawn_samples(self, batch, fname):
        return None

    def plot_losses(self, losses, saveto):
        return None


class Trainer_colorization(Trainer_Pix2Pix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pix2pix.data_demo.coco import reverse_transform

        self.reverse_transform = reverse_transform

    def plot_losses(self, losses, saveto):
        lw = 1
        alp = 0.2

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(losses["l1_loss"], "bo", alpha=alp)
        ax[0].plot(compute_ema(losses["l1_loss"]), "b-", linewidth=lw, label="L1 Loss")
        ax[0].set_title("Generator L1 Loss")
        ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)

        ax[1].plot(losses["loss_D"], "ko", alpha=alp)
        ax[1].plot(
            compute_ema(losses["loss_D"]),
            "k-",
            linewidth=lw,
            label="Discriminator Loss",
        )
        ax[1].plot(losses["gan_loss"], "ro", alpha=alp)
        ax[1].plot(
            compute_ema(losses["gan_loss"]),
            "r-",
            linewidth=lw,
            label="Generator-GAN Loss",
        )
        ax[1].plot([0, len(losses["loss_D"])], [0.69, 0.69], "k--")
        ax[1].legend()

        fig.tight_layout()
        plt.savefig(saveto)
        plt.close()
        return

    def plot_drawn_samples(self, batch, fname):
        real_A = batch[self.xkey].to(dtype=torch.float32, device="cuda")
        real_B = batch[self.ykey].to(dtype=torch.float32, device="cuda")
        snum = np.minimum(self.snum, real_A.shape[0])
        indices = np.arange(0, snum, 1)

        real_A = real_A[indices]
        real_B = real_B[indices]
        pred_B = self.model.forward(real_A)

        fig, ax = plt.subplots(3, snum, figsize=(3 * snum, 9))
        for i in range(snum):
            gs, rgb = self.reverse_transform(real_A[i], pred_B[i])
            _, rgb_gt = self.reverse_transform(real_A[i], real_B[i])
            ax[0, i].imshow(gs, cmap="gray")
            ax[1, i].imshow(rgb)
            ax[2, i].imshow(rgb_gt)

        for axi in ax.flatten():
            axi.axis("off")
        plt.tight_layout()
        plt.savefig(self.ckpt_path + f"train_logs/{fname}")
        plt.close()

        return
