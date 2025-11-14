import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW

from functools import partial
from pix2pix.nn_blocks import *
from pix2pix.load_utils import instantiate_from_config


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# standard 8 moduli
class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ds_block_channels=[64, 128, 256, 512, 512, 512, 512, 512],
        ds_dropout=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        us_dropout=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        attn_on_scale=[],
        attn_on_upsample=True,
        attn_head_dim=32,
        group_norm_num=32,
        use_checkpoint=True,
        use_xformer=True,
        norm_type="batchnorm2d",
    ):
        super().__init__()
        assert len(ds_block_channels) == len(ds_dropout) == len(us_dropout)

        use_ds = partial(
            DownsampleBlock,
            use_checkpoint=use_checkpoint,
            norm_type=norm_type,
            group_norm_num=group_norm_num,
        )
        use_us = partial(
            UpsampleBlock,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
            norm_type=norm_type,
            mode="bilinear",
        )
        use_attn = partial(
            AttentionBlock,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
            use_xformer=use_xformer,
        )

        # Create Encoder part of the UNet
        self.input_blocks = nn.ModuleList()

        in_chs = []
        chs = in_channels
        sc = 1
        for block_num, ch in enumerate(ds_block_channels[:-1]):
            layers = [use_ds(chs, ch, ds_dropout[block_num])]
            sc = sc * 2
            if sc in attn_on_scale:
                layers.append(use_attn(in_channels=ch, attn_heads=ch // attn_head_dim))
            self.input_blocks.append(nn.Sequential(*layers))
            in_chs.append(ch)
            chs = ch

        # Seperate the last downsample block as "middle" since we wont skip
        mch = ds_block_channels[-1]
        sc = sc * 2
        layers = [use_ds(chs, mch, ds_dropout[-1])]
        if sc in attn_on_scale:
            layers.append(use_attn(in_channels=mch, attn_heads=mch // attn_head_dim))
        sc = sc // 2
        layers.append(use_us(mch, chs, us_dropout[-1]))
        if sc in attn_on_scale and attn_on_upsample:
            layers.append(use_attn(in_channels=chs, attn_heads=chs // attn_head_dim))
        self.middle_block = nn.Sequential(*layers)

        # Create Decoder part of the UNet
        self.output_blocks = nn.ModuleList()
        for block_num, ch in list(enumerate(ds_block_channels[:-2]))[::-1]:
            use_dropp = us_dropout[block_num]
            sc = sc // 2
            layers = [use_us(chs + in_chs.pop(), ch, use_dropp)]
            if sc in attn_on_scale and attn_on_upsample:
                print("Here: ", sc)
                layers.append(use_attn(in_channels=ch, attn_heads=ch // attn_head_dim))
            self.output_blocks.append(nn.Sequential(*layers))
            chs = ch

        # Create the output projection
        self.out = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(chs + in_chs.pop(), out_channels, 3, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        hx = []
        for module in self.input_blocks:
            x = module(x)
            hx.append(x)

        x = self.middle_block(x)

        for module in self.output_blocks:
            cat_in = torch.cat([x, hx.pop()], dim=1)
            x = module(cat_in)
        x = torch.cat([x, hx.pop()], dim=1)

        return self.out(x)


class PatchEncoder(nn.Module):
    def __init__(
        self,
        c_channels,
        targ_channels,
        ds_block_channels=[64, 128, 256, 512],
        attn_on_scale=[],
        use_checkpoint=True,
        group_norm_num=32,
        dropout=0.0,
        attn_head_dim=32,
        use_xformer=True,
        norm_type="batchnorm2d",
    ):
        super().__init__()
        chs = c_channels + targ_channels  # concantenate the gen and target

        use_ds = partial(
            DownsampleBlock,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
            norm_type=norm_type,
        )
        use_attn = partial(
            AttentionBlock,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
            use_xformer=use_xformer,
        )

        layers = []
        sc = 1
        # Downsample blocks
        for block_num, ch in enumerate(ds_block_channels[:-1]):
            layers.append(use_ds(chs, ch, dropout))
            sc = sc * 2
            if sc in attn_on_scale:
                layers.append(use_attn(ch, ch // attn_head_dim))
            chs = ch

        # Middle block
        chm = ds_block_channels[-1]
        layers.append(
            ConvBlock(
                chs,
                chm,
                kernel_size=3,
                stride=1,
                padding=1,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                group_norm_num=group_norm_num,
                norm_type=norm_type,
            )
        )
        if sc in attn_on_scale:
            layers.append(use_attn(chm, chm // attn_head_dim))

        # Out Block
        layers.append(nn.Conv2d(chm, 1, kernel_size=3, stride=1, padding=1))

        self.ops = nn.Sequential(*layers)

    def forward(self, input_img, target_img):
        return self.ops(torch.cat((input_img, target_img), axis=1))


class Pix2Pix(pl.LightningModule, nn.Module):
    def __init__(
        self,
        generator_config,
        discriminator_config,
        train_gen=True,
        train_disc=True,
        lambda_reg=100,
    ):
        super().__init__()
        self.generator = self.__instantiate_model(generator_config, train_gen)
        self.discriminator = self.__instantiate_model(discriminator_config, train_disc)
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionReg = nn.L1Loss()
        self.lambda_Reg = lambda_reg

    def __instantiate_model(self, config, trainable):
        model = instantiate_from_config(
            config, ckpt_path=config["ckpt_path"], strict=False
        )
        if not trainable:
            model = model.eval()
            model.train = disabled_train
            for param in model.parameters():
                param.requires_grad = False
        return model

    def compute_generator_loss(self, real_A, real_B):
        """
        Computes the generator loss consisting of GAN loss and L1 loss.
        real_A: Input image
        real_B: Target image
        """
        fake_B = self.generator(real_A)
        pred_fake = self.discriminator(fake_B, real_A)
        real_labels = torch.ones_like(pred_fake, device=pred_fake.device)
        gan_loss = self.criterionGAN(pred_fake, real_labels)
        l1_loss = self.criterionReg(fake_B, real_B) * self.lambda_Reg

        total_gen_loss = gan_loss + l1_loss
        return total_gen_loss, gan_loss, l1_loss

    def compute_discriminator_loss(self, real_A, real_B, fake_B):
        """
        Computes the discriminator loss for both real and fake pairs.
        real_A: Input image
        real_B: Target image
        fake_B: Generated image
        """

        # Real images
        pred_real = self.discriminator(real_B, real_A)
        real_labels = torch.ones_like(pred_real, device=pred_real.device)
        loss_D_real = self.criterionGAN(pred_real, real_labels)

        # Fake images
        pred_fake = self.discriminator(fake_B.detach(), real_A)
        fake_labels = torch.zeros_like(pred_fake, device=pred_fake.device)
        loss_D_fake = self.criterionGAN(pred_fake, fake_labels)

        # Total discriminator loss
        total_disc_loss = (loss_D_real + loss_D_fake) * 0.5
        return total_disc_loss, loss_D_real, loss_D_fake

    def gen_model(self):
        return self.generator

    @torch.no_grad()
    def forward(self, real_A):
        return self.generator(real_A)

class UNet5(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ds_block_channels=[64, 128, 256, 512, 512],
        ds_dropout=[0.0, 0.0, 0.0, 0.0, 0.0],
        us_dropout=[0.0, 0.0, 0.0, 0.0, 0.0],
        attn_on_scale=[],
        attn_on_upsample=True,
        attn_head_dim=32,
        group_norm_num=32,
        use_checkpoint=True,
        use_xformer=True,
        norm_type="batchnorm2d",
    ):
        super().__init__()
        assert len(ds_block_channels) == len(ds_dropout) == len(us_dropout)

        use_ds = partial(
            DownsampleBlock,
            use_checkpoint=use_checkpoint,
            norm_type=norm_type,
            group_norm_num=group_norm_num,
        )
        use_us = partial(
            UpsampleBlock,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
            norm_type=norm_type,
            mode="bilinear",
        )
        use_attn = partial(
            AttentionBlock,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
            use_xformer=use_xformer,
        )

        # Create Encoder part of the UNet
        self.input_blocks = nn.ModuleList()

        in_chs = []
        chs = in_channels
        sc = 1
        for block_num, ch in enumerate(ds_block_channels[:-1]):
            layers = [use_ds(chs, ch, ds_dropout[block_num])]
            sc = sc * 2
            if sc in attn_on_scale:
                layers.append(use_attn(in_channels=ch, attn_heads=ch // attn_head_dim))
            self.input_blocks.append(nn.Sequential(*layers))
            in_chs.append(ch)
            chs = ch

        # Seperate the last downsample block as "middle" since we wont skip
        mch = ds_block_channels[-1]
        sc = sc * 2
        layers = [use_ds(chs, mch, ds_dropout[-1])]
        if sc in attn_on_scale:
            layers.append(use_attn(in_channels=mch, attn_heads=mch // attn_head_dim))
        sc = sc // 2
        layers.append(use_us(mch, chs, us_dropout[-1]))
        if sc in attn_on_scale and attn_on_upsample:
            layers.append(use_attn(in_channels=chs, attn_heads=chs // attn_head_dim))
        self.middle_block = nn.Sequential(*layers)

        # Create Decoder part of the UNet
        self.output_blocks = nn.ModuleList()
        for block_num, ch in list(enumerate(ds_block_channels[:-2]))[::-1]:
            use_dropp = us_dropout[block_num]
            sc = sc // 2
            layers = [use_us(chs + in_chs.pop(), ch, use_dropp)]
            if sc in attn_on_scale and attn_on_upsample:
                print("Here: ", sc)
                layers.append(use_attn(in_channels=ch, attn_heads=ch // attn_head_dim))
            self.output_blocks.append(nn.Sequential(*layers))
            chs = ch

        # Create the output projection
        self.out = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(chs + in_chs.pop(), out_channels, 3, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        hx = []
        for module in self.input_blocks:
            x = module(x)
            hx.append(x)

        x = self.middle_block(x)

        for module in self.output_blocks:
            cat_in = torch.cat([x, hx.pop()], dim=1)
            x = module(cat_in)
        x = torch.cat([x, hx.pop()], dim=1)

        return self.out(x)

class UNet4(UNet):
    def __init__(
        self,
        in_channels,
        out_channels,
        ds_block_channels=[64, 128, 256, 512],
        ds_dropout=[0.0, 0.0, 0.0, 0.0],
        us_dropout=[0.0, 0.0, 0.0, 0.0],
        attn_on_scale=[],
        attn_on_upsample=True,
        attn_head_dim=32,
        group_norm_num=32,
        use_checkpoint=True,
        use_xformer=True,
        norm_type="batchnorm2d",
    ):
        super().__init__(in_channels, out_channels,ds_block_channels, 
                    ds_dropout, us_dropout, 
                    attn_on_scale, attn_on_upsample, attn_head_dim, group_norm_num, 
                    use_checkpoint, use_xformer, norm_type )
    
class UNet3(UNet):
    def __init__(
        self,
        in_channels,
        out_channels,
        ds_block_channels=[64, 128, 256],
        ds_dropout=[0.0, 0.0, 0.0],
        us_dropout=[0.0, 0.0, 0.0],
        attn_on_scale=[],
        attn_on_upsample=True,
        attn_head_dim=32,
        group_norm_num=32,
        use_checkpoint=True,
        use_xformer=True,
        norm_type="batchnorm2d",
    ):        
        super().__init__(in_channels, out_channels,ds_block_channels, 
                         ds_dropout, us_dropout, 
                         attn_on_scale, attn_on_upsample, attn_head_dim, group_norm_num, 
                         use_checkpoint, use_xformer, norm_type )
        
