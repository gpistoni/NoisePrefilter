import numpy as np
import matplotlib.pyplot as plt

def compute_ema(data, alpha=0.1):
    ema = [data[0]]  
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
    return np.array(ema)

def plot_losses(losses, out_dir):
    lw = 1
    alp = 0.2

    # Enable interactive mode
    plt.ion()
    fig, ax = plt.subplots(1,3, figsize=(30, 10), dpi=100 )

    ax[0].plot(losses["l1_loss"], 'bo', alpha=alp)
    ax[0].plot(losses["test_l1_loss"], 'rx', alpha=alp)
    ax[0].plot(compute_ema(losses["l1_loss"]), 'b-', linewidth=lw, label="L1 Loss")
    ax[0].plot(compute_ema(losses["test_l1_loss"]), 'r-', linewidth=lw, label="test L1 Loss")
    ax[0].set_title("Generator L1 Loss")
    ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    ax[1].plot(losses["gan_loss"], 'ro', alpha=alp)
    ax[1].plot(compute_ema(losses["gan_loss"]), 'r-', linewidth=lw, label="Generator-GAN Loss")
    ax[1].set_title("Generator-GAN Loss")
    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    ax[2].plot(losses["loss_D"], 'ko', alpha=alp)
    ax[2].plot(compute_ema(losses["loss_D"]), 'k-', linewidth=lw, label="Discriminator Loss")
    #ax[2].plot([0, len(losses["loss_D"])], [0.69, 0.69], 'k--')
    ax[2].set_title("Discriminator Loss")
    ax[2].grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout()  
    plt.savefig(out_dir + "/losses.png")
    plt.close()
    return

def visualize(model, real_A, real_B, disp_num, reverse_transform, save_to=None):
    pred_B = model.forward(real_A)
    snum = np.minimum(disp_num, pred_B.shape[0])
    fig, ax = plt.subplots(3, snum, figsize=(3 * snum, 9))
    for i in range(snum):
        gs, rgb = reverse_transform(real_A[i], pred_B[i])
        _, rgb_gt = reverse_transform(real_A[i], real_B[i])
        ax[0, i].imshow(gs)
        ax[1, i].imshow(rgb)
        ax[2, i].imshow(rgb_gt)

    for axi in ax.flatten():
        axi.axis("off")
    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    return

def visualize_gray(model, real_A, real_B, disp_num, reverse_transform, save_to=None):
    pred_B = model.forward(real_A)
    snum = np.minimum(disp_num, pred_B.shape[0])
    
    fig, ax = plt.subplots(3, snum, figsize=(3 * snum, 9))
    for i in range(snum):
        S = reverse_transform(real_A[i])
        P = reverse_transform(pred_B[i])
        D = reverse_transform(real_B[i])
        ax[0, i].imshow(S, cmap='gray')
        ax[1, i].imshow(P, cmap='gray')
        ax[2, i].imshow(D, cmap='gray')

    for axi in ax.flatten():
        axi.axis("off")
    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    return

def visualize_dataset(listA, listB, reverse_transform):
    disp_num = 6
    fig, ax = plt.subplots(2, disp_num, figsize=(3*disp_num, 6))
    for i in range(disp_num):
        gs, rgb = reverse_transform(listA[i], listB[i])
        ax[0, i].imshow(gs, cmap='gray')
        ax[1, i].imshow(rgb)

    for axi in ax.flatten():
        axi.axis('off')
    ax[0,0].set_ylabel("Grayscale")
    ax[1,0].set_ylabel("RGB")


def visualize_dataset_gray(listA, listB, reverse_transform):
    disp_num = 6
    fig, ax = plt.subplots(2, disp_num, figsize=(3*disp_num, 6))

    for i in range(disp_num):
        A, B = reverse_transform(listA[i], listB[i])
        ax[0, i].imshow(A, cmap='gray')
        ax[1, i].imshow(B, cmap='gray')


def visualize_dataset_rgb(listA, listB, reverse_transform):
    disp_num = 6
    fig, ax = plt.subplots(2, disp_num, figsize=(3*disp_num, 6))
    for i in range(disp_num):
        a, b = reverse_transform(listA[i], listB[i])
        ax[0, i].imshow(a)
        ax[1, i].imshow(b)

    for axi in ax.flatten():
        axi.axis('off')
    ax[0,0].set_ylabel("Source")
    ax[1,0].set_ylabel("Dest")