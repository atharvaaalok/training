import matplotlib.pyplot as plt
import torch


def plot_surface(nodes_b, f_b1, f_b2, v=None, title_str=None, save_path='./sample.png'):

    if isinstance(nodes_b, torch.Tensor):
        nodes_b = nodes_b.detach().cpu().numpy()
    if isinstance(f_b1, torch.Tensor):
        f_b1 = f_b1.detach().cpu().numpy()
    if isinstance(f_b2, torch.Tensor):
        f_b2 = f_b2.detach().cpu().numpy()

    nodes_b = nodes_b.reshape(-1, 2)
    f_b1 = f_b1.reshape(-1)
    f_b2 = f_b2.reshape(-1)

    xb, yb = nodes_b[..., 0], nodes_b[..., 1]

    if v is None:
        vmin, vmax = min(f_b1.min(),f_b2.min()), max(f_b1.max(),f_b2.max())
    else:
        vmin, vmax = v

    fig, ax = plt.subplots(ncols=2, figsize=(20, 7))
    
    ax[0].scatter(xb, yb, c=f_b1, vmin=vmin, vmax=vmax)
    ax[0].axis('equal')

    sc = ax[1].scatter(xb, yb, c=f_b2, vmin=vmin, vmax=vmax)
    ax[1].axis('equal')
    

    plt.colorbar(sc, ax=ax)

    if title_str is not None:
        fig.suptitle(title_str)

    if save_path is not None:
        plt.savefig(save_path, dpi=256)

    plt.close(fig)
    
    # fig, ax = plt.subplots(figsize=(6, 3))
    # ax.plot(xb, yb)
    # ax.axis('equal')
    # ax.axis('off')
    
    # if save_path is not None:
    #     plt.savefig(save_path, dpi=256)

    # plt.close(fig)

def plot_field(nodes, f, v=None, save_path=None, title_str=None):

    if isinstance(nodes, torch.Tensor):
        nodes = nodes.detach().cpu().numpy()
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()

    nodes = nodes.reshape(-1, 2)
    f = f.reshape(-1)
    x, y = nodes[..., 0], nodes[..., 1]

    if v is None:
        vmin, vmax = f.min(), f.max()
    else:
        vmin, vmax = v

    fig, ax = plt.subplots(ncols=2, figsize=(20, 7))
    ax[0].scatter(x, y, c=f, cmap='viridis', vmin=vmin, vmax=vmax)
    sc = ax[1].scatter(x, y, c=f, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_ylim([-0.75, 0.75])
    ax[1].set_xlim([-0.75, 0.75])

    plt.colorbar(sc, ax=ax)

    if title_str is not None:
        fig.suptitle(title_str)

    if save_path is not None:
        plt.savefig(save_path, dpi=256)

    plt.close(fig)


def plot_metrices(loss, ratio, save_path=None):

    fig, ax = plt.subplots(ncols=2, figsize=(16, 7))
    ax[0].plot(loss, label="Loss")
    ax[1].plot(ratio, label="Lift / Drag")

    ax[0].legend()
    ax[1].legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=256)
    plt.close(fig)
