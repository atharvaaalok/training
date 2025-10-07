import os
import sys
from pathlib import Path
from typing import Union, List, Optional
import matplotlib.pyplot as plt

import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from surrogates.data.datasets.dict_dataset import DictDataset
from surrogates.data.transforms.data_processors import DataProcessor
from surrogates.data.transforms.normalizers import get_normalizer, UnitGaussianNormalizer, RangeNormalizer
from airfoil_rans.utils import get_sample, compute_force


def generate_latent_queries(query_res, pad=0, domain_lims=[[-40, 40], [-40, 40]]):
    oneDMeshes = []
    for lower, upper in domain_lims:
        oneDMesh = np.linspace(lower, upper, query_res)
        if pad > 0:
            start = np.linspace(lower - pad / query_res, lower, pad + 1)
            stop = np.linspace(upper, upper + pad / query_res, pad + 1)
            oneDMesh = np.concatenate([start, oneDMesh, stop])
        oneDMeshes.append(oneDMesh)
    grid = np.stack(np.meshgrid(*oneDMeshes, indexing='xy'))  # c, x, y, z(?)
    grid = torch.from_numpy(grid.astype(np.float32))
    latent_queries = grid.permute(*list(range(1, len(domain_lims) + 1)), 0)
    return latent_queries


class Dataset:
    def __init__(self,
                 data_path: Union[str, Path] = "./",
                 n_data: int = 4587,
                 n_train: int = 4000,
                 n_test: int = 200,
                 only_surface: bool = True,
                 in_p_res: int = 64,
                 domain_lims: List[List[int]] = [[-2, 2], [-2, 2]],
                 normalize_y: Optional[str] = "Gauss",
                 skin_friction_lift: bool = False):

        data_path = Path(data_path).expanduser().resolve()

        in_p = generate_latent_queries(in_p_res, domain_lims=domain_lims)
        global_features = np.load(data_path / 'raw_data' / 'global_features.npz')
        CL_list = global_features['cl']
        CD_list = global_features['cd']

        data_dicts = []
        global_min, global_max = None, None
        for idx in range(n_data):

            surface, features = get_sample(idx, data_path)

            CL, CD = torch.tensor(CL_list[idx], dtype=torch.float), torch.tensor(CD_list[idx], dtype=torch.float)
            surface = torch.tensor(surface, dtype=torch.float)
            features = torch.tensor(features, dtype=torch.float)

            if not skin_friction_lift:
                features = features[..., :2]

            if idx < n_train:
                tmp_min = torch.min(features, dim=0).values
                tmp_max = torch.max(features, dim=0).values
                if global_min is None:
                    global_min = tmp_min.clone()
                    global_max = tmp_max.clone()
                else:
                    global_min = torch.min(global_min, tmp_min)
                    global_max = torch.max(global_max, tmp_max)

            sdf = self.get_sdf(in_p, surface)
            if only_surface:
                data_dicts.append({'sdf': sdf,
                                   'out_p': surface,
                                   'y': features,
                                   'CL': CL,
                                   'CD': CD})
            else:
                raise NotImplementedError

        if normalize_y == "Range":
            print(global_min, global_max)
            normalizer = RangeNormalizer()
            normalizer.min_data = global_min
            normalizer.max_data = global_max
            self.normalizers = {'y': normalizer}
        elif normalize_y == "Gauss":
            self.normalizers = UnitGaussianNormalizer.from_dataset(data_dicts[:n_train], dim=[0, 1], keys='y')
        else:
            self.normalizers = None

        self.train_db = DictDataset(data_dicts[:n_train], constant={'in_p': in_p})
        self.test_db = DictDataset(data_dicts[-n_test:], constant={'in_p': in_p})

    def get_sdf(self, in_p, x):

        res = in_p.shape[0]

        input_geom = in_p.view(1, -1, 2)
        dists = torch.cdist(input_geom, x)
        min_dists, _ = dists.min(dim=-1)

        return min_dists.reshape(res, res, 1)


class Processor(DataProcessor):
    def __init__(self, normalizers, model=None, device='cuda'):
        super().__init__()

        self.device = device
        self.model = model
        self.normalizers = normalizers
        self.to(device)

    def forward(self, sample):

        sample = self.preprocess(sample)
        out = self.model(**sample)
        out, sample = self.postprocess(out, sample)

    def to(self, device):

        self.device = device
        if self.normalizers is not None:
            self.normalizers['y'] = self.normalizers['y'].to(device)

        return self

    def wrap(self):
        super().wrap()

    def preprocess(self, sample):

        f = sample['sdf']
        in_p = sample['in_p']
        out_p = sample['out_p']
        y = sample['y']
        CL, CD = sample['CL'], sample['CD']

        in_p, out_p, f, y = in_p.to(self.device), out_p.to(self.device), f.to(self.device), y.to(self.device)

        if self.training:
            if self.normalizers is not None:
                y = self.normalizers['y'].transform(y)

        data_dict = {
            'in_p': in_p.to(self.device).squeeze(0),
            'out_p': out_p.to(self.device).squeeze(0),
            'f': f.to(self.device).squeeze(0),
            'y': y.to(self.device),
            'CL': CL.to(self.device).squeeze(0),
            'CD': CD.to(self.device).squeeze(0)
        }

        return data_dict

    def postprocess(self, out, sample):

        if not self.training:
            if self.normalizers is not None:
                out = self.normalizers['y'].inverse_transform(out.unsqueeze(0))

        return out.squeeze(0), sample


def _plot3(nodes_b, f_b1, f_b2, v=None, title_str=None, save_path='./sample.png'):

    if isinstance(nodes_b, torch.Tensor):
        nodes_b = nodes_b.detach().cpu().numpy()
    if isinstance(f_b1, torch.Tensor):
        f_b1 = f_b1.detach().cpu().numpy()
    if isinstance(f_b2, torch.Tensor):
        f_b2 = f_b2.detach().cpu().numpy()

    nodes_b = nodes_b.reshape(-1, 2)
    f_b1 = f_b1.reshape(-1, 3)
    f_b2 = f_b2.reshape(-1, 3)

    xb, yb = nodes_b[..., 0], nodes_b[..., 1]

    if v is None:
        vmin = np.minimum(f_b1.min(axis=0), f_b2.min(axis=0))
        vmax = np.maximum(f_b1.max(axis=0), f_b2.max(axis=0))
    else:
        vmin, vmax = v

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(30, 15))

    for j in range(3):

        ax[j, 0].scatter(xb, yb, c=f_b1[:, j], vmin=vmin[j], vmax=vmax[j])
        ax[j, 0].axis('equal')
        ax[j, 0].axis('off')

        sc = ax[j, 1].scatter(xb, yb, c=f_b2[:, j], vmin=vmin[j], vmax=vmax[j])
        ax[j, 1].axis('equal')
        ax[j, 1].axis('off')
        plt.colorbar(sc, ax=[ax[j, 0], ax[j, 1]])

        sc = ax[j, 2].scatter(xb, yb, c=np.abs(f_b1[:, j] - f_b2[:, j]))
        ax[j, 2].axis('equal')
        ax[j, 2].axis('off')
        plt.colorbar(sc, ax=[ax[j, 2]])

    if title_str is not None:
        fig.suptitle(title_str)

    if save_path is not None:
        plt.savefig(save_path, dpi=128)

    plt.close(fig)


def _plot2(nodes_b, f_b1, f_b2, v=None, title_str=None, save_path='./sample.png'):

    if isinstance(nodes_b, torch.Tensor):
        nodes_b = nodes_b.detach().cpu().numpy()
    if isinstance(f_b1, torch.Tensor):
        f_b1 = f_b1.detach().cpu().numpy()
    if isinstance(f_b2, torch.Tensor):
        f_b2 = f_b2.detach().cpu().numpy()

    nodes_b = nodes_b.reshape(-1, 2)
    f_b1 = f_b1.reshape(-1, 2)
    f_b2 = f_b2.reshape(-1, 2)

    xb, yb = nodes_b[..., 0], nodes_b[..., 1]

    if v is None:
        vmin = np.minimum(f_b1.min(axis=0), f_b2.min(axis=0))
        vmax = np.maximum(f_b1.max(axis=0), f_b2.max(axis=0))
    else:
        vmin, vmax = v

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 10))

    for j in range(2):

        ax[j, 0].scatter(xb, yb, c=f_b1[:, j], vmin=vmin[j], vmax=vmax[j])
        ax[j, 0].axis('equal')
        ax[j, 0].axis('off')

        sc = ax[j, 1].scatter(xb, yb, c=f_b2[:, j], vmin=vmin[j], vmax=vmax[j])
        ax[j, 1].axis('equal')
        ax[j, 1].axis('off')
        plt.colorbar(sc, ax=[ax[j, 0], ax[j, 1]])

        sc = ax[j, 2].scatter(xb, yb, c=np.abs(f_b1[:, j] - f_b2[:, j]))
        ax[j, 2].axis('equal')
        ax[j, 2].axis('off')
        plt.colorbar(sc, ax=[ax[j, 2]])

    if title_str is not None:
        fig.suptitle(title_str)

    if save_path is not None:
        plt.savefig(save_path, dpi=128)

    plt.close(fig)


class PlotSamples(object):
    def __init__(self, skin_friction_lift):
        self.epoch = 0
        self.skin_friction_lift = skin_friction_lift
        self.out_channels = 3 if skin_friction_lift else 2

    def __call__(self, out, sample, eval_step_losses, save_dir):

        save_path = os.path.join(save_dir, f'samples/sample_{self.epoch:04d}.png')
        os.makedirs(os.path.join(save_dir, 'samples/'), exist_ok=True)

        nodes = sample['out_p'].reshape(-1, 2)
        out = out.reshape(-1, self.out_channels)

        cl_pred, cd_pred = compute_force(nodes, out, self.skin_friction_lift)

        if self.skin_friction_lift:
            title_str = f"SU2: CL={sample['CL']}, CD={sample['CD']}\nPred: CL={cl_pred.item()}, CD={cd_pred.item()}\nError: p {eval_step_losses['p']}, fx {eval_step_losses['fx']}, fy {eval_step_losses['fy']}"
            _plot3(nodes, out, sample['y'], title_str=title_str, save_path=save_path)
        else:
            title_str = f"SU2: CL={sample['CL']}, CD={sample['CD']}\nPred: CL={cl_pred.item()}, CD={cd_pred.item()}\nError: p {eval_step_losses['p']}, fx {eval_step_losses['fx']}"
            _plot2(nodes, out, sample['y'], title_str=title_str, save_path=save_path)

        self.epoch += 1
