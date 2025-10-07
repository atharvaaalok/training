from typing import List, Union
from pathlib import Path

import numpy as np
import torch
from .web_utils import download_from_zenodo_record

from .mesh_datamodule import MeshDataModule
from ..transforms.data_processors import DataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer
from ..transforms.data_processors import DataProcessor

# from geo.distance import compute_distance_2d
from ..transforms.normalizers import RangeNormalizer
from .auxed_dataset import PCNOAuxedDataset


class GINOCarCFDDataset(MeshDataModule):
    """CarCFDDataset is a processed version of the dataset introduced in
    [1]_, which encodes a triangular mesh over the surface of a 3D model car
    and provides the air pressure at each centroid and vertex of the mesh when
    the car is placed in a simulated wind tunnel with a recorded inlet velocity.
    In our case, inputs are a signed distance function evaluated over a regular
    3D grid of query points, as well as the inlet velocity. Outputs are pressure 
    values at each centroid of the triangle mesh.

        .. warning:: 

        ``CarCFDDataset`` inherits from ``MeshDataModule``, which requires the optional ``open3d`` dependency.
        See :ref:`open3d_dependency` for more information. 

    We also add additional manifest files to split the provided examples
    into training and testing sets, as well as remove instances that are corrupted.

    Data is also stored on Zenodo: https://zenodo.org/records/13936501

    Parameters
    ----------
    root_dir : Union[str, Path]
        root directory at which data is stored.
    n_train : int, optional
        Number of training instances to load, by default 1
    n_test : int, optional
        Number of testing instances to load, by default 1
    query_res : List[int], optional
        Dimension-wise resolution of signed distance function 
        (SDF) query cube, by default [32,32,32]
    download : bool, optional
        Whether to download data from Zenodo, by default True


    Attributes
    ----------
    train_loader: torch.utils.data.DataLoader
        dataloader of training examples
    test_loader: torch.utils.data.DataLoader
        dataloader of testing examples

    References
    ----------
    .. [1] : Umetani, N. and Bickel, B. (2018). "Learning three-dimensional flow for interactive 
        aerodynamic design". ACM Transactions on Graphics, 2018. 
        https://dl.acm.org/doi/10.1145/3197517.3201325.
    """

    def __init__(self,
                 root_dir: Union[str, Path],
                 n_train: int = 1,
                 n_test: int = 1,
                 query_res: List[int] = [32, 32, 32],
                 download: bool = False):
        """Initialize the CarCFDDataset.
        """
        self.zenodo_record_id = "13936501"

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        if download:
            download_from_zenodo_record(record_id=self.zenodo_record_id,
                                        root=root_dir)
        super().__init__(
            root_dir=root_dir,
            item_dir_name='',
            n_train=n_train,
            n_test=n_test,
            query_res=query_res,
            attributes=['press']
        )

        # process data list to remove specific vertices from pressure to match number of vertices
        for i, data in enumerate(self.train_data.data_list):
            press = data['press']
            self.train_data.data_list[i]['press'] = torch.cat((press[:, 0:16], press[:, 112:]), axis=1)
        for i, data in enumerate(self.test_data.data_list):
            press = data['press']
            self.test_data.data_list[i]['press'] = torch.cat((press[:, 0:16], press[:, 112:]), axis=1)


class GINOCarCFDProcessor(DataProcessor):
    """
    Implements logic to preprocess data/handle model outputs
    to train an GINO on the CFD car-pressure dataset
    """

    def __init__(self, normalizer, device='cuda'):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None

    def preprocess(self, sample):
        # Turn a data dictionary returned by MeshDataModule's DictDataset
        # into the form expected by the GINO

        # input geometry: just vertices
        in_p = sample['vertices'].squeeze(0).to(self.device)
        latent_queries = sample['query_points'].squeeze(0).to(self.device)
        out_p = sample['vertices'].squeeze(0).to(self.device)
        f = sample['distance'].to(self.device)

        # Output data
        truth = sample['press'].squeeze(0).unsqueeze(-1)

        # Take the first 3586 vertices of the output mesh to correspond to pressure
        # if there are less than 3586 vertices, take the maximum number of truth points
        output_vertices = truth.shape[1]
        if out_p.shape[0] > output_vertices:
            out_p = out_p[:output_vertices, :]

        truth = truth.to(self.device)

        batch_dict = dict(input_geom=in_p,
                          latent_queries=latent_queries,
                          output_queries=out_p,
                          latent_features=f,
                          y=truth,
                          x=None)

        sample.update(batch_dict)

        return sample

    def postprocess(self, out, sample):
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = self.normalizer.inverse_transform(sample['y'].squeeze(0))
            sample['y'] = y

        return out, sample

    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(device)
        return self

    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample


class PCFNOCarCFDDataset:
    def __init__(self,
                 data_path: Union[str, Path] = "./pcno_carcfd_data.npz",
                 n_data: int = 611,
                 n_train: int = 500,
                 n_test: int = 111,
                 normalize_x: bool = False,
                 normalize_y: bool = True,
                 should_equal_weight: bool = False):

        data_path = Path(data_path).expanduser().resolve()
        assert n_train + n_test <= n_data

        # Load data from .npz file
        data = dict(np.load(data_path))

        # Convert to torch.Tensor
        for key in data:
            np_dtype = data[key].dtype
            if np.issubdtype(np_dtype, np.floating):
                data[key] = torch.from_numpy(data[key]).to(torch.float32)
            elif np.issubdtype(np_dtype, np.integer):
                data[key] = torch.from_numpy(data[key]).to(torch.int64)
            else:
                raise ValueError(f"Unsupported dtype {np_dtype} for key {key}")

        nodes = data["nodes"]
        mask = data["mask"]
        features = data["features"]

        weights = data["equal_weights"] if should_equal_weight else data["weights"]
        rhos = data["equal_rhos"] if should_equal_weight else data["rhos"]

        # Features: x is the input, truth is the output
        x = torch.cat([nodes.clone(), rhos], dim=-1)

        y = features[..., [0]]
        x_train, x_test = x[:n_train, ...], x[-n_test:, ...]
        y_train, y_test = y[:n_train, ...], y[-n_test:, ...]

        # Normalize x, y
        self._normalizers = {key: None for key in ['x', 'y']}
        mask_train = mask[:n_train, ...]
        if normalize_x:
            x_train, x_test, self._normalizers["x"] = self.get_normalizer(x_train, x_test, mask_train)
        if normalize_y:
            y_train, y_test, self._normalizers["y"] = self.get_normalizer(y_train, y_test, mask_train)

        print(f"x_train:{x_train.shape}, y_train:{y_train.shape}, x_test:{x_test.shape}, y_test:{y_test.shape}", flush=True)

        self.in_dim = x_train.shape[-1]

        # Aux data: mask, nodes_all, weights
        aux_train = (mask[:n_train, ...], nodes[:n_train, ...], weights[:n_train, ...])
        aux_test = (mask[-n_test:, ...], nodes[-n_test:, ...], weights[-n_test:, ...])

        # Datasets
        self._train_db = PCNOAuxedDataset(x_train, y_train, aux_train)
        self._test_db = PCNOAuxedDataset(x_test, y_test, aux_test)

    def get_normalizer(self,
                       f_train, f_test,
                       mask_train=None,
                       normalizer_type="Gauss"):

        if normalizer_type == "Gauss":
            normalizer = UnitGaussianNormalizer(dim=(0, 1,))
            normalizer.fit(f_train)
        elif normalizer_type == "Range":
            normalizer = RangeNormalizer(min=-1.0, max=1.0)
            normalizer.fit(f_train, mask_train)

        f_train = normalizer.transform(f_train)
        f_test = normalizer.transform(f_test)
        return f_train, f_test, normalizer

    @property
    def normalizers(self):
        return self._normalizers

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_db(self):
        return self._test_db


class PCFNOCarCFDDataProcessor(DataProcessor):
    """
    Implements logic to preprocess data/handle model outputs
    to train an PCFNO on the Airfoil2k dataset
    """

    def __init__(self, normalizer, device='cuda'):
        super().__init__()

        self.normalizer = normalizer
        self.device = device
        self.model = None

    def preprocess(self, sample):

        x, y, aux = sample['x'], sample['y'], sample['aux']
        x, y = x.to(self.device), y.to(self.device)
        aux = tuple(a.to(self.device) for a in aux)

        batch_dict = dict(x=x,
                          y=y,
                          aux=aux)

        sample.update(batch_dict)

        return sample

    def postprocess(self, out, sample, should_inverse=False):

        if not self.training:
            if self.normalizer is not None and should_inverse:
                out = self.normalizer.inverse_transform(out)
                y = self.normalizer.inverse_transform(sample['y'])
                sample['y'] = y

        mask = sample['aux'][0]
        out = out * mask
        sample['y'] = sample['y'] * mask

        return out, sample

    def to(self, device):
        self.device = device
        if self.normalizer is not None:
            self.normalizer = self.normalizer.to(device)
        return self

    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample
