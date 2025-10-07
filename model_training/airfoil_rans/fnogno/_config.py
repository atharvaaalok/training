from typing import Optional, Union, List
from zencfg import ConfigBase


# small domain
class DatasetConfig(ConfigBase):
    n_data: int = 18186
    n_train: int = 17500
    n_test: int = 500

    # # # test
    # n_data: int = 101
    # n_train: int = 110
    # n_test: int = 5

    only_surface: bool = True
    in_p_res: int = 64
    domain_lims: List[List[int]] = [[-2, 2], [-2, 2]]
    normalize_y: Optional[str] = "Gauss"
    skin_friction_lift: bool = True


class ModelConfig(ConfigBase):
    model_arch: str = "FNOGNO"
    in_channels: int = 1
    out_channels: int = 3
    gno_coord_dim: int = 2
    gno_radius: float = 0.2
    gno_weighting_function: str = 'half_cos'
    gno_weight_function_scale: float = 1.0
    gno_use_open3d: bool = False
    fno_n_modes: tuple[int, int] = (16, 16)
    fno_n_layers: int = 4
    fno_factorization: Optional[str]= None
    fno_rank: float=1.0


class TrainingConfig(ConfigBase):

    n_epochs: int = 1000
    batch_size: int = 1

    learning_rate: float = 5e-4
    weight_decay: float = 1e-6

    # scheduler: str = "ReduceLROnPlateau"
    # gamma: float = 0.95
    # scheduler_patience: int = 5

    # scheduler: str = "OneCycleLR"
    # div_factor: float = 2.0
    # final_div_factor: float = 100.0
    # pct_start: float = 0.5

    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.8

    save_dir: Optional[str] = None
    save_every: int = 100


class DefaultConfig(ConfigBase):
    data: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    train: TrainingConfig = TrainingConfig()
