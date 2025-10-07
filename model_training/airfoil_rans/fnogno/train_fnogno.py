from datetime import datetime
import os
import sys
from copy import deepcopy

import torch
import numpy as np
from zencfg import cfg_from_commandline


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from airfoil_rans.fnogno._config import DefaultConfig
from airfoil_rans.fnogno._dataset import Dataset, Processor, PlotSamples
from airfoil_rans.utils import compute_force
from surrogates.models import FNOGNO
from surrogates.training import AdamW, get_scheduler, Trainer
from surrogates.losses import LpLoss


class CLLoss(object):
    def __init__(self, skin_friction_lift):
        self.only_surface = True
        self.skin_friction_lift = skin_friction_lift
        self.out_channels = 3 if skin_friction_lift else 2

    def __call__(self, out, y=None, out_p=None, CL=None, **kwds):

        if self.only_surface:
            nodes = out_p.reshape(-1, 2)
            out = out.reshape(-1, self.out_channels)

        cl_pred, _ = compute_force(nodes, out, self.skin_friction_lift)

        return torch.abs(CL - cl_pred) / torch.abs(CL)


class CDLoss(object):
    def __init__(self, skin_friction_lift):
        self.only_surface = True
        self.skin_friction_lift = skin_friction_lift
        self.out_channels = 3 if skin_friction_lift else 2

    def __call__(self, out, y=None, out_p=None, CD=None, **kwds):

        if self.only_surface:
            nodes = out_p.reshape(-1, 2)
            out = out.reshape(-1, self.out_channels)

        _, cd_pred = compute_force(nodes, out, self.skin_friction_lift)

        return torch.abs(CD - cd_pred) / torch.abs(CD)


class LDRatioLoss(object):
    def __init__(self, skin_friction_lift):
        self.reductoin = "ratio"
        self.skin_friction_lift = skin_friction_lift
        self.out_channels = 3 if skin_friction_lift else 2

    def __call__(self, out, y=None, out_p=None, CL=None, CD=None, **kwds):

        nodes = out_p.reshape(-1, 2)
        out = out.reshape(-1, self.out_channels)
        cl_pred, cd_pred = compute_force(nodes, out, self.skin_friction_lift)

        ratio_pred = cl_pred / cd_pred
        ratio_gt = CL / CD

        return torch.abs(ratio_pred - ratio_gt) / torch.abs(ratio_gt)


class ChannelLoss(object):
    def __init__(self, out_channles, index=0):
        self.lploss = LpLoss(d=2, p=2)
        self.out_channles = out_channles
        self.index = index

    def __call__(self, out, y=None, **kwds):

        out = out.reshape(-1, self.out_channles)
        y = y.reshape(-1, self.out_channles)

        return self.lploss.rel(out[:, [self.index]], y[:, [self.index]])


if __name__ == '__main__':
    TEST_MODE = True

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = '/workspace/data/airfoil_subsonic_rans/'
    SAVE_PATH = './tmp/'

    torch.manual_seed(0)
    np.random.seed(0)

    print('Train FNOGNO On Airfoil RANS Dataset', flush=True)

    ################################################################################
    # Configs
    ################################################################################
    config = cfg_from_commandline(DefaultConfig)
    if not config.data.skin_friction_lift and config.model.out_channels == 3:
        config.model.out_channels = 2
        print("config.model.out_channels has been set to 2 becuase skin friction lift is ignored", flush=True)
    if config.train.save_dir is None:
        time_str = datetime.now().strftime("%m%d_%H%M%S")
        save_dir = os.path.join(SAVE_PATH, time_str)
        config.train.save_dir = save_dir
    os.makedirs(config.train.save_dir, exist_ok=True)
    print(config.to_dict(), flush=True)

    ################################################################################
    # Data & Processor
    ################################################################################
    dataset = Dataset(data_path=DATA_PATH,
                      n_data=config.data.n_data,
                      n_train=config.data.n_train,
                      n_test=config.data.n_test,
                      only_surface=config.data.only_surface,
                      in_p_res=config.data.in_p_res,
                      domain_lims=config.data.domain_lims,
                      normalize_y=config.data.normalize_y,
                      skin_friction_lift=config.data.skin_friction_lift)

    processor = Processor(normalizers=dataset.normalizers, device=DEVICE)
    torch.save(processor.state_dict(), os.path.join(config.train.save_dir, "processor_state_dict.pt"))

    ################################################################################
    # Model
    ################################################################################
    model = FNOGNO(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        gno_coord_dim=config.model.gno_coord_dim,
        fno_n_modes=config.model.fno_n_modes,
        fno_n_layers=config.model.fno_n_layers,
        fno_factorization=config.model.fno_factorization,
        fno_rank=config.model.fno_rank,
        gno_radius=config.model.gno_radius,
        gno_weighting_function=config.model.gno_weighting_function,
        gno_weight_function_scale=config.model.gno_weight_function_scale,
        gno_use_open3d=False,
    ).to(DEVICE)

    ################################################################################
    # Train
    ################################################################################
    train_loader = torch.utils.data.DataLoader(dataset.train_db, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset.test_db, batch_size=1, shuffle=False)

    train_loss_fn = LpLoss(d=2, p=2)
    test_loss_fn = LpLoss(d=2, p=2)

    test_p_loss_fn = ChannelLoss(config.model.out_channels, index=0)
    test_fx_loss_fn = ChannelLoss(config.model.out_channels, index=1)
    test_fy_loss_fn = ChannelLoss(config.model.out_channels, index=2)
    test_cl_loss_fn = CLLoss(config.data.skin_friction_lift)
    test_cd_loss_fn = CDLoss(config.data.skin_friction_lift)
    test_ratio_loss_fn = LDRatioLoss(config.data.skin_friction_lift)

    if config.data.skin_friction_lift:
        eval_losses = {"l2": test_loss_fn,
                       "p": test_p_loss_fn,
                       "fx": test_fx_loss_fn,
                       "fy": test_fy_loss_fn,
                       "cl": test_cl_loss_fn,
                       "cd": test_cd_loss_fn,
                       "ratio": test_ratio_loss_fn}
    else:
        eval_losses = {"l2": test_loss_fn,
                       "p": test_p_loss_fn,
                       "fx": test_fx_loss_fn,
                       "cl": test_cl_loss_fn,
                       "cd": test_cd_loss_fn,
                       "ratio": test_ratio_loss_fn}

    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = get_scheduler(optimizer, config.train)

    plot = PlotSamples(config.data.skin_friction_lift)
    trainer = Trainer(model=model,
                      n_epochs=config.train.n_epochs,
                      data_processor=processor,
                      device=DEVICE,
                      wandb_log=False,
                      log_output=False,
                      verbose=True,
                      plot=plot)

    trainer.train(train_loader=train_loader,
                  test_loaders={'test': test_loader},
                  optimizer=optimizer,
                  scheduler=scheduler,
                  training_loss=train_loss_fn,
                  eval_losses=eval_losses,
                  regularizer=None,
                  resume_from_dir="/workspace/ShapeOpt/airfoil_rans/fnogno/tmp/1005_071006/",
                  save_every=config.train.save_every,
                  save_dir=config.train.save_dir)
