# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from random import random

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}

    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams["class_balanced"] = (False, False)
    hparams["optimizer"] = ("adam", "adam")

    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    if dataset not in SMALL_IMAGES:
        hparams["lr"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5.5)))
        if algorithm == "ARM":
            hparams["batch_size"] = (8, 8)
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))

    if algorithm in ["DANN", "CDANN"]:
        if dataset not in SMALL_IMAGES:
            hparams["lr_g"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
            hparams["lr_d"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        else:
            hparams["lr_g"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
            hparams["lr_d"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))

        if dataset in SMALL_IMAGES:
            hparams["weight_decay_g"] = (0.0, 0.0)
        else:
            hparams["weight_decay_g"] = (0.0, 10 ** random_state.uniform(-6, -2))

        hparams["lambda"] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams["weight_decay_d"] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams["d_steps_per_g_step"] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams["grad_penalty"] = (0.0, 10 ** random_state.uniform(-2, 1))
        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
        hparams["mlp_width"] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams["mlp_depth"] = (3, int(random_state.choice([3, 4, 5])))
        hparams["mlp_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    elif algorithm in ['Fish', 'Fish_GGA']:
        hparams["meta_lr"] = (0.5, random_state.choice([0.05, 0.1, 0.5]))
    elif algorithm in ["RSC", "RSC_GGA"]:
        hparams["rsc_f_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
        hparams["rsc_b_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
    elif algorithm in ["SagNet", "SagNet_GGA"]:
        hparams["sag_w_adv"] = (0.1, 10 ** random_state.uniform(-2, 1))
    elif algorithm in ["IRM", "IRM_GGA"]:
        hparams["irm_lambda"] = (1e2, 10 ** random_state.uniform(-1, 5))
        hparams["irm_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm in ["Mixup", "OrgMixup", "Mixup_GGA"]:
        hparams["mixup_alpha"] = (0.2, 10 ** random_state.uniform(-1, -1))
    elif algorithm == "GroupDRO":
        hparams["groupdro_eta"] = (1e-2, 10 ** random_state.uniform(-3, -1))
    elif algorithm in ("MMD", "MMD_GGA", "CORAL", "CORAL_GGA"):
        hparams["mmd_gamma"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm in ("MLDG", "SOMLDG", "MLDG_GGA"):
        hparams["mldg_beta"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm in ["MTL", "MTL_GGA"]:
        hparams["mtl_ema"] = (0.99, random_state.choice([0.5, 0.9, 0.99, 1.0]))
    elif algorithm in ["VREx", "VREx_GGA"]:
        hparams["vrex_lambda"] = (1e1, 10 ** random_state.uniform(-1, 5))
        hparams["vrex_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm in ["SAGM_DG", "SAGM_DG_CUSTOM", "SAGM_DG_CUSTOM_ALL",
                       "SAM", "SAM_CUSTOM", "SAM_CUSTOM_ALL", "SAM_CALC_LOSS",
                       "SAGM_GGA", "SAM_CUSTOM_ALL_SIM","GSAM", "GSAM_GGA", "SAM_GGA"]:
        hparams["rho"] = (0.05, random_state.choice([0.01, 0.02, 0.05, 0.1]))
        hparams["alpha"] = (0.001, random_state.choice([0.01, 0.02, 0.05, 0.1]))

    elif algorithm == "CutMix":
        hparams["beta"] = (1.0, 1.0)
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        hparams["cutmix_prob"] = (1.0, 1.0)

    elif algorithm in ["ANDMask", "ANDMask_GGA"]:
        hparams['tau'] = (1.0, 1.0 ** random_state.uniform(0.5, 1))

    elif algorithm in ["GroupDRO", "GroupDRO_GGA"]:
        hparams['groupdro_eta'] = (1e-2, 10 ** random_state.uniform(-3, -1))

    elif algorithm in ['DANN', 'DANN_GGA', 'CDANN','CDANN_GGA']:
        hparams['lr_g'] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        hparams['lr_d'] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        hparams['weight_decay_g'] = (0., 10 ** random_state.uniform(-6, -2))
        hparams['lambda'] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams['weight_decay_d'] = (0., 10 ** random_state.uniform(-6, -2))
        hparams['d_steps_per_g_step'] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams['grad_penalty'] = (0., 10 ** random_state.uniform(-2, 1))
        hparams['beta1'] = (0.5, random_state.choice([0., 0.5]))
        hparams['mlp_width'] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams['mlp_depth'] = (3, int(random_state.choice([3, 4, 5])))
        hparams['mlp_dropout'] = (0., random_state.choice([0., 0.1, 0.5]))

    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
