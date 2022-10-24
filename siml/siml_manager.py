from collections import OrderedDict
import io
import pathlib
import random
import re
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from . import data_parallel
from . import setting
from . import util
from .loss_operations.loss_calculator import LossCalculator


class SimlManager():

    @classmethod
    def read_settings(cls, settings_yaml):
        """Read settings.yaml to generate SimlManager object.

        Parameters
        ----------
        settings_yaml: str or pathlib.Path
            setting.yaml file name.

        Returns
        --------
            siml.SimlManager
        """
        main_setting = setting.MainSetting.read_settings_yaml(settings_yaml)
        return cls(main_setting)

    def __init__(self,
                 settings,
                 *,
                 optuna_trial=None,
                 user_loss_fundtion_dic:
                 dict[str, Callable[[Tensor, Tensor], Tensor]] = None):
        """Initialize SimlManager object.

        Parameters
        ----------
        settings: siml.setting.MainSetting object or pathlib.Path
            Setting descriptions.
        model: siml.networks.Network object
            Model to be trained.
        optuna_trial: optuna.Trial
            Optuna trial object. Used for pruning.

        Returns
        --------
        None
        """
        if isinstance(settings, pathlib.Path) or isinstance(
                settings, io.TextIOBase):
            self.setting = setting.MainSetting.read_settings_yaml(
                settings)
        elif isinstance(settings, setting.MainSetting):
            self.setting = settings
        else:
            raise ValueError(
                f"Unknown type for settings: {settings.__class__}")
        self.inference_mode = False
        self.user_loss_function_dic = user_loss_fundtion_dic

        self._update_setting_if_needed()
        self.optuna_trial = optuna_trial
        return

    def _select_device(self, gpu_id=None):
        if gpu_id is None:
            gpu_id = self.setting.trainer.gpu_id

        if self._is_gpu_supporting():
            if self.setting.trainer.data_parallel:
                if self.setting.trainer.time_series:
                    raise ValueError(
                        'So far both data_parallel and time_series cannot be '
                        'True')
                self.device = 'cuda:0'
                self.output_device = self.device
                gpu_count = torch.cuda.device_count()
                # TODO: Use DistributedDataParallel
                # torch.distributed.init_process_group(backend='nccl')
                # self.model = torch.nn.parallel.DistributedDataParallel(
                #     self.model)
                self.model = data_parallel.DataParallel(self.model)
                self.model.to(self.device)
                print(f"Data parallel enabled with {gpu_count} GPUs.")
            elif self.setting.trainer.model_parallel:
                self.device = 'cuda:0'
                gpu_count = torch.cuda.device_count()
                self.output_device = f"cuda:{gpu_count-1}"
                print(f"Model parallel enabled with {gpu_count} GPUs.")
            elif gpu_id != -1:
                self.device = f"cuda:{gpu_id}"
                self.output_device = self.device
                self.model.to(self.device)
                print(f"GPU device: {gpu_id}")
            else:
                self.device = 'cpu'
                self.output_device = self.device
        else:
            if gpu_id != -1 \
                    or self.setting.trainer.data_parallel \
                    or self.setting.trainer.model_parallel:
                raise ValueError('No GPU found.')
            self.setting.trainer.gpu_id = -1
            self.device = 'cpu'
            self.output_device = self.device
        return

    def _determine_element_wise(self):
        if self.setting.trainer.time_series:
            return False
        else:
            if self.setting.trainer.element_wise \
                    or self.setting.trainer.simplified_model:
                return True
            else:
                return False

    def set_seed(self):
        seed = self.setting.trainer.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def _update_setting(self, path, *, only_model=False):
        if path.is_file():
            yaml_file = path
        elif path.is_dir():
            yamls = list(path.glob('*.y*ml')) + list(path.glob('*.y*ml.enc'))
            if len(yamls) != 1:
                yaml_file_candidates = [
                    y for y in yamls if 'setting' in str(y)]
                if len(yaml_file_candidates) == 1:
                    yaml_file = yaml_file_candidates[0]
                else:
                    raise ValueError(
                        f"{len(yamls)} yaml files found in {path}")
            else:
                yaml_file = yamls[0]

        key = None
        if self.setting.trainer.model_key is not None:
            key = self.setting.trainer.model_key
            trainer_has_key = True
        else:
            trainer_has_key = False
        if self.setting.inferer.model_key is not None:
            key = self.setting.inferer.model_key
            inferer_has_key = True
        else:
            inferer_has_key = False

        if 'enc' in str(yaml_file):
            yaml_file = util.decrypt_file(key, yaml_file, return_stringio=True)

        if only_model:
            self.setting.model = setting.MainSetting.read_settings_yaml(
                yaml_file).model
        else:
            self.setting = setting.MainSetting.read_settings_yaml(yaml_file)

        if trainer_has_key:
            self.setting.trainer.model_key = key
            self.setting.data.decrypt_key = key
        if inferer_has_key:
            self.setting.inferer.model_key = key
            self.setting.data.decrypt_key = key

        if not self.inference_mode:
            if self.setting.trainer.output_directory.exists():
                print(
                    f"{self.setting.trainer.output_directory} exists "
                    'so reset output directory.')
                self.setting.trainer.output_directory = \
                    setting.TrainerSetting([], []).output_directory
        return

    def _update_setting_if_needed(self):
        if self.setting.trainer.restart_directory is not None \
                and self.setting.trainer.pretrain_directory is not None:
            raise ValueError(
                'Restart directory and pretrain directory cannot be specified '
                'at the same time.')

        if self.setting.trainer.restart_directory is not None:
            restart_directory = self.setting.trainer.restart_directory
            self._update_setting(self.setting.trainer.restart_directory)
            self.setting.trainer.restart_directory = restart_directory
        return

    def _load_pretrained_model_if_needed(self, *, model_file=None):
        if self.setting.trainer.pretrain_directory is None \
                and model_file is None:
            return
        if model_file:
            snapshot = model_file
        else:
            snapshot = self._select_snapshot(
                self.setting.trainer.pretrain_directory,
                method=self.setting.trainer.snapshot_choise_method)

        key = None
        if self.setting.trainer.model_key is not None:
            key = self.setting.trainer.model_key
        if self.setting.inferer.model_key is not None:
            key = self.setting.inferer.model_key

        if snapshot.suffix == '.enc':
            if key is None:
                raise ValueError('Feed key to load encrypted model')
            checkpoint = torch.load(
                util.decrypt_file(key, snapshot), map_location=self.device)
        else:
            checkpoint = torch.load(snapshot, map_location=self.device)

        if self.setting.trainer.state_dict_strict:
            if len(self.model.state_dict()) != len(
                    checkpoint['model_state_dict']):
                raise ValueError('Model parameter length invalid')
            # Convert new state_dict in case DataParallel wraps model
            model_state_dict = OrderedDict({
                k1: checkpoint['model_state_dict'][k2] for k1, k2 in zip(
                    sorted(self.model.state_dict().keys()),
                    sorted(checkpoint['model_state_dict'].keys()))})
        else:
            model_state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(
            model_state_dict, strict=self.setting.trainer.state_dict_strict)
        print(f"{snapshot} loaded as a pretrain model.")
        return

    def _select_snapshot(self, path, method='best', infer_epoch: int = None):
        if not path.exists():
            raise ValueError(f"{path} doesn't exist")

        if path.is_file():
            return path
        elif path.is_dir():
            snapshots = list(path.glob('snapshot_epoch_*'))
            if '.enc' in str(snapshots[0]):
                suffix = 'pth.enc'
            else:
                suffix = 'pth'

            if method == 'latest':
                return max(
                    snapshots, key=lambda p: int(re.search(
                        r'snapshot_epoch_(\d+)', str(p)).groups()[0]))
            elif method == 'best':
                df = pd.read_csv(
                    path / 'log.csv', header=0, index_col=None,
                    skipinitialspace=True)
                if np.any(np.isnan(df['validation_loss'])):
                    return self._select_snapshot(path, method='train_best')
                best_epoch = df['epoch'].iloc[
                    df['validation_loss'].idxmin()]
                return path / f"snapshot_epoch_{best_epoch}.{suffix}"
            elif method == 'train_best':
                df = pd.read_csv(
                    path / 'log.csv', header=0, index_col=None,
                    skipinitialspace=True)
                best_epoch = df['epoch'].iloc[
                    df['train_loss'].idxmin()]
                return path / f"snapshot_epoch_{best_epoch}.{suffix}"
            elif method == "specified":
                return path / f"snapshot_epoch_{infer_epoch}.{suffix}"
            else:
                raise ValueError(f"Unknown snapshot choise method: {method}")

        else:
            raise ValueError(f"{path} had unknown property.")

    def _is_gpu_supporting(self):
        return torch.cuda.is_available()

    def _create_loss_function(self, pad=False, allow_no_answer=False):
        if pad:
            raise ValueError('pad = True is no longer supported')
        if allow_no_answer:
            loss_with_answer = self._create_loss_function()
            def loss_function_with_allowing_no_answer(
                    y_pred, y, original_shapes, **kwargs):
                try:
                    if y is None or len(y) == 0 or self.min_len(y) == 0:
                        return None
                    else:
                        return loss_with_answer(
                            y_pred, y, original_shapes=original_shapes)
                except Exception:
                    print('Skip loss computation.')
                    return None
            return loss_function_with_allowing_no_answer

        loss_setting = self.setting.trainer.loss_function
        output_is_dict = isinstance(
            self.setting.trainer.outputs.variables, dict)
        return LossCalculator(
            loss_setting=loss_setting,
            output_is_dict=output_is_dict,
            time_series=self.setting.trainer.time_series,
            output_skips=self.setting.trainer.output_skips,
            output_dims=self.setting.trainer.output_dims,
            user_loss_function_dic=self.user_loss_function_dic)

    def min_len(self, x):
        if isinstance(x, torch.Tensor):
            return len(x)
        elif isinstance(x, list):
            return np.min([self.min_len(x_) for x_ in x])
        elif isinstance(x, dict):
            return np.min([self.min_len(v) for v in x.values()])
