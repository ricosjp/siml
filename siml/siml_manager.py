from collections import OrderedDict
import io
import pathlib
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from . import data_parallel
from . import setting
from . import util


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

    def __init__(self, settings, *, optuna_trial=None):
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

        self._update_setting_if_needed()
        self.optuna_trial = optuna_trial
        return

    def _select_device(self):
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
            elif self.setting.trainer.gpu_id != -1:
                self.device = f"cuda:{self.setting.trainer.gpu_id}"
                self.output_device = self.device
                self.model.to(self.device)
                print(f"GPU device: {self.setting.trainer.gpu_id}")
            else:
                self.device = 'cpu'
                self.output_device = self.device
        else:
            if self.setting.trainer.gpu_id != -1 \
                    or self.setting.trainer.data_parallel \
                    or self.setting.trainer.model_parallel:
                raise ValueError('No GPU found.')
            self.setting.trainer.gpu_id = -1
            self.device = 'cpu'
            self.output_device = self.device

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
            yamls = list(path.glob('*.y*ml'))
            if len(yamls) != 1:
                raise ValueError(f"{len(yamls)} yaml files found in {path}")
            yaml_file = yamls[0]
        if only_model:
            self.setting.model = setting.MainSetting.read_settings_yaml(
                yaml_file).model
        else:
            self.setting = setting.MainSetting.read_settings_yaml(yaml_file)
        if self.setting.trainer.output_directory.exists():
            print(
                f"{self.setting.trainer.output_directory} exists "
                'so reset output directory.')
            self.setting.trainer.output_directory = \
                setting.TrainerSetting([], []).output_directory
        return

    def _update_setting_if_needed(self):
        if self.setting.trainer.restart_directory is not None:
            restart_directory = self.setting.trainer.restart_directory
            self._update_setting(self.setting.trainer.restart_directory)
            self.setting.trainer.restart_directory = restart_directory
        elif self.setting.trainer.pretrain_directory is not None:
            pretrain_directory = self.setting.trainer.pretrain_directory
            self._update_setting(
                self.setting.trainer.pretrain_directory, only_model=True)
            self.setting.trainer.pretrain_directory = pretrain_directory
        elif self.setting.trainer.restart_directory is not None \
                and self.setting.trainer.pretrain_directory is not None:
            raise ValueError(
                'Restart directory and pretrain directory cannot be specified '
                'at the same time.')
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

        key = self.setting.inferer.model_key
        if key is not None and snapshot.suffix == '.enc':
            checkpoint = torch.load(
                util.decrypt_file(key, snapshot), map_location=self.device)
        else:
            checkpoint = torch.load(snapshot, map_location=self.device)

        if len(self.model.state_dict()) != len(checkpoint['model_state_dict']):
            raise ValueError('Model parameter length invalid')
        # Convert new state_dict in case DataParallel wraps model
        model_state_dict = OrderedDict({
            k1: checkpoint['model_state_dict'][k2] for k1, k2 in zip(
                sorted(self.model.state_dict().keys()),
                sorted(checkpoint['model_state_dict'].keys()))})
        self.model.load_state_dict(model_state_dict)
        print(f"{snapshot} loaded as a pretrain model.")
        return

    def _select_snapshot(self, path, method='best'):
        if not path.exists():
            raise ValueError(f"{path} doesn't exist")

        if path.is_file():
            return path
        elif path.is_dir():
            snapshots = path.glob('snapshot_epoch_*')
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
                return path / f"snapshot_epoch_{best_epoch}.pth"
            elif method == 'train_best':
                df = pd.read_csv(
                    path / 'log.csv', header=0, index_col=None,
                    skipinitialspace=True)
                best_epoch = df['epoch'].iloc[
                    df['train_loss'].idxmin()]
                return path / f"snapshot_epoch_{best_epoch}.pth"
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
                if y is None or len(y) == 0:
                    return None
                else:
                    return loss_with_answer(
                        y_pred, y, original_shapes=original_shapes)
            return loss_function_with_allowing_no_answer

        loss_name = self.setting.trainer.loss_function.lower()
        output_is_dict = isinstance(self.setting.trainer.outputs, dict)
        return LossFunction(
            loss_name=loss_name, output_is_dict=output_is_dict,
            time_series=self.setting.trainer.time_series,
            output_skips=self.setting.trainer.output_skips,
            output_dims=self.setting.trainer.output_dims)


class LossFunction:

    def __init__(
            self, *, loss_name='mse', time_series=False,
            output_is_dict=False, output_skips=None, output_dims=None):
        if loss_name == 'mse':
            self.loss_core = functional.mse_loss
        else:
            raise ValueError(f"Unknown loss function name: {loss_name}")
        self.output_is_dict = output_is_dict
        self.output_dims = output_dims

        self.mask_function = self._generate_mask_function(output_skips)

        if time_series:
            self.loss = self.loss_function_time_with_padding
        else:
            if self.output_is_dict:
                self.loss = self.loss_function_dict
            else:
                self.loss = self.loss_function_without_padding

        return

    def __call__(self, y_pred, y, original_shapes=None):
        return self.loss(y_pred, y, original_shapes)

    def loss_function_dict(self, y_pred, y, original_shapes=None):
        return torch.mean(torch.stack([
            self.loss_core(*self.mask_function(
                y_pred[key].view(y[key].shape), y[key], key))
            for key in y.keys()]))

    def loss_function_without_padding(self, y_pred, y, original_shapes=None):
        return self.loss_core(*self.mask_function(y_pred.view(y.shape), y))

    def loss_function_time_with_padding(self, y_pred, y, original_shapes):
        split_y_pred = torch.split(
            y_pred, list(original_shapes[:, 1]), dim=1)
        concatenated_y_pred = torch.cat([
            sy[:s].reshape(-1)
            for s, sy in zip(original_shapes[:, 0], split_y_pred)])
        split_y = torch.split(
            y, list(original_shapes[:, 1]), dim=1)
        concatenated_y = torch.cat([
            sy[:s].reshape(-1)
            for s, sy in zip(original_shapes[:, 0], split_y)])
        return self.loss_core(
            *self.mask_function(concatenated_y_pred, concatenated_y))

    def _generate_mask_function(self, output_skips):
        if isinstance(output_skips, list):
            if not np.any(output_skips):
                return self._identity_mask
        elif isinstance(output_skips, dict):
            if np.all([not np.any(v) for v in output_skips.values()]):
                return self._identity_mask
        else:
            raise NotImplementedError

        print(f"output_skips: {output_skips}")
        if self.output_is_dict:
            self.mask = {
                key: self._generate_mask(skip_value, self.output_dims[key])
                for key, skip_value in output_skips.items()}
            return self._dict_mask
        else:
            self.mask = self._generate_mask(output_skips, self.output_dims)
            return self._array_mask

    def _generate_mask(self, skips, dims):
        return ~np.array(np.concatenate([
            [skip] * dim for skip, dim in zip(skips, dims)]))

    def _identity_mask(self, y_pred, y, key=None):
        return y_pred, y

    def _dict_mask(self, y_pred, y, key):
        masked_y_pred = y_pred[..., self.mask[key]]
        masked_y = y[..., self.mask[key]]
        if torch.numel(masked_y) == 0:
            return torch.zeros(1).to(y.device), torch.zeros(1).to(y.device)
        else:
            return masked_y_pred, masked_y

    def _array_mask(self, y_pred, y, key=None):
        return y_pred[..., self.mask], y[..., self.mask]
