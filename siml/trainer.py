from collections import OrderedDict
import enum
import io
import pathlib
import random
import re
import time

import ignite
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as functional
from tqdm import tqdm
import yaml

from . import data_parallel
from . import datasets
from . import networks
from . import setting
from . import util


class Trainer():

    @classmethod
    def read_settings(cls, settings_yaml):
        """Read settings.yaml to generate Trainer object.

        Parameters
        ----------
            settings_yaml: str or pathlib.Path
                setting.yaml file name.
        Returns
        --------
            trainer: siml.Trainer
                Generater Trainer object.
        """
        main_setting = setting.MainSetting.read_settings_yaml(settings_yaml)
        return cls(main_setting)

    def __init__(self, settings, *, optuna_trial=None):
        """Initialize Trainer object.

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

    def train(self):
        """Perform training.

        Parameters
        ----------
        None

        Returns
        --------
        loss: float
            Loss value after training.
        """

        print(f"Ouput directory: {self.setting.trainer.output_directory}")
        self.setting.trainer.output_directory.mkdir(parents=True)

        self.prepare_training()

        setting.write_yaml(
            self.setting,
            self.setting.trainer.output_directory / 'settings.yml')

        print(
            self._display_mergin('epoch')
            + self._display_mergin('train_loss')
            + self._display_mergin('validation_loss')
            + self._display_mergin('elapsed_time'))
        with open(self.log_file, 'w') as f:
            f.write('epoch, train_loss, validation_loss, elapsed_time\n')
        self.start_time = time.time()

        self.pbar = self.create_pbar(
            len(self.train_loader) * self.setting.trainer.log_trigger_epoch)

        self.trainer.run(
            self.train_loader, max_epochs=self.setting.trainer.n_epoch)
        self.pbar.close()

        df = pd.read_csv(
            self.log_file, header=0, index_col=None, skipinitialspace=True)
        validation_loss = np.min(df['validation_loss'])

        return validation_loss

    def evaluate(self, evaluate_test=False, load_best_model=False):
        if load_best_model:
            self.setting.trainer.pretrain_directory \
                = self.setting.trainer.output_directory
            self._load_pretrained_model_if_needed()
        train_state = self.evaluator.run(self.train_loader)
        validation_state = self.evaluator.run(self.validation_loader)

        if evaluate_test:
            test_state = self.evaluator.run(self.test_loader)
            return train_state, validation_state, test_state
        else:
            return train_state, validation_state

    def _display_mergin(self, input_string, reference_string=None):
        if not reference_string:
            reference_string = input_string
        return input_string.ljust(
            len(reference_string) + self.setting.trainer.display_mergin, ' ')

    def prepare_training(self, draw=True):
        self.set_seed()

        if len(self.setting.trainer.input_names) == 0:
            raise ValueError('No input_names fed')
        if len(self.setting.trainer.output_names) == 0:
            raise ValueError('No output_names fed')

        # Define model
        self.model = networks.Network(self.setting.model, self.setting.trainer)
        if self.setting.trainer.draw_network and draw:
            self.model.draw(
                self.setting.trainer.output_directory / 'network.pdf')

        self.element_wise = self._determine_element_wise()
        self.loss = self._create_loss_function()

        # Manage settings
        if self.optuna_trial is None \
                and self.setting.trainer.prune:
            self.setting.trainer.prune = False
            print('No optuna.trial fed. Set prune = False.')

        self._select_device()
        self._generate_trainer()

        # Manage restart and pretrain
        self._load_pretrained_model_if_needed()
        self._load_restart_model_if_needed()

        # Expand data directories
        self.setting.data.train = self.train_loader.dataset.data_directories
        self.setting.data.validation \
            = self.validation_loader.dataset.data_directories
        self.setting.data.test = self.test_loader.dataset.data_directories

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

    def output_stats(self):
        dict_stats = {
            name: self._calculate_stats(parameter)
            for name, parameter in self.model.named_parameters()}
        iteration = self.trainer.state.iteration
        epoch = self.trainer.state.epoch
        dict_stats.update({
            'iteration': self.trainer.state.iteration,
            'epoch': self.trainer.state.epoch,
            'epoch_length': self.trainer.state.epoch_length})

        file_name = self.setting.trainer.output_directory \
            / f"stats_epoch{epoch}_iteration{iteration}.yml"
        with open(file_name, 'w') as f:
            yaml.dump(dict_stats, f)
        return dict_stats

    def _calculate_stats(self, parameter):
        dict_stats = {}
        if parameter.grad is not None:
            dict_stats.update(
                self._calculate_tensor_stats(parameter.grad, 'grad_'))
        dict_stats.update(self._calculate_tensor_stats(parameter, ''))
        return dict_stats

    def _calculate_tensor_stats(self, tensor, prefix):
        numpy_tensor = tensor.detach().numpy()
        abs_numpy_tensor = np.abs(numpy_tensor)
        return {
            f"{prefix}mean": float(np.mean(numpy_tensor)),
            f"{prefix}std": float(np.std(numpy_tensor)),
            f"{prefix}max": float(np.max(numpy_tensor)),
            f"{prefix}min": float(np.min(numpy_tensor)),
            f"{prefix}absmax": float(np.max(abs_numpy_tensor)),
            f"{prefix}absmin": float(np.min(abs_numpy_tensor)),
        }

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

        checkpoint = torch.load(snapshot, map_location=self.device)

        if len(self.model.state_dict()) != len(checkpoint['model_state_dict']):
            raise ValueError('Model parameter length invalid')
        # Convert new state_dict in case DataParallel wraps model
        model_state_dict = OrderedDict({
            key: value for key, value in zip(
                self.model.state_dict().keys(),
                checkpoint['model_state_dict'].values())})
        self.model.load_state_dict(model_state_dict)
        print(f"{snapshot} loaded as a pretrain model.")
        return

    def _load_restart_model_if_needed(self):
        if self.setting.trainer.restart_directory is None:
            return
        snapshot = self._select_snapshot(
            self.setting.trainer.restart_directory, method='latest')
        checkpoint = torch.load(snapshot)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trainer.load_state_dict({
            'epoch': checkpoint['epoch'],
            'validation_loss': checkpoint['validation_loss'],
            'seed': self.setting.trainer.seed,
            'max_epochs': self.setting.trainer.n_epoch,
            'epoch_length': len(self.train_loader),
        })
        self.trainer.state.epoch = checkpoint['epoch']
        # self.loss = checkpoint['loss']
        print(f"{snapshot} loaded for restart.")
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

    def _generate_trainer(self):

        if self.element_wise:
            if self.setting.trainer.element_batch_size > 0:
                batch_size = self.setting.trainer.element_batch_size
                validation_batch_size = \
                    self.setting.trainer.validation_element_batch_size
            else:
                if self.setting.trainer.simplified_model:
                    batch_size = self.setting.trainer.batch_size
                    validation_batch_size \
                        = self.setting.trainer.validation_batch_size
                else:
                    raise ValueError(
                        'element_batch_size is '
                        f"{self.setting.trainer.element_batch_size} < 1 "
                        'while element_wise is set to be true.')
        else:
            if self.setting.trainer.element_batch_size > 1 \
                    and self.setting.trainer.batch_size > 1:
                raise ValueError(
                    'batch_size cannot be > 1 when element_batch_size > 1.')
            batch_size = self.setting.trainer.batch_size
            validation_batch_size = self.setting.trainer.validation_batch_size

        if len(self.setting.trainer.split_ratio) > 0:
            develop_dataset = datasets.LazyDataset(
                self.setting.trainer.input_names,
                self.setting.trainer.output_names,
                self.setting.data.develop,
                decrypt_key=self.setting.data.encrypt_key)

            train, validation, test = util.split_data(
                develop_dataset.data_directories,
                **self.setting.trainer.split_ratio)
            self.setting.data.train = train
            self.setting.data.validation = validation
            self.setting.data.test = test

        input_is_dict = isinstance(self.setting.trainer.inputs, dict)
        output_is_dict = isinstance(self.setting.trainer.outputs, dict)
        self.collate_fn = datasets.CollateFunctionGenerator(
            time_series=self.setting.trainer.time_series,
            dict_input=input_is_dict, dict_output=output_is_dict,
            use_support=self.setting.trainer.support_inputs,
            element_wise=self.element_wise,
            data_parallel=self.setting.trainer.data_parallel)
        self.prepare_batch = self.collate_fn.prepare_batch

        if self.setting.trainer.lazy:
            self.train_loader, self.validation_loader, self.test_loader = \
                self._get_data_loaders(
                    datasets.LazyDataset, batch_size, validation_batch_size,
                    decrypt_key=self.setting.data.encrypt_key)
        else:
            if self.element_wise:
                self.train_loader, self.validation_loader, self.test_loader = \
                    self._get_data_loaders(
                        datasets.ElementWiseDataset, batch_size,
                        validation_batch_size,
                        decrypt_key=self.setting.data.encrypt_key)
            else:
                self.train_loader, self.validation_loader, self.test_loader = \
                    self._get_data_loaders(
                        datasets.OnMemoryDataset, batch_size,
                        validation_batch_size,
                        decrypt_key=self.setting.data.encrypt_key)
        self._check_data_dimension(self.setting.trainer.input_names)
        self._check_data_dimension(self.setting.trainer.output_names)

        self.optimizer = self._create_optimizer()

        self.trainer = self._create_supervised_trainer()
        self.evaluator = self._create_supervised_evaluator()

        self.desc = "loss: {:.5e}"
        # trainer_tick = max(len(self.train_loader) // 100, 1)
        trainer_tick = 1

        @self.trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            self.pbar.desc = self.desc.format(engine.state.output)
            self.pbar.update(trainer_tick)
            return

        self.evaluator_desc = "evaluating"
        # evaluator_tick = max(
        #     (len(self.train_loader) + len(self.validation_loader)) // 100, 1)
        evaluator_tick = 1

        @self.evaluator.on(
            ignite.engine.Events.ITERATION_COMPLETED(every=evaluator_tick))
        def log_evaluation(engine):
            self.evaluation_pbar.desc = self.evaluator_desc
            self.evaluation_pbar.update(evaluator_tick)
            return

        self.log_file = self.setting.trainer.output_directory / 'log.csv'
        self.plot_file = self.setting.trainer.output_directory / 'plot.pdf'

        @self.trainer.on(
            ignite.engine.Events.EPOCH_COMPLETED(
                every=self.setting.trainer.log_trigger_epoch))
        def log_training_results(engine):
            self.pbar.close()

            self.evaluation_pbar = tqdm(
                initial=0, leave=False,
                total=len(self.train_loader) + len(self.validation_loader),
                desc=self.evaluator_desc, ncols=80, ascii=True)
            self.evaluator.run(self.train_loader)
            train_loss = self.evaluator.state.metrics['loss']

            self.evaluator.run(self.validation_loader)
            validation_loss = self.evaluator.state.metrics['loss']
            self.evaluation_pbar.close()

            elapsed_time = time.time() - self.start_time

            # Print log
            tqdm.write(
                self._display_mergin(f"{engine.state.epoch}", 'epoch')
                + self._display_mergin(f"{train_loss:.5e}", 'train_loss')
                + self._display_mergin(
                    f"{validation_loss:.5e}", 'validation_loss')
                + self._display_mergin(f"{elapsed_time:.2f}", 'elapsed_time'))

            self.pbar = self.create_pbar(
                len(self.train_loader)
                * self.setting.trainer.log_trigger_epoch)
            self.pbar.n = self.pbar.last_print_n = 0

            # Save checkpoint
            torch.save(
                {
                    'epoch': engine.state.epoch,
                    'validation_loss': validation_loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },
                self.setting.trainer.output_directory
                / f"snapshot_epoch_{engine.state.epoch}.pth")

            # Write log
            with open(self.log_file, 'a') as f:
                f.write(
                    f"{engine.state.epoch}, {train_loss:.5e}, "
                    f"{validation_loss:.5e}, {elapsed_time:.2f}\n")

            # Plot
            fig = plt.figure()
            df = pd.read_csv(
                self.log_file, header=0, index_col=None, skipinitialspace=True)
            plt.plot(df['epoch'], df['train_loss'], label='train loss')
            plt.plot(
                df['epoch'], df['validation_loss'], label='validation loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.yscale('log')
            plt.legend()
            plt.savefig(self.plot_file)
            plt.close(fig)

            return

        # Add early stopping
        class StopTriggerEvents(enum.Enum):
            EVALUATED = 'evaluated'

        @self.trainer.on(
            ignite.engine.Events.EPOCH_COMPLETED(
                every=self.setting.trainer.stop_trigger_epoch))
        def fire_stop_trigger(engine):
            self.evaluator.fire_event(StopTriggerEvents.EVALUATED)
            return

        def score_function(engine):
            return -engine.state.metrics['loss']

        self.evaluator.register_events(*StopTriggerEvents)
        self.early_stopping_handler = ignite.handlers.EarlyStopping(
            patience=self.setting.trainer.patience,
            score_function=score_function,
            trainer=self.trainer)
        self.evaluator.add_event_handler(
            StopTriggerEvents.EVALUATED, self.early_stopping_handler)

        # Add pruning setting
        if self.optuna_trial is not None:
            pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(
                self.optuna_trial, 'loss', self.trainer)
            self.evaluator.add_event_handler(
                StopTriggerEvents.EVALUATED, pruning_handler)

        return

    def create_pbar(self, total):
        return tqdm(
            initial=0, leave=False,
            total=len(self.train_loader)
            * self.setting.trainer.log_trigger_epoch,
            desc=self.desc.format(0), ncols=80, ascii=True)

    def _create_supervised_trainer(self):

        def update_with_element_batch(x, y, model, optimizer):
            y_pred = model(x)

            optimizer.zero_grad()
            split_y_pred = torch.split(
                y_pred, self.setting.trainer.element_batch_size)
            split_y = torch.split(
                y, self.setting.trainer.element_batch_size)
            for syp, sy in zip(split_y_pred, split_y):
                optimizer.zero_grad()
                loss = self.loss(y_pred, y)
                loss.backward(retain_graph=True)
            self.optimizer.step()

            loss = self.loss(y_pred, y)
            return loss

        def update_standard(x, y, model, optimizer):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = self.loss(y_pred, y, x['original_shapes'])
            loss.backward()
            self.optimizer.step()
            return loss

        if not self.element_wise \
                and self.setting.trainer.element_batch_size > 0:
            update_function = update_with_element_batch
        else:
            update_function = update_standard

        def update_model(engine, batch):
            self.model.train()
            x, y = self.prepare_batch(
                batch, device=self.device, output_device=self.output_device,
                non_blocking=self.setting.trainer.non_blocking)
            loss = update_function(x, y, self.model, self.optimizer)
            if self.setting.trainer.output_stats:
                self.output_stats()
            return loss.item()

        return ignite.engine.Engine(update_model)

    def _create_supervised_evaluator(self):

        def _inference(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = self.prepare_batch(
                    batch, device=self.device,
                    output_device=self.output_device,
                    non_blocking=self.setting.trainer.non_blocking)
                y_pred = self.model(x)
                return y_pred, y, {'original_shapes': x['original_shapes']}

        evaluator_engine = ignite.engine.Engine(_inference)

        metrics = {'loss': ignite.metrics.Loss(self.loss)}

        for name, metric in metrics.items():
            metric.attach(evaluator_engine, name)
        return evaluator_engine

    def _get_data_loaders(
            self, dataset_generator, batch_size, validation_batch_size=None,
            decrypt_key=None):
        if validation_batch_size is None:
            validation_batch_size = batch_size

        x_variable_names = self.setting.trainer.input_names
        y_variable_names = self.setting.trainer.output_names
        train_directories = self.setting.data.train
        validation_directories = self.setting.data.validation
        test_directories = self.setting.data.test
        supports = self.setting.trainer.support_inputs
        num_workers = self.setting.trainer.num_workers

        train_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            train_directories, supports=supports, num_workers=num_workers,
            decrypt_key=decrypt_key)
        validation_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            validation_directories, supports=supports, num_workers=num_workers,
            allow_no_data=True, decrypt_key=decrypt_key)
        test_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            test_directories, supports=supports, num_workers=num_workers,
            allow_no_data=True, decrypt_key=decrypt_key)

        print(f"num_workers for data_loader: {num_workers}")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, collate_fn=self.collate_fn,
            batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, collate_fn=self.collate_fn,
            batch_size=validation_batch_size, shuffle=False,
            num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, collate_fn=self.collate_fn,
            batch_size=validation_batch_size, shuffle=False,
            num_workers=num_workers)

        return train_loader, validation_loader, test_loader

    def _create_optimizer(self):
        optimizer_name = self.setting.trainer.optimizer.lower()
        if optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                **self.setting.trainer.optimizer_setting)
        else:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    def _create_loss_function(self, pad=None):
        loss_name = self.setting.trainer.loss_function.lower()
        if loss_name == 'mse':
            loss_core = functional.mse_loss
        else:
            raise ValueError(f"Unknown loss function name: {loss_name}")

        def loss_function_dict(y_pred, y, original_shapes=None):
            return torch.mean(torch.stack([
                loss_core(y_pred[key].view(y[key].shape), y[key])
                for key in y.keys()]))

        def loss_function_without_padding(y_pred, y, original_shapes=None):
            return loss_core(y_pred.view(y.shape), y)

        def loss_function_time_with_padding(y_pred, y, original_shapes):
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
            return loss_core(concatenated_y_pred, concatenated_y)

        output_is_dict = isinstance(self.setting.trainer.outputs, dict)

        if self.setting.trainer.time_series:
            if pad is False:
                return loss_function_without_padding
            else:
                return loss_function_time_with_padding
        else:
            if output_is_dict:
                return loss_function_dict
            else:
                return loss_function_without_padding

    def _check_data_dimension(self, variable_names):
        data_directory = self.train_loader.dataset.data_directories[0]

        if isinstance(variable_names, dict):
            for value in variable_names.values():
                for variable_name in value:
                    self._check_single_variable_dimension(
                        data_directory, variable_name)

        elif isinstance(variable_names, list):
            for variable_name in variable_names:
                self._check_single_variable_dimension(
                    data_directory, variable_name)

        else:
            raise ValueError(f"Unexpected variable_names: {variable_names}")

    def _check_single_variable_dimension(self, data_directory, variable_name):
        loaded_variable = util.load_variable(
            data_directory, variable_name,
            decrypt_key=self.setting.data.encrypt_key)
        shape = loaded_variable.shape
        variable_information = self.setting.trainer.variable_information[
            variable_name]
        if len(shape) == 2:
            if shape[-1] != variable_information['dim']:
                raise ValueError(
                    f"{variable_name} dimension incorrect: "
                    f"{shape} vs {variable_information['dim']}")
        else:
            pass
        return
