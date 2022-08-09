import enum
import io
import time

import ignite
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm
import yaml

from . import datasets
from . import networks
from . import setting
from . import siml_manager
from . import util


class Trainer(siml_manager.SimlManager):

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

        print(f"Output directory: {self.setting.trainer.output_directory}")
        self.setting.trainer.output_directory.mkdir(parents=True)

        self.prepare_training()

        setting.write_yaml(
            self.setting,
            self.setting.trainer.output_directory / 'settings.yml',
            key=self.setting.trainer.model_key)

        print(
            self._display_mergin('epoch')
            + self._display_mergin('train_loss')
            + ''.join([
                'train/' + self._display_mergin(k)
                for k in self.model.get_loss_keys()])
            + self._display_mergin('validation_loss')
            + ''.join([
                'validation/' + self._display_mergin(k)
                for k in self.model.get_loss_keys()])
            + self._display_mergin('elapsed_time'))
        with open(self.log_file, 'w') as f:
            f.write(
                'epoch, train_loss, '
                + ''.join([
                    f"train/{k}, " for k in self.model.get_loss_keys()])
                + 'validation_loss, '
                + ''.join([
                    f"validation/{k}, " for k in self.model.get_loss_keys()])
                + 'elapsed_time\n')
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
        if len(self.validation_loader) > 0:
            validation_state = self.evaluator.run(self.validation_loader)
        else:
            validation_state = None

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
            self.model.draw(self.setting.trainer.output_directory)

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
        numpy_tensor = tensor.detach().cpu().numpy()
        abs_numpy_tensor = np.abs(numpy_tensor)
        return {
            f"{prefix}mean": float(np.mean(numpy_tensor)),
            f"{prefix}std": float(np.std(numpy_tensor)),
            f"{prefix}max": float(np.max(numpy_tensor)),
            f"{prefix}min": float(np.min(numpy_tensor)),
            f"{prefix}absmax": float(np.max(abs_numpy_tensor)),
            f"{prefix}absmin": float(np.min(abs_numpy_tensor)),
        }

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
                recursive=self.setting.trainer.recursive,
                decrypt_key=self.setting.data.encrypt_key)

            train, validation, test = util.split_data(
                develop_dataset.data_directories,
                **self.setting.trainer.split_ratio)
            self.setting.data.train = train
            self.setting.data.validation = validation
            self.setting.data.test = test

        input_is_dict = isinstance(
            self.setting.trainer.inputs.variables, dict)
        output_is_dict = isinstance(
            self.setting.trainer.outputs.variables, dict)
        if isinstance(self.setting.trainer.inputs.time_series, dict):
            self.input_time_series_keys = [
                k for k, v in self.setting.trainer.inputs.time_series.items()
                if np.any(v)]
        else:
            self.input_time_series_keys = []
        if isinstance(self.setting.trainer.outputs.time_series, dict):
            self.output_time_series_keys = [
                k for k, v in self.setting.trainer.outputs.time_series.items()
                if np.any(v)]
        else:
            self.output_time_series_keys = []
        self.input_time_slices = self.setting.trainer.inputs.time_slice
        self.output_time_slices = self.setting.trainer.outputs.time_slice
        self.collate_fn = datasets.CollateFunctionGenerator(
            time_series=self.setting.trainer.time_series,
            dict_input=input_is_dict, dict_output=output_is_dict,
            use_support=self.setting.trainer.support_inputs,
            element_wise=self.element_wise,
            data_parallel=self.setting.trainer.data_parallel,
            input_time_series_keys=self.input_time_series_keys,
            output_time_series_keys=self.output_time_series_keys,
            input_time_slices=self.input_time_slices,
            output_time_slices=self.output_time_slices)
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
        self.plot_file = self.setting.trainer.output_directory \
            / f"plot.{self.setting.trainer.figure_format}"

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
            train_other_losses = {
                k: v for k, v in self.evaluator.state.metrics.items()
                if k != 'loss'}

            if len(self.validation_loader) > 0:
                self.evaluator.run(self.validation_loader)
                validation_loss = self.evaluator.state.metrics['loss']
                validation_other_losses = {
                    k: v for k, v in self.evaluator.state.metrics.items()
                    if k != 'loss'}
            else:
                validation_loss = np.nan
                validation_other_losses = {}
            self.evaluation_pbar.close()

            elapsed_time = time.time() - self.start_time

            # Print log
            tqdm.write(
                self._display_mergin(f"{engine.state.epoch}", 'epoch')
                + self._display_mergin(f"{train_loss:.5e}", 'train_loss')
                + ''.join([
                    self._display_mergin(f"{v:.5e}", 'train/' + k)
                    for k, v in train_other_losses.items()])
                + self._display_mergin(
                    f"{validation_loss:.5e}", 'validation_loss')
                + ''.join([
                    self._display_mergin(f"{v:.5e}", 'validation/' + k)
                    for k, v in validation_other_losses.items()])
                + self._display_mergin(f"{elapsed_time:.2f}", 'elapsed_time'))

            self.pbar = self.create_pbar(
                len(self.train_loader)
                * self.setting.trainer.log_trigger_epoch)
            self.pbar.n = self.pbar.last_print_n = 0

            # Save checkpoint
            self.save_model(engine, validation_loss)

            # Write log
            with open(self.log_file, 'a') as f:
                f.write(
                    f"{engine.state.epoch}, {train_loss:.5e}, "
                    + ''.join([
                        f"{v:.5e}, " for v in train_other_losses.values()])
                    + f"{validation_loss:.5e}, "
                    + ''.join([
                        f"{v:.5e}, " for v
                        in validation_other_losses.values()])
                    + f"{elapsed_time:.2f}\n")

            # Plot
            fig = plt.figure(figsize=(16 / 2, 9 / 2))
            df = pd.read_csv(
                self.log_file, header=0, index_col=None, skipinitialspace=True)
            plt.plot(df['epoch'], df['train_loss'], label='train loss')
            plt.plot(
                df['epoch'], df['validation_loss'], label='validation loss')
            for k in self.model.get_loss_keys():
                plt.plot(df['epoch'], df[f"train/{k}"], label=f"train/{k}")
            for k in self.model.get_loss_keys():
                plt.plot(
                    df['epoch'], df[f"validation/{k}"],
                    label=f"validation/{k}")
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

    def save_model(self, engine, validation_loss):
        if self.setting.trainer.model_key is None:
            torch.save(
                {
                    'epoch': engine.state.epoch,
                    'validation_loss': validation_loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },
                self.setting.trainer.output_directory
                / f"snapshot_epoch_{engine.state.epoch}.pth")
        else:
            b = io.BytesIO()
            torch.save(
                {
                    'epoch': engine.state.epoch,
                    'validation_loss': validation_loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },
                b)
            util.encrypt_file(
                self.setting.trainer.model_key,
                self.setting.trainer.output_directory
                / f"snapshot_epoch_{engine.state.epoch}.pth.enc",
                b)
        return

    def create_pbar(self, total):
        return tqdm(
            initial=0, leave=False,
            total=len(self.train_loader)
            * self.setting.trainer.log_trigger_epoch,
            desc=self.desc.format(0), ncols=80, ascii=True)

    def _create_supervised_trainer(self):

        def _clip_if_needed(model):
            model.clip_if_needed()
            if self.setting.trainer.clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), self.setting.trainer.clip_grad_value)
            if self.setting.trainer.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.setting.trainer.clip_grad_norm)
            return model

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
                other_loss = self._calculate_other_loss(model, y_pred, y)
                (loss + other_loss).backward(retain_graph=True)
                loss.backward(retain_graph=True)
            model = _clip_if_needed(model)
            self.optimizer.step()
            model.reset()

            loss = self.loss(y_pred, y)
            return loss

        def update_standard(x, y, model, optimizer):
            split_xs, split_ys = self._split_data_if_needed(
                x, y, self.setting.trainer.time_series_split)

            for split_x, split_y in zip(split_xs, split_ys):
                optimizer.zero_grad()
                split_x['x'] = self._send(split_x['x'], self.device)
                split_y = self._send(split_y, self.output_device)

                split_y_pred = model(split_x)
                loss = self.loss(
                    self._slice(split_y_pred),
                    self._slice(split_y),
                    split_x['original_shapes'])
                other_loss = self._calculate_other_loss(
                    model,
                    self._slice(split_y_pred),
                    self._slice(split_y),
                    split_x['original_shapes'])
                (loss + other_loss).backward()
                model = _clip_if_needed(model)
                self.optimizer.step()
                model.reset()
            return loss

        if not self.element_wise \
                and self.setting.trainer.element_batch_size > 0:
            update_function = update_with_element_batch
        else:
            update_function = update_standard

        def update_model(engine, batch):
            self.model.train()
            if self.setting.trainer.time_series_split is None:
                input_device = self.device
                support_device = self.device
                output_device = self.output_device
            else:
                input_device = 'cpu'
                support_device = self.device
                output_device = 'cpu'
            x, y = self.prepare_batch(
                batch, device=input_device, output_device=output_device,
                support_device=support_device,
                non_blocking=self.setting.trainer.non_blocking)
            loss = update_function(x, y, self.model, self.optimizer)
            if self.setting.trainer.output_stats:
                self.output_stats()
            return loss.item()

        return ignite.engine.Engine(update_model)

    def _slice(self, x):
        if isinstance(x, torch.Tensor):
            return x[self.setting.trainer.loss_slice]
        elif isinstance(x, dict):
            return {k: self._slice(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._slice(v) for v in x]
        else:
            raise ValueError(f"Invalid format: {x.__class__}")

    def _send(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: self._send(v, device) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._send(v, device) for v in x]
        else:
            raise ValueError(f"Invalid format: {x.__class__}")

    def _split_data_if_needed(self, x, y, time_series_split):
        if time_series_split is None:
            return [x], [y]
        split_x_tensors = self._split_core(
            x['x'], self.input_time_series_keys, time_series_split)
        split_xs = [
            {
                'x': split_x_tensor,
                'original_shapes':
                self._update_original_shapes(
                    split_x_tensor, x['original_shapes']),
                'supports': x['supports']}
            for split_x_tensor in split_x_tensors]
        split_ys = self._split_core(
            y, self.output_time_series_keys, time_series_split)
        return split_xs, split_ys

    def _update_original_shapes(self, x, previous_shapes):
        if previous_shapes is None:
            return None
        if isinstance(x, torch.Tensor):
            previous_shapes[:, 0] = len(x)
            return previous_shapes
        elif isinstance(x, dict):
            return {
                k: self._update_original_shapes(v, previous_shapes[k])
                for k, v in x.items()}
        else:
            raise ValueError(f"Invalid format: {x}")

    def _split_core(self, x, time_series_keys, time_series_split):
        if isinstance(x, torch.Tensor):
            len_x = len(x)
        elif isinstance(x, dict):
            lens = np.array([
                len(v) for k, v in x.items() if k in time_series_keys])
            if not np.all(lens == lens[0]):
                raise ValueError(
                    f"Time series length mismatch: {time_series_keys}, {lens}")
            len_x = lens[0]
        else:
            raise ValueError(f"Invalid format: {x}")

        start, step, length = time_series_split
        range_ = range(start, len_x - length + 1, step)

        if isinstance(x, torch.Tensor):
            return [x[s:s+length] for s in range_]

        elif isinstance(x, dict):
            return [{
                k:
                v[s:s+length] if k in time_series_keys
                else v for k, v in x.items()} for s in range_]

        else:
            raise ValueError(f"Invalid format: {x}")

    def _calculate_other_loss(self, model, y_pred, y, original_shapes=None):
        if original_shapes is None:
            original_shapes = [y_pred.shape]
        loss = 0.
        loss_coeffs = model.get_loss_coeffs()
        for loss_key, loss_value in model.get_losses().items():
            if 'residual' in loss_key:
                loss += loss_value * loss_coeffs[loss_key]
            else:
                raise ValueError(f"Unexpected loss_key: {loss_key}")
        return loss

    def _create_supervised_evaluator(self):

        def _inference(engine, batch):
            self.model.eval()
            if self.setting.trainer.time_series_split_evaluation is None:
                input_device = self.device
                support_device = self.device
            else:
                input_device = 'cpu'
                support_device = self.device

            with torch.no_grad():
                x, y = self.prepare_batch(
                    batch, device=input_device,
                    output_device='cpu',
                    support_device=support_device,
                    non_blocking=self.setting.trainer.non_blocking)
                split_xs, split_ys = self._split_data_if_needed(
                    x, y,
                    self.setting.trainer.time_series_split_evaluation)

                y_pred = []
                for split_x in split_xs:
                    split_x['x'] = self._send(split_x['x'], self.device)
                    y_pred.append(self.model(split_x))

                if self.setting.trainer.time_series_split_evaluation is None:
                    original_shapes = x['original_shapes']
                else:
                    cat_x = util.cat_time_series(
                        [split_x['x'] for split_x in split_xs],
                        time_series_keys=self.input_time_series_keys)
                    original_shapes = self._update_original_shapes(
                        cat_x, x['original_shapes'])
                y_pred = util.cat_time_series(
                    y_pred, time_series_keys=self.output_time_series_keys)
                y = self._send(
                    util.cat_time_series(
                        split_ys,
                        time_series_keys=self.output_time_series_keys),
                    self.output_device)
                return y_pred, y, {
                    'original_shapes': original_shapes,
                    'model': self.model}

        evaluator_engine = ignite.engine.Engine(_inference)

        metrics = {'loss': ignite.metrics.Loss(self.loss)}
        for loss_key in self.model.get_loss_keys():
            metrics.update({loss_key: self._generate_metric(loss_key)})

        for name, metric in metrics.items():
            metric.attach(evaluator_engine, name)
        return evaluator_engine

    def _generate_metric(self, loss_key):
        if 'residual' in loss_key:
            def gather_loss_key(x):
                model = x[2]['model']
                return model.get_losses()[loss_key]
            metric = ignite.metrics.Average(output_transform=gather_loss_key)
        else:
            raise ValueError(f"Unexpected loss type: {loss_key}")
        return metric

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
        recursive = self.setting.trainer.recursive

        train_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            train_directories, supports=supports, num_workers=num_workers,
            recursive=recursive, decrypt_key=decrypt_key)
        validation_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            validation_directories, supports=supports, num_workers=num_workers,
            recursive=recursive, allow_no_data=True, decrypt_key=decrypt_key)
        test_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            test_directories, supports=supports, num_workers=num_workers,
            recursive=recursive, allow_no_data=True, decrypt_key=decrypt_key)

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
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                **self.setting.trainer.optimizer_setting)
        else:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")

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

    def _load_restart_model_if_needed(self):
        if self.setting.trainer.restart_directory is None:
            return
        snapshot = self._select_snapshot(
            self.setting.trainer.restart_directory, method='latest')

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
