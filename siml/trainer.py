import enum
import io
import random
from typing import Callable

import ignite
import numpy as np
import optuna
import torch
import yaml
from torch import Tensor
from tqdm import tqdm

from siml.loss_operations import LossCalculatorBuilder
from siml.services.environment import ModelEnvironmentSetting
from siml.services.model_builder import ModelBuilder
from siml.services.model_selector import ModelSelectorBuilder
from siml.services.training import (InnerTrainingSetting, LogRecordItems,
                                    SimlTrainingConsoleLogger,
                                    SimlTrainingFileLogger)
from siml.siml_variables import siml_tensor_variables
from siml.utils.timer import SimlStopWatch

from . import datasets, setting, update_functions, util


class Trainer:

    def __init__(self,
                 main_settings: setting.MainSetting,
                 *,
                 optuna_trial=None,
                 user_loss_function_dic:
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

        if not isinstance(main_settings, setting.MainSetting):
            raise ValueError(
                f"Unknown type for settings: {main_settings.__class__}")
        self.inference_mode = False
        self.user_loss_function_dic = user_loss_function_dic

        inner_setting = InnerTrainingSetting(main_settings=main_settings)
        # HACK: temporarly hack. Better to handle as a inner setting.
        self.setting = inner_setting.main_settings
        overwrite_restart_mode = self._overwrite_restart_mode()

        self.optuna_trial = optuna_trial
        self._env_setting = self._create_model_env_setting()

        self.prepare_training()
        self._console_logger = SimlTrainingConsoleLogger(
            display_margin=self.setting.trainer.display_mergin,
            loss_keys=self.model.get_loss_keys()
        )
        self._file_logger = SimlTrainingFileLogger(
            file_path=inner_setting.log_file_path,
            loss_figure_path=inner_setting.loss_figure_path,
            loss_keys=self.model.get_loss_keys(),
            continue_mode=overwrite_restart_mode
        )
        time_offset = self._file_logger.read_offset_start_time()
        self._stop_watch = SimlStopWatch(offset=time_offset)

    def _create_model_env_setting(self) -> ModelEnvironmentSetting:
        trainer_setting = self.setting.trainer
        _model_env = ModelEnvironmentSetting(
            gpu_id=trainer_setting.gpu_id,
            seed=trainer_setting.seed,
            data_parallel=trainer_setting.data_parallel,
            model_parallel=trainer_setting.model_parallel,
            time_series=trainer_setting.time_series
        )
        return _model_env

    def train(self, draw_model: bool = True):
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
        overwrite_restart_mode = self._overwrite_restart_mode()

        self.setting.trainer.output_directory.mkdir(
            parents=True, exist_ok=True
        )
        if self.setting.trainer.draw_network and draw_model:
            self.model.draw(self.setting.trainer.output_directory)

        yaml_file_name = f"restart_settings_{util.date_string()}.yml" \
            if overwrite_restart_mode else "settings.yml"
        setting.write_yaml(
            self.setting,
            self.setting.trainer.output_directory / yaml_file_name,
            key=self.setting.trainer.model_key)

        print(self._console_logger.output_header())
        self._file_logger.write_header_if_needed()
        self._stop_watch.start()

        self.pbar = self.create_pbar(
            len(self.train_loader) * self.setting.trainer.log_trigger_epoch)

        self.trainer.run(
            self.train_loader, max_epochs=self.setting.trainer.n_epoch)
        self.pbar.close()

        df = self._file_logger.read_history()
        validation_loss = np.min(df['validation_loss'])

        return validation_loss

    def evaluate(self, evaluate_test=False, load_best_model=False):
        if load_best_model:
            self.setting.trainer.pretrain_directory \
                = self.setting.trainer.output_directory
            self.model = self._setup_model()
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

    def _overwrite_restart_mode(self) -> bool:
        if self.setting.trainer.restart_directory is None:
            return False

        if self.setting.trainer.output_directory == \
                self.setting.trainer.restart_directory:
            return True

        return False

    def prepare_training(self):
        self._env_setting.set_seed()

        if len(self.setting.trainer.input_names) == 0:
            raise ValueError('No input_names fed')
        if len(self.setting.trainer.output_names) == 0:
            raise ValueError('No output_names fed')

        # Define model
        self.model = self._setup_model()

        self.element_wise = self.setting.trainer.determine_element_wise()
        self.loss = LossCalculatorBuilder.create(
            trainer_setting=self.setting.trainer,
            user_loss_function_dic=self.user_loss_function_dic
        )

        # Manage settings
        if self.optuna_trial is None \
                and self.setting.trainer.prune:
            self.setting.trainer.prune = False
            print('No optuna.trial fed. Set prune = False.')

        # Manage restart and pretrain
        self._generate_trainer()
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

            log_record = LogRecordItems(
                epoch=engine.state.epoch,
                train_loss=train_loss,
                train_other_losses=train_other_losses,
                validation_loss=validation_loss,
                validation_other_losses=validation_other_losses,
                elapsed_time=self._stop_watch.watch()
            )

            # Print log
            tqdm.write(self._console_logger.output(log_record))

            self.pbar = self.create_pbar(
                len(self.train_loader)
                * self.setting.trainer.log_trigger_epoch)
            self.pbar.n = self.pbar.last_print_n = 0

            # Save checkpoint
            self.save_model(engine, validation_loss)

            # Write log
            self._file_logger.write(log_record)

            # Plot
            self._file_logger.save_figure()
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

    def _select_step_update_function(
        self
    ) -> update_functions.IStepUpdateFunction:
        if self.setting.trainer.pseudo_batch_size >= 1:
            return update_functions.PseudoBatchStep(
                batch_size=self.setting.trainer.pseudo_batch_size,
                loss_func=self.loss,
                other_loss_func=self._calculate_other_loss,
                split_data_func=self._split_data_if_needed,
                device=self._env_setting.get_device(),
                output_device=self._env_setting.get_output_device(),
                loss_slice=self.setting.trainer.loss_slice,
                time_series_split=self.setting.trainer.time_series_split,
                clip_grad_norm=self.setting.trainer.clip_grad_norm,
                clip_grad_value=self.setting.trainer.clip_grad_value
            )
        elif not self.element_wise \
                and self.setting.trainer.element_batch_size > 0:
            return update_functions.ElementBatchUpdate(
                element_batch_size=self.setting.trainer.element_batch_size,
                loss_func=self.loss,
                other_loss_func=self._calculate_other_loss,
                split_data_func=self._split_data_if_needed,
                clip_grad_norm=self.setting.trainer.clip_grad_norm,
                clip_grad_value=self.setting.trainer.clip_grad_value
            )
        else:
            return update_functions.StandardUpdate(
                loss_func=self.loss,
                other_loss_func=self._calculate_other_loss,
                split_data_func=self._split_data_if_needed,
                device=self._env_setting.get_device(),
                output_device=self._env_setting.get_output_device(),
                loss_slice=self.setting.trainer.loss_slice,
                time_series_split=self.setting.trainer.time_series_split,
                clip_grad_norm=self.setting.trainer.clip_grad_norm,
                clip_grad_value=self.setting.trainer.clip_grad_value
            )

    def _create_supervised_trainer(self):

        update_function = self._select_step_update_function()

        def update_model(engine, batch):
            self.model.train()
            if self.setting.trainer.time_series_split is None:
                input_device = self._env_setting.get_device()
                support_device = self._env_setting.get_device()
                output_device = self._env_setting.get_output_device()
            else:
                input_device = 'cpu'
                support_device = self._env_setting.get_device()
                output_device = 'cpu'
            x, y = self.prepare_batch(
                batch, device=input_device, output_device=output_device,
                support_device=support_device,
                non_blocking=self.setting.trainer.non_blocking)
            loss = update_function(x, y, self.model, self.optimizer)
            if self.setting.trainer.output_stats:
                self.output_stats()
            return loss

        return ignite.engine.Engine(update_model)

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
                input_device = self._env_setting.get_device()
                support_device = self._env_setting.get_device()
            else:
                input_device = 'cpu'
                support_device = self._env_setting.get_device()

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
                device = self._env_setting.get_device()
                for split_x in split_xs:
                    siml_x = siml_tensor_variables(split_x['x'])
                    split_x['x'] = siml_x.send(device).get_values()
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

                ans_y = util.cat_time_series(
                    split_ys,
                    time_series_keys=self.output_time_series_keys
                )
                ans_siml_y = siml_tensor_variables(ans_y)
                output_device = self._env_setting.get_output_device()
                y = ans_siml_y.send(output_device).get_values()
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

    def _seed_worker(worker_id) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

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

        random_generator = torch.Generator()
        random_generator.manual_seed(self.setting.trainer.seed)

        print(f"num_workers for data_loader: {num_workers}")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=self._seed_worker,
            generator=random_generator
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            collate_fn=self.collate_fn,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=self._seed_worker,
            generator=random_generator
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=self.collate_fn,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=self._seed_worker,
            generator=random_generator
        )

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

    def _setup_model(self):
        model_loader = ModelBuilder(
            model_setting=self.setting.model,
            trainer_setting=self.setting.trainer,
            env_setting=self._env_setting
        )
        if self.setting.trainer.pretrain_directory is None:
            return model_loader.create_initialized()

        selector = ModelSelectorBuilder.create(
            self.setting.trainer.snapshot_choise_method
        )
        snapshot_file = selector.select_model(
            self.setting.trainer.pretrain_directory)

        return model_loader.create_loaded(
            snapshot_file.file_path,
            decrypt_key=self.setting.get_crypt_key()
        )

    def _load_restart_model_if_needed(self):
        if self.setting.trainer.restart_directory is None:
            return

        selector = ModelSelectorBuilder.create('latest')
        snapshot_file = selector.select_model(
            self.setting.trainer.restart_directory
        )
        checkpoint = snapshot_file.load(
            device=self._env_setting.get_device(),
            decrypt_key=self.setting.get_crypt_key()
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trainer.load_state_dict({
            'epoch': checkpoint['epoch'],
            'validation_loss': checkpoint['validation_loss'],
            'seed': self.setting.trainer.seed,
            'max_epochs': self.setting.trainer.n_epoch,
            'epoch_length': len(self.train_loader),
        })

        if self.setting.trainer.n_epoch == checkpoint['epoch']:
            raise FileExistsError(
                "Checkpoint at last epoch exists. "
                "Model to restart has already finished"
            )

        # self.loss = checkpoint['loss']
        print(f"{snapshot_file.file_path} loaded for restart.")
        return
