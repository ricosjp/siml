import random
import time

import femio
import ignite
import numpy as np
import matplotlib.pyplot as plt
# import optuna
import pandas as pd
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from . import datasets
from . import networks
from . import prepost
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

    def __init__(self, main_setting, *, optuna_trial=None):
        """Initialize Trainer object.

        Parameters
        ----------
            main_setting: siml.setting.MainSetting object
                Setting descriptions.
            model: siml.networks.Network object
                Model to be trained.
            optuna_trial: optuna.Trial
                Optuna trial object. Used for pruning.
        Returns
        --------
            None
        """
        self.setting = main_setting
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

        self._prepare_training()

        setting.write_yaml(
            self.setting,
            self.setting.trainer.output_directory / 'settings.yaml')

        print(
            self._display_mergin('epoch')
            + self._display_mergin('train_loss')
            + self._display_mergin('validation_loss')
            + self._display_mergin('elapsed_time'))
        with open(self.log_file, 'w') as f:
            f.write('epoch, train_loss, validation_loss, elapsed_time\n')
        self.pbar = tqdm(
            initial=0, leave=False,
            total=len(self.train_loader)
            * self.setting.trainer.log_trigger_epoch,
            desc=self.desc.format(0))
        self.start_time = time.time()

        self.trainer.run(
            self.train_loader, max_epochs=self.setting.trainer.n_epoch)
        self.pbar.close()

        self.evaluator.run(self.validation_loader)
        validation_loss = self.evaluator.state.metrics['loss']

        return validation_loss

    def _display_mergin(self, input_string, reference_string=None):
        if not reference_string:
            reference_string = input_string
        return input_string.ljust(
            len(reference_string) + self.setting.trainer.display_mergin, ' ')

    def _prepare_training(self):
        self.set_seed()

        if len(self.setting.trainer.input_names) == 0:
            raise ValueError('No input_names fed')
        if len(self.setting.trainer.output_names) == 0:
            raise ValueError('No output_names fed')

        # Define model
        self.model = networks.Network(self.setting.model, self.setting.trainer)
        if self.setting.trainer.time_series:
            self.element_wise = False
        else:
            if self.setting.trainer.element_wise \
                    or self.setting.trainer.simplified_model:
                self.element_wise = True
            else:
                self.element_wise = False
        self.loss = self._create_loss_function()

        # Manage settings
        if self.optuna_trial is None \
                and self.setting.trainer.prune:
            self.setting.trainer.prune = False
            print('No optuna.trial fed. Set prune = False.')

        if self._is_gpu_supporting():
            if self.setting.trainer.gpu_id != -1:
                self.device = f"cuda:{self.setting.trainer.gpu_id}"
                print(f"GPU device: {self.setting.trainer.gpu_id}")
            else:
                self.device = 'cpu'
        else:
            if self.setting.trainer.gpu_id != -1:
                raise ValueError('No GPU found.')
            self.setting.trainer.gpu_id = -1
            self.device = 'cpu'

        self._generate_trainer()

        # Manage restart and pretrain
        self._load_pretrained_model_if_needed()
        self._load_restart_model_if_needed()

    def infer(
            self, *, model_directory=None, model_file=None,
            save=True, overwrite=False, output_directory=None,
            preprocessed_data_directory=None, raw_data_directory=None,
            raw_data_basename=None,
            write_simulation=False, write_npy=True, write_yaml=True,
            write_simulation_base=None, write_simulation_stem=None,
            read_simulation_type='fistr', write_simulation_type='fistr',
            converter_parameters_pkl=None, conversion_function=None,
            data_addition_function=None):
        """Perform inference.

        Parameters
        ----------
            inference_directories: list of pathlib.Path
                Directories for inference.
            model_directory: pathlib.Path, optional [None]
                Model directory path. If not fed,
                TrainerSetting.pretrain_directory will be used.
            model_file: pathlib.Path, optional [None]
                Model directory path. If not fed,
                model_directory or TrainerSetting.pretrain_directory will be
                used.
            save: bool, optional [False]
                If True, save inference results.
            output_directory: pathlib.Path, optional [None]
                Output directory name. If not fed, data/inferred will be the
                default output directory base.
            preprocessed_data_directory: pathlib.Path, optional [None]
                Preprocessed data directories. If not fed, DataSetting.test
                will be used.
            raw_data_directory: pathlib.Path, optional [None]
                Raw data directories. If not fed, DataSetting.test
                will be used.
            raw_data_basename: pathlib.Path, optional [None]
                Raw data basename (without extention).
            write_simulation: bool, optional [False]
                If True, write simulation data file(s) based on the inference.
            write_npy: bool, optional [True]
                If True, write npy files of inferences.
            write_yaml: bool, optional [True]
                If True, write yaml file used to make inference.
            write_simulation_base: pathlib.Path, optional [None]
                Base of simulation data to be used for write_simulation option.
                If not fed, try to find from the input directories.
            read_simulation_type: str, optional ['fistr']
                Simulation file type to read.
            write_simulation_type: str, optional ['fistr']
                Simulation file type to write.
            converter_parameters_pkl: pathlib.Path, optional [None]
                Pickel file of converter parameters. IF not fed,
                DataSetting.preprocessed is used.
            conversion_function: function, optional [None]
                Conversion function to preprocess raw data. It should receive
                two parameters, fem_data and raw_directory. If not fed,
                no additional conversion occurs.
            data_addition_function: function, optional [None]
                Function to add some data at simulation data writing phase.
                If not fed, no data addition occurs.
        Returns
        --------
            inference_results: list
            Inference results contains:
                    - input variables
                    - output variables
                    - loss
        """
        self._prepare_inference(
            model_file, model_directory=model_directory,
            converter_parameters_pkl=converter_parameters_pkl)

        # Load data
        if raw_data_directory is None and raw_data_basename is None:
            # Inference based on preprocessed data
            if preprocessed_data_directory is None:
                input_directories = self.setting.data.test
            else:
                input_directories = [preprocessed_data_directory]

            dict_dir_x = self._load_data(
                self.setting.trainer.input_names, input_directories,
                return_dict=True)
            dict_dir_y = self._load_data(
                self.setting.trainer.output_names, input_directories,
                return_dict=True)

        else:
            # Inference based on raw data
            if preprocessed_data_directory is not None:
                raise ValueError(
                    'Both preprocessed_data_directory and raw_data_directory '
                    'cannot be specified at the same time')
            if raw_data_basename is not None:
                if raw_data_directory is not None:
                    raise ValueError(
                        'Both raw_data_basename and raw_data_directory cannot'
                        'be fed at the same time')
                raw_data_directory = raw_data_basename.parent
                raw_data_stem = raw_data_basename.stem
            else:
                raw_data_stem = None

            if write_simulation_base is None:
                write_simulation_base = raw_data_directory
            if write_simulation_stem is None:
                write_simulation_stem = raw_data_stem
            x, y = self._preprocess_data(
                read_simulation_type,
                raw_data_directory=raw_data_directory,
                raw_data_stem=raw_data_stem,
                prepost_converter=self.prepost_converter,
                conversion_function=conversion_function)
            dict_dir_x = {preprocessed_data_directory: x}
            if y is None:
                dict_dir_y = {}
            else:
                dict_dir_y = {preprocessed_data_directory: y}

        # Perform inference
        inference_results = [
            self._infer_single_directory(
                self.prepost_converter, directory, x, dict_dir_y,
                save=save,
                overwrite=overwrite, output_directory=output_directory,
                write_simulation=write_simulation, write_npy=write_npy,
                write_yaml=write_yaml,
                write_simulation_base=write_simulation_base,
                write_simulation_stem=write_simulation_stem,
                write_simulation_type=write_simulation_type,
                read_simulation_type=read_simulation_type,
                data_addition_function=data_addition_function)
            for directory, x in dict_dir_x.items()]
        return inference_results

    def _prepare_inference(
            self, model_file,
            *, model_directory=None, converter_parameters_pkl=None):
        # Define model
        if model_file is None:
            if model_directory is None:
                if self.setting.trainer.pretrain_directory is None:
                    raise ValueError(
                        f'No pretrain directory is specified for inference.')
            else:
                self.setting.trainer.pretrain_directory = model_directory
            self._update_setting_if_needed()

        self.model = networks.Network(self.setting.model, self.setting.trainer)
        self._load_pretrained_model_if_needed(model_file=model_file)

        if self.setting.trainer.element_wise \
                or self.setting.trainer.simplified_model:
            self.element_wise = True
        else:
            self.element_wise = False
        self.loss = self._create_loss_function(pad=False)
        self.model.eval()
        if converter_parameters_pkl is None:
            converter_parameters_pkl = self.setting.data.preprocessed \
                / 'preprocessors.pkl'
        self.prepost_converter = prepost.Converter(converter_parameters_pkl)

    def _preprocess_data(
            self, simulation_type, prepost_converter, raw_data_directory,
            *, raw_data_stem=None,
            conversion_function=None):
        fem_data = femio.FEMData.read_directory(
            simulation_type, raw_data_directory, stem=raw_data_stem,
            save=False)
        dict_data = prepost.extract_variables(
            fem_data, self.setting.conversion.mandatory,
            optional_variables=self.setting.conversion.optional)
        if conversion_function is not None:
            dict_data.update(conversion_function(fem_data, raw_data_directory))

        converted_dict_data = prepost_converter.preprocess(dict_data)
        input_data = np.concatenate([
            converted_dict_data[input_info['name']]
            for input_info in self.setting.trainer.inputs], axis=1).astype(
            np.float32)
        if np.all([
                output_info['name'] in dict_data
                for output_info in self.setting.trainer.outputs]):
            output_data = np.concatenate(
                [
                    converted_dict_data[output_info['name']]
                    for output_info in self.setting.trainer.outputs
                ], axis=1).astype(np.float32)
        else:
            output_data = None

        if self.setting.trainer.element_wise \
                or self.setting.trainer.simplified_model:
            return input_data, output_data
        else:
            if output_data is None:
                extended_output_data = None
            else:
                extended_output_data = output_data[None, :, :]
            return input_data[None, :, :], extended_output_data

    def infer_simplified_model(
            self, model_file, raw_dict_x, *,
            answer_raw_dict_y=None, model_directory=None,
            converter_parameters_pkl=None):
        """
        Infer with simplified model.

        Parameters
        ----------
            model_file: pathlib.Path
                Model file name.
            raw_dict_x: dict
                Dict of raw x data.
            answer_raw_dict_y: dict, optional [None]
                Dict of answer raw y data.
            model_directory: pathlib.Path
                Model directory name.
            converter_parameters_pkl: pathlib.Path
                Converter parameters pkl data.
        """
        self._prepare_inference(
            model_file, model_directory=model_directory,
            converter_parameters_pkl=converter_parameters_pkl)

        # Preprocess data
        preprocessed_x = self.prepost_converter.preprocess(raw_dict_x)
        x = np.concatenate(
            [
                preprocessed_x[variable_name]
                for variable_name in self.setting.trainer.input_names],
            axis=1).astype(np.float32)

        if answer_raw_dict_y is not None:
            answer_preprocessed_y = self.prepost_converter.preprocess(
                answer_raw_dict_y)
            answer_y = np.concatenate(
                [
                    answer_preprocessed_y[variable_name]
                    for variable_name in self.setting.trainer.output_names],
                axis=1).astype(np.float32)
        else:
            answer_y = None

        _, inversed_dict_y, loss = self._infer_single_data(
            self.prepost_converter, x, answer_y=answer_y)
        return inversed_dict_y, loss

    def _infer_single_data(
            self, postprocessor, x, *, answer_y=None,
            overwrite=False,
            output_directory=None, write_simulation=False, write_npy=True,
            write_simulation_base=None, write_simulation_stem=None,
            write_simulation_type='fistr', read_simulation_type='fistr',
            data_addition_function=None):

        x = torch.from_numpy(x)

        # Inference
        self.model.eval()
        with torch.no_grad():
            inferred_y = self.model({'x': x})
        if len(x.shape) == 2:
            x = x[None, :, :]
            inferred_y = inferred_y[None, :, :]
        dict_var_x = self._separate_data(x, self.setting.trainer.inputs)
        dict_var_inferred_y = self._separate_data(
            inferred_y, self.setting.trainer.outputs)

        # Postprocess
        inversed_dict_x, inversed_dict_y = postprocessor.postprocess(
            dict_var_x, dict_var_inferred_y,
            output_directory=output_directory, overwrite=overwrite,
            write_simulation=write_simulation, write_npy=write_npy,
            write_simulation_base=write_simulation_base,
            write_simulation_stem=write_simulation_stem,
            write_simulation_type=write_simulation_type,
            read_simulation_type=read_simulation_type,
            data_addition_function=data_addition_function)

        # Compute loss
        if answer_y is not None:
            answer_y = torch.from_numpy(answer_y)
            if len(answer_y.shape) == 2:
                loss = self.loss(inferred_y[0], answer_y).detach().numpy()
            elif len(answer_y.shape) == 3:
                loss = self.loss(inferred_y, answer_y).detach().numpy()
            else:
                raise ValueError(
                    f"Unknown shape of answer_y: {answer_y.shape}")
        else:
            # Answer data does not exist
            loss = None

        return inversed_dict_x, inversed_dict_y, loss

    def _infer_single_directory(
            self, postprocessor, directory, x, dict_dir_y, *, save=True,
            overwrite=False,
            output_directory=None, write_simulation=False, write_npy=True,
            write_yaml=True,
            write_simulation_base=None, write_simulation_stem=None,
            write_simulation_type='fistr', read_simulation_type='fistr',
            data_addition_function=None):

        if directory in dict_dir_y:
            # Answer data exists
            answer_y = dict_dir_y[directory]
        else:
            answer_y = None

        if save:
            if output_directory is None:
                output_directory = prepost.determine_output_directory(
                    directory, self.setting.data.inferred,
                    self.setting.data.preprocessed.stem) \
                    / f"{self.setting.trainer.name}_{util.date_string()}"
            output_directory.mkdir(parents=True, exist_ok=overwrite)
        else:
            output_directory = None

        inversed_dict_x, inversed_dict_y, loss = self._infer_single_data(
            postprocessor, x, answer_y=answer_y,
            overwrite=overwrite,
            output_directory=output_directory,
            write_simulation=write_simulation, write_npy=write_npy,
            write_simulation_base=write_simulation_base,
            write_simulation_stem=write_simulation_stem,
            write_simulation_type=write_simulation_type,
            read_simulation_type=read_simulation_type,
            data_addition_function=data_addition_function)

        if loss is not None:
            print(f"data: {directory}")
            print(f"loss: {loss}")

        if save:
            if write_yaml:
                setting.write_yaml(
                    self.setting, output_directory / 'settings.yml',
                    overwrite=overwrite)
            with open(output_directory / 'loss.dat', 'w') as f:
                f.write(f"loss: {loss}")
            print(f"Inferred data saved in: {output_directory}")

        return inversed_dict_x, inversed_dict_y, loss

    def set_seed(self):
        seed = self.setting.trainer.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def _separate_data(self, data, descriptions, *, axis=2):
        data_dict = {}
        index = 0
        data = np.swapaxes(data.detach(), 0, axis)
        for description in descriptions:
            data_dict.update({
                description['name']:
                np.swapaxes(data[index:index+description['dim']], 0, axis)})
            index += description['dim']
        return data_dict

    def _update_setting(self, path, *, only_model=False):
        if path.is_file():
            yaml_file = path
        elif path.is_dir():
            yamls = list(path.glob('*.yaml'))
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

        checkpoint = torch.load(snapshot)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"{model_file} loaded as a pretrain model.")
        return

    def _load_restart_model_if_needed(self):
        if self.setting.trainer.restart_directory is None:
            return
        snapshot = self._select_snapshot(
            self.setting.trainer.restart_directory, method='latest')
        checkpoint = torch.load(snapshot)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
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
                return max(snapshots, key=lambda p: p.stat().st_ctime)
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

        self._check_data_dimension()
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

        if self.setting.trainer.support_inputs:
            if self.setting.trainer.time_series:
                self.collate_fn = datasets.collate_fn_time_with_support
            else:
                self.collate_fn = datasets.collate_fn_with_support
            self.prepare_batch = datasets.prepare_batch_with_support
        else:
            if self.element_wise:
                self.collate_fn = datasets.collate_fn_element_wise
                self.prepare_batch = datasets.prepare_batch_without_support
            else:
                if self.setting.trainer.time_series:
                    self.collate_fn = datasets.collate_fn_time_without_support
                else:
                    self.collate_fn = datasets.collate_fn_without_support
                self.prepare_batch = datasets.prepare_batch_without_support

        if self.setting.trainer.lazy:
            self.train_loader, self.validation_loader = \
                self._get_data_loaders(
                    datasets.LazyDataset, batch_size, validation_batch_size)
        else:
            if self.element_wise:
                self.train_loader, self.validation_loader = \
                    self._get_data_loaders(
                        datasets.ElementWiseDataset, batch_size,
                        validation_batch_size)
            else:
                self.train_loader, self.validation_loader = \
                    self._get_data_loaders(
                        datasets.OnMemoryDataset, batch_size,
                        validation_batch_size)

        self.optimizer = self._create_optimizer()

        self.trainer = self._create_supervised_trainer()
        self.evaluator = self._create_supervised_evaluator()

        self.desc = "ITERATION - loss: {:.5e}"

        tick = max(len(self.train_loader) // 100, 1)

        @self.trainer.on(ignite.engine.Events.ITERATION_COMPLETED(every=tick))
        def log_training_loss(engine):
            self.pbar.desc = self.desc.format(engine.state.output)
            self.pbar.update(tick)

        self.log_file = self.setting.trainer.output_directory / 'log.csv'
        self.plot_file = self.setting.trainer.output_directory / 'plot.png'

        @self.trainer.on(
            ignite.engine.Events.EPOCH_COMPLETED(
                every=self.setting.trainer.log_trigger_epoch))
        def log_training_results(engine):
            self.pbar.refresh()

            self.evaluator.run(self.train_loader)
            train_loss = self.evaluator.state.metrics['loss']

            self.evaluator.run(self.validation_loader)
            validation_loss = self.evaluator.state.metrics['loss']

            elapsed_time = time.time() - self.start_time

            # Print log
            tqdm.write(
                self._display_mergin(f"{engine.state.epoch}", 'epoch')
                + self._display_mergin(f"{train_loss:.5e}", 'train_loss')
                + self._display_mergin(
                    f"{validation_loss:.5e}", 'validation_loss')
                + self._display_mergin(f"{elapsed_time:.2f}", 'elapsed_time'))
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

        # TODO: Add early stopping

        # TODO: Add Pruning setting
        return

    def _create_supervised_trainer(self):
        if self.device:
            self.model.to(self.device)

        def update_with_element_batch(x, y, model, optimizer):
            y_pred = model(x)

            optimizer.zero_grad()
            for _y_pred, _y in zip(y_pred, y):
                split_y_pred = torch.split(
                    _y_pred, self.setting.trainer.element_batch_size)
                split_y = torch.split(
                    _y, self.setting.trainer.element_batch_size)
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
                batch, device=self.device,
                non_blocking=self.setting.trainer.non_blocking)
            loss = update_function(x, y, self.model, self.optimizer)
            return loss.item()

        return ignite.engine.Engine(update_model)

    def _create_supervised_evaluator(self):
        if self.device:
            self.model.to(self.device)

        def _inference(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = self.prepare_batch(
                    batch, device=self.device,
                    non_blocking=self.setting.trainer.non_blocking)
                y_pred = self.model(x)
                return y_pred, y, {'original_shapes': x['original_shapes']}

        evaluator_engine = ignite.engine.Engine(_inference)

        metrics = {'loss': ignite.metrics.Loss(self.loss)}

        for name, metric in metrics.items():
            metric.attach(evaluator_engine, name)
        return evaluator_engine

    def _get_data_loaders(
            self, dataset_generator, batch_size, validation_batch_size):
        x_variable_names = self.setting.trainer.input_names
        y_variable_names = self.setting.trainer.output_names
        train_directories = self.setting.data.train
        validation_directories = self.setting.data.validation
        supports = self.setting.trainer.support_inputs
        num_workers = self.setting.trainer.num_workers

        train_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            train_directories, supports=supports)
        validation_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            validation_directories, supports=supports)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, collate_fn=self.collate_fn,
            batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, collate_fn=self.collate_fn,
            batch_size=validation_batch_size, shuffle=False,
            num_workers=num_workers)

        return train_loader, validation_loader

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

        def loss_function_with_padding(y_pred, y, original_shapes):
            concatenated_y_pred = torch.cat([
                _yp[:_l[0]] for _yp, _l in zip(y_pred, original_shapes)])
            return loss_core(concatenated_y_pred, y)

        def loss_function_without_padding(y_pred, y, original_shapes=None):
            return loss_core(y_pred, y)

        def loss_function_time_with_padding(y_pred, y, original_shapes):
            concatenated_y_pred = torch.cat([
                y_pred[:s[0], i_batch, :s[1]].reshape(-1)
                for i_batch, s in enumerate(original_shapes)])
            concatenated_y = torch.cat([
                y[:s[0], i_batch, :s[1]].reshape(-1)
                for i_batch, s in enumerate(original_shapes)])
            return loss_core(concatenated_y_pred, concatenated_y)

        if self.setting.trainer.time_series:
            return loss_function_time_with_padding
        else:
            if pad is None:
                if self.element_wise or self.setting.trainer.batch_size == 1:
                    return loss_function_without_padding
                else:
                    return loss_function_with_padding
            else:
                if pad:
                    return loss_function_with_padding
                else:
                    return loss_function_without_padding

    def _check_data_dimension(self):
        variable_names = self.setting.trainer.input_names
        directories = self.setting.data.train

        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=[f"{variable_names[0]}.npy"])
        # Check data dimension correctness
        if len(data_directories) > 0:
            data_wo_concatenation = {
                variable_name:
                util.load_variable(data_directories[0], variable_name)
                for variable_name in variable_names}
            for input_setting in self.setting.trainer.inputs:
                if input_setting['name'] in data_wo_concatenation and \
                        (data_wo_concatenation[input_setting['name']].shape[-1]
                         != input_setting['dim']):
                    setting_dim = input_setting['dim']
                    actual_dim = data_wo_concatenation[
                        input_setting['name']].shape[-1]
                    raise ValueError(
                        f"{input_setting['name']} dimension incorrect: "
                        f"{setting_dim} vs {actual_dim}")
        return

    def _load_data(
            self, variable_names, directories, *,
            return_dict=False, supports=None):
        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=[f"{variable_names[0]}.npy"])

        if supports is None:
            supports = []

        data = [
            util.concatenate_variable([
                util.load_variable(data_directory, variable_name)
                for variable_name in variable_names])
            for data_directory in data_directories]
        support_data = [
            [
                util.load_variable(data_directory, support)
                for support in supports]
            for data_directory in data_directories]
        if len(data) == 0:
            raise ValueError(f"No data found for: {directories}")
        if self.setting.trainer.element_wise \
                or self.setting.trainer.simplified_model:
            if len(support_data[0]) > 0:
                raise ValueError(
                    'Cannot use support_input if '
                    'element_wise or simplified_model is True')
            if return_dict:
                return {
                    data_directory:
                    d for data_directory, d in zip(data_directories, data)}
            else:
                return np.concatenate(data), None
        if return_dict:
            if len(supports) > 0:
                return {
                    data_directory: [d[None, :], [s]]

                    for data_directory, d, s
                    in zip(data_directories, data, support_data)}
            else:
                return {
                    data_directory: d[None, :]

                    for data_directory, d in zip(data_directories, data)}
        else:
            return data, support_data
