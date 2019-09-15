import random

import chainer as ch
import daz
import numpy as np
import optuna
import pandas as pd

from . import femio
from . import util
from . import networks
from . import prepost
from . import setting
from . import updaters


class Trainer():

    @classmethod
    def read_settings(cls, settings_yaml):
        """Read settings.yaml to generate Trainer object.

        Args:
            settings_yaml: str or pathlib.Path
                setting.yaml file name.
        Returns:
            trainer: siml.Trainer
                Generater Trainer object.
        """
        main_setting = setting.MainSetting.read_settings_yaml(settings_yaml)
        return cls(main_setting)

    def __init__(self, main_setting, *, optuna_trial=None):
        """Initialize Trainer object.

        Args:
            main_setting: siml.setting.MainSetting object
                Setting descriptions.
            model: siml.networks.Network object
                Model to be trained.
            optuna_trial: optuna.Trial
                Optuna trial object. Used for pruning.
        Returns:
            None
        """
        self.setting = main_setting
        self._update_setting_if_needed()
        self.optuna_trial = optuna_trial

    def train(self):
        """Perform training.

        Args:
            None
        Returns:
            loss: float
                Overall loss value.
        """
        self._prepare_training()

        print(f"Ouput directory: {self.setting.trainer.output_directory}")
        self.setting.trainer.output_directory.mkdir(parents=True)
        setting.write_yaml(
            self.setting,
            self.setting.trainer.output_directory / 'settings.yaml')

        self.trainer.run()
        loss = np.min([
            l['validation/main/loss']
            for l in self.log_report_extension.log])
        return loss

    def _prepare_training(self):
        self.set_seed()

        # Define model
        self.model = networks.Network(self.setting.model, self.setting.trainer)
        self.classifier = networks.Classifier(
            self.model, lossfun=self._create_loss_function(),
            element_batch_size=self.setting.trainer.element_batch_size,
            element_wise=self.setting.trainer.element_wise)
        self.classifier.compute_accuracy = \
            self.setting.trainer.compute_accuracy

        # Manage settings
        if self.optuna_trial is None \
                and self.setting.trainer.prune:
            self.setting.trainer.prune = False
            print('No optuna.trial fed. Set prune = False.')
        if self._is_gpu_supporting():
            self.setting.trainer.gpu_id = self.setting.trainer.gpu_id
            print(f"GPU device: {self.setting.trainer.gpu_id}")
        else:
            if self.setting.trainer.gpu_id != -1:
                print(f"GPU not found. Using CPU.")
            self.setting.trainer.gpu_id = -1
            daz.set_daz()
            daz.set_ftz()

        # Generate trainer
        self.trainer = self._generate_trainer(
            self.setting.trainer.input_names,
            self.setting.trainer.output_names,
            self.setting.data.train, self.setting.data.validation,
            supports=self.setting.trainer.support_inputs)

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

        Args:
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
        Returns:
            inference_results: list
            Inference results contains:
                    - input variables
                    - output variables
                    - loss
        """
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
        self.classifier = networks.Classifier(
            self.model, lossfun=self._create_loss_function(),
            element_batch_size=self.setting.trainer.element_batch_size)
        self.classifier.compute_accuracy = \
            self.setting.trainer.compute_accuracy
        if converter_parameters_pkl is None:
            converter_parameters_pkl = self.setting.data.preprocessed \
                / 'preprocessors.pkl'
        prepost_converter = prepost.Converter(converter_parameters_pkl)

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
                prepost_converter=prepost_converter,
                conversion_function=conversion_function)
            dict_dir_x = {preprocessed_data_directory: x}
            if y is None:
                dict_dir_y = {}
            else:
                dict_dir_y = {preprocessed_data_directory: y}

        # Perform inference
        with ch.using_config('train', False):
            inference_results = [
                self._infer_single_data(
                    prepost_converter, directory, x, dict_dir_y, save=save,
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

        if self.setting.trainer.element_wise:
            return input_data, output_data
        else:
            if output_data is None:
                extended_output_data = None
            else:
                extended_output_data = output_data[None, :, :]
            return input_data[None, :, :], extended_output_data

    def _infer_single_data(
            self, postprocessor, directory, x, dict_dir_y, *, save=True,
            overwrite=False,
            output_directory=None, write_simulation=False, write_npy=True,
            write_yaml=True,
            write_simulation_base=None, write_simulation_stem=None,
            write_simulation_type='fistr', read_simulation_type='fistr',
            data_addition_function=None):

        # Inference
        inferred_y = self.model(x).data
        if len(x.shape) == 2:
            x = x[None, :, :]
            inferred_y = inferred_y[None, :, :]
        dict_var_x = self._separate_data(x, self.setting.trainer.inputs)
        dict_var_inferred_y = self._separate_data(
            inferred_y, self.setting.trainer.outputs)

        # Postprocess
        if save:
            if output_directory is None:
                output_directory = prepost.determine_output_directory(
                    directory, self.setting.data.inferred,
                    self.setting.data.preprocessed.stem) \
                    / f"{self.setting.trainer.name}_{util.date_string()}"
            output_directory.mkdir(parents=True, exist_ok=overwrite)
            if write_yaml:
                setting.write_yaml(
                    self.setting, output_directory / 'settings.yml',
                    overwrite=overwrite)
        else:
            output_directory = None

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
        if directory in dict_dir_y:
            # Answer data exists
            loss = self.classifier(x, dict_dir_y[directory][0]).data
            print(f"data: {directory}")
            print(f"loss: {loss}")
            if save:
                with open(output_directory / 'loss.dat', 'w') as f:
                    f.write(f"loss: {loss}")
        else:
            # Answer data does not exist
            loss = None

        print(f"Inferred data saved in: {output_directory}")
        return inversed_dict_x, inversed_dict_y, loss

    def set_seed(self):
        seed = self.setting.trainer.seed
        random.seed(seed)
        np.random.seed(seed)
        if ch.cuda.available and self.setting.trainer.gpu_id >= 0:
            ch.cuda.cupy.random.seed(seed)
        return

    def _separate_data(self, data, descriptions, *, axis=2):
        data_dict = {}
        index = 0
        data = np.swapaxes(data, 0, axis)
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
        if model_file is None:
            model_file = self._select_snapshot(
                self.setting.trainer.pretrain_directory,
                method=self.setting.trainer.snapshot_choise_method)
        ch.serializers.load_npz(
            model_file, self.model, path='updater/model:main/predictor/')
        print(f"{model_file} loaded as a pretrain model.")
        return

    def _load_restart_model_if_needed(self):
        if self.setting.trainer.restart_directory is None:
            return
        snapshot = self._select_snapshot(
            self.setting.trainer.restart_directory, method='latest')
        ch.serializers.load_npz(snapshot, self.trainer)
        print(f"{snapshot} loaded for restart.")
        return

    def _select_snapshot(self, path, method='best'):
        if not path.exists():
            raise ValueError(f"{path} doesn't exist")

        if path.is_file():
            return path
        elif path.is_dir():
            snapshots = path.glob('snapshot_*')
            if method == 'latest':
                return max(snapshots, key=lambda p: p.stat().st_ctime)
            elif method == 'best':
                df = pd.read_json(path / 'log')
                best_epoch = df['epoch'].iloc[
                    df['validation/main/loss'].idxmin()]
                return path / f"snapshot_epoch_{best_epoch}"
            elif method == 'train_best':
                df = pd.read_json(path / 'log')
                best_epoch = df['epoch'].iloc[
                    df['main/loss'].idxmin()]
                return path / f"snapshot_epoch_{best_epoch}"
            else:
                raise ValueError(f"Unknown snapshot choise method: {method}")

        else:
            raise ValueError(f"{path} had unknown property.")

    def _is_gpu_supporting(self):
        return ch.cuda.available

    def _generate_trainer(
            self, x_variable_names, y_variable_names,
            train_directories, validation_directories, *,
            supports=None):

        x_train, support_train = self._load_data(
            x_variable_names, train_directories, supports=supports)
        y_train, _ = self._load_data(y_variable_names, train_directories)
        if supports is None:
            train_iter = ch.iterators.SerialIterator(
                ch.datasets.DictDataset(**{'x': x_train, 't': y_train}),
                batch_size=self.setting.trainer.batch_size, shuffle=True)
        else:
            train_iter = ch.iterators.SerialIterator(
                ch.datasets.DictDataset(**{
                    'x': x_train, 't': y_train, 'supports': support_train}),
                batch_size=self.setting.trainer.batch_size, shuffle=True)

        optimizer = self._create_optimizer()
        optimizer.setup(self.classifier)

        # Converter setting
        if self.setting.trainer.element_wise:
            converter = ch.dataset.concat_examples
        else:
            if self.setting.trainer.support_inputs is None:
                converter = util.concat_examples
            else:
                converter = util.generate_converter(support_train)

        # Updater setting
        if self.setting.trainer.use_siml_updater:
            updater = updaters.SimlUpdater(
                train_iter, optimizer, device=self.setting.trainer.gpu_id,
                converter=converter)
        else:
            if self.setting.trainer.element_batch_size >= 0:
                print(
                    f"When use_siml_updater: True, "
                    f"cannot set element_batch_size >= 0. Set to -1.")
                self.setting.trainer.element_batch_size = -1
            updater = ch.training.updaters.StandardUpdater(
                train_iter, optimizer, device=self.setting.trainer.gpu_id,
                converter=converter)

        stop_trigger = ch.training.triggers.EarlyStoppingTrigger(
            monitor='validation/main/loss', check_trigger=(
                self.setting.trainer.stop_trigger_epoch, 'epoch'),
            max_trigger=(self.setting.trainer.n_epoch, 'epoch'))

        trainer = ch.training.Trainer(
            updater, stop_trigger, out=self.setting.trainer.output_directory)

        self.log_report_extension = ch.training.extensions.LogReport(
            trigger=(self.setting.trainer.log_trigger_epoch, 'epoch'))
        trainer.extend(self.log_report_extension)
        trainer.extend(ch.training.extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
        trainer.extend(
            ch.training.extensions.PlotReport(
                ['main/loss', 'validation/main/loss'],
                'epoch',
                trigger=(self.setting.trainer.log_trigger_epoch, 'epoch')))
        trainer.extend(
            ch.training.extensions.snapshot(
                filename='snapshot_epoch_{.updater.epoch}'),
            trigger=(self.setting.trainer.log_trigger_epoch, 'epoch'))
        trainer.extend(ch.training.extensions.ProgressBar())

        if self.setting.trainer.prune:
            trainer.extend(
                optuna.integration.ChainerPruningExtension(
                    self.optuna_trial, 'validation/main/loss',
                    (self.setting.trainer.stop_trigger_epoch, 'epoch')))

        # Manage validation
        if len(validation_directories) > 0:
            x_validation, support_validation = self._load_data(
                x_variable_names, validation_directories, supports=supports)
            y_validation, _ = self._load_data(
                y_variable_names, validation_directories)
            if supports is None:
                validation_iter = ch.iterators.SerialIterator(
                    ch.datasets.DictDataset(**{
                        'x': x_validation, 't': y_validation}),
                    batch_size=self.setting.trainer.batch_size,
                    shuffle=False, repeat=False)
            else:
                validation_iter = ch.iterators.SerialIterator(
                    ch.datasets.DictDataset(**{
                        'x': x_validation, 't': y_validation,
                        'supports': support_validation}),
                    batch_size=self.setting.trainer.batch_size,
                    shuffle=False, repeat=False)
            trainer.extend(ch.training.extensions.Evaluator(
                validation_iter, self.classifier,
                device=self.setting.trainer.gpu_id, converter=converter))
        return trainer

    def _create_optimizer(self):
        optimizer_name = self.setting.trainer.optimizer.lower()
        if optimizer_name == 'adam':
            return ch.optimizers.Adam(**self.setting.trainer.optimizer_setting)
        else:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    def _create_loss_function(self):
        loss_name = self.setting.trainer.loss_function.lower()
        if loss_name == 'mse':
            return ch.functions.mean_squared_error
        else:
            raise ValueError(f"Unknown loss function name: {loss_name}")

    def _load_data(
            self, variable_names, directories, *,
            return_dict=False, supports=None):
        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=[f"{variable_names[0]}.npy"])

        if supports is None:
            supports = []

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

        data = [
            self._concatenate_variable([
                util.load_variable(data_directory, variable_name)
                for variable_name in variable_names])
            for data_directory in data_directories]
        support_data = [
            [
                util.load_variable(data_directory, support)
                for support in supports]
            for data_directory in data_directories]
        if self.setting.trainer.element_wise:
            if len(support_data[0]) > 0:
                raise ValueError(
                    'Cannot use support_input if element_wise is True')
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

    def _concatenate_variable(self, variables):

        concatenatable_variables = np.concatenate(
            [
                variable for variable in variables
                if isinstance(variable, np.ndarray)],
            axis=1)
        unconcatenatable_variables = [
            variable for variable in variables
            if not isinstance(variable, np.ndarray)]
        if len(unconcatenatable_variables) == 0:
            return concatenatable_variables
        else:
            return concatenatable_variables, unconcatenatable_variables
