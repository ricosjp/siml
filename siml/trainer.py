import random

import chainer as ch
import daz
import numpy as np
import optuna
import pandas as pd

# from femio import FEMData
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

        self.set_seed()

        # Define model
        self.model = networks.Network(self.setting.model, self.setting.trainer)
        self.classifier = networks.Classifier(
            self.model, lossfun=self._create_loss_function(),
            element_batch_size=self.setting.trainer.element_batch_size)
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

    def train(self):
        """Perform training.

        Args:
            None
        Returns:
            loss: float
                Overall loss value.
        """
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

    def infer(self):
        """Perform inference.

        Args:
            inference_directories: list of pathlib.Path
                Directories for inference.
        Returns:
            loss: float, optional (when the answer is available)
                Overall loss value.
        """

        if self.setting.trainer.pretrain_directory is None:
            raise ValueError(
                f'No pretrain directory is specified for inference.')

        dict_dir_x = self._load_data(
            self.setting.trainer.input_names, self.setting.data.test,
            return_dict=True)
        dict_dir_y = self._load_data(
            self.setting.trainer.output_names, self.setting.data.test,
            return_dict=True)

        postprocessor = prepost.Postprocessor.read_main_setting(self.setting)

        with ch.using_config('train', False):
            losses = np.array([
                self._infer_single_data(
                    postprocessor, directory, x, dict_dir_y)
                for directory, x in dict_dir_x.items()])
        return losses

    def _infer_single_data(self, postprocessor, directory, x, dict_dir_y):
        inferred_y = self.model(x).data
        dict_var_x = self._separate_data(x, self.setting.trainer.inputs)
        dict_var_inferred_y = self._separate_data(
            inferred_y, self.setting.trainer.outputs)

        output_directory = prepost.determine_output_directory(
            directory, self.setting.data.inferred,
            self.setting.data.preprocessed.stem) \
            / f"{self.setting.trainer.name}_{util.date_string()}"
        output_directory.mkdir(parents=True)

        postprocessor.postprocess(
            dict_var_x, dict_var_inferred_y,
            output_directory=output_directory)
        if directory in dict_dir_y:
            loss = self.classifier(x, dict_dir_y[directory]).data
            print(f"data: {directory}")
            print(f"loss: {loss}")
            with open(output_directory / 'loss.dat', 'w') as f:
                f.write(f"loss: {loss}")
        else:
            loss = None

        setting.write_yaml(self.setting, output_directory / 'settings.yml')
        print(f"Inferred data saved in: {output_directory}")
        return loss

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

    def _load_pretrained_model_if_needed(self):
        if self.setting.trainer.pretrain_directory is None:
            return
        snapshot = self._select_snapshot(
            self.setting.trainer.pretrain_directory,
            method=self.setting.trainer.snapshot_choise_method)
        ch.serializers.load_npz(
            snapshot, self.model, path='updater/model:main/predictor/')
        print(f"{snapshot} loaded as a pretrain model.")
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
        if self.setting.trainer.support_inputs is None:
            converter = ch.dataset.concat_examples
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
