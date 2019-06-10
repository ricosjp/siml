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

    def __init__(self, main_setting):
        """Initialize Trainer object.

        Args:
            main_setting: siml.setting.MainSetting object
                Setting descriptions.
            model: siml.network.Network object
                Model to be trained.
        Returns:
            None
        """
        self.setting = main_setting
        self._update_setting_if_needed()

        self.set_seed()

        # Define model
        self.model = networks.Network(self.setting.model)
        self.classifier = ch.links.Classifier(
            self.model, lossfun=self._create_loss_function())
        self.classifier.compute_accuracy = \
            self.setting.trainer.compute_accuracy

        # Manage settings
        if self.setting.trainer.optuna_trial is None \
                and self.setting.trainer.prune:
            raise ValueError(f"Cannot prune without optuna_trial. Feed it.")
        if self._is_gpu_supporting():
            self.setting.trainer.gpu_id = self.setting.trainer.gpu_id
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
            self.setting.data.train, self.setting.data.validation)

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
        loss = self.log_report_extension.log[-1]['validation/main/loss']
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

    def _update_setting(self, path):
        if path.is_file():
            yaml_file = path
        elif path.is_dir():
            yamls = list(path.glob('*.yaml'))
            if len(yamls) != 1:
                raise ValueError(f"{len(yamls)} yaml files found in {path}")
            yaml_file = yamls[0]
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
            self._update_setting(self.setting.trainer.pretrain_directory)
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
            train_directories, validation_directories):

        x_train = self._load_data(x_variable_names, train_directories)
        y_train = self._load_data(y_variable_names, train_directories)
        train_iter = ch.iterators.SerialIterator(
            ch.datasets.TupleDataset(x_train, y_train),
            batch_size=self.setting.trainer.batch_size, shuffle=True)

        optimizer = self._create_optimizer()
        optimizer.setup(self.classifier)
        updater = ch.training.StandardUpdater(
            train_iter, optimizer, device=self.setting.trainer.gpu_id)
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
            ['epoch', 'main/loss', 'validation/main/loss']))
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
                    self.setting.trainer.trial, 'validation/main/loss',
                    (self.setting.trainer.stop_trigger_epoch, 'epoch')))

        # Manage validation
        if len(validation_directories) > 0:
            x_validation = self._load_data(
                x_variable_names, validation_directories)
            y_validation = self._load_data(
                y_variable_names, validation_directories)
            validation_iter = ch.iterators.SerialIterator(
                ch.datasets.TupleDataset(x_validation, y_validation),
                batch_size=self.setting.trainer.batch_size,
                shuffle=False, repeat=False)
            trainer.extend(ch.training.extensions.Evaluator(
                validation_iter, self.classifier))
        return trainer

    def _create_optimizer(self):
        optimizer_name = self.setting.trainer.optimizer.lower()
        if optimizer_name == 'adam':
            return ch.optimizers.Adam()
        else:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    def _create_loss_function(self):
        loss_name = self.setting.trainer.loss_function.lower()
        if loss_name == 'mse':
            return ch.functions.mean_squared_error
        else:
            raise ValueError(f"Unknown loss function name: {loss_name}")

    def _load_data(self, variable_names, directories, *, return_dict=False):
        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=['*.npy'])

        data = [
            self._concatenate_variable([
                util.load_variable(data_directory, variable_name)
                for variable_name in variable_names])
            for data_directory in data_directories]
        if return_dict:
            return {
                data_directory: d[None, :]

                for data_directory, d in zip(data_directories, data)}
        else:
            return data

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
        elif len(unconcatenatable_variables) == 1:
            return concatenatable_variables, unconcatenatable_variables
        else:
            raise ValueError(
                f"{len(unconcatenatable_variables)} "
                'unconcatenatable variables found.\n'
                'Must be less than 2.')
