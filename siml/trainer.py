from pathlib import Path

import chainer as ch
import daz
import numpy as np
import optuna

# from femio import FEMData
from . import util
from . import networks


class Trainer():

    @classmethod
    def read_settings(
            cls, settings_yaml, output_directory, *,
            gpu_id=-1):
        """Read settings.yaml to generate Trainer object.

        Args:
            settings_yaml: str or pathlib.Path
                setting.yaml file name.
            output_directory: str or pathlib.Path
                Output directory name.
            gpu_id: int, optional [-1]
                GPU ID. Specify non negative value to use GPU. -1 Meaning CPU.
        Returns:
            trainer: siml.Trainer
                Generater Trainer object.
        """
        dict_settings = util.load_yaml_file(settings_yaml)
        model = networks.Network(dict_settings['model'])
        return cls(
            model, dict_settings['trainer']['inputs'],
            dict_settings['trainer']['outputs'],
            dict_settings['data']['train'], output_directory,
            validation_directories=dict_settings['data']['validation'],
            batch_size=dict_settings['trainer']['batch_size'],
            n_epoch=dict_settings['trainer']['n_epoch'],
            log_trigger_epoch=dict_settings['trainer']['log_trigger_epoch'],
            stop_trigger_epoch=dict_settings['trainer']['stop_trigger_epoch'],
            gpu_id=gpu_id,
        )

    def __init__(
            self, model, x_variable_names, y_variable_names,
            train_directories, output_directory, *,
            validation_directories=[],
            restart_directory=None, pretrain_diectory=None,
            loss_function=ch.functions.mean_squared_error,
            optimizer=ch.optimizers.Adam,
            compute_accuracy=False, batch_size=10, n_epoch=1000, gpu_id=-1,
            log_trigger_epoch=1, stop_trigger_epoch=10,
            optuna_trial=None, prune=False):
        """Initialize Trainer object.

        Args:
            model: siml.network.Network object
                Model to be trained.
            x_variable_names: list of str
                Variable names of inputs.
            y_variable_names: list of str
                Variable names of outputs.
            train_directories: list of str or pathlib.Path
                Training data directories.
            output_directory: str or pathlib.Path
                Output directory name.
            validation_directories, list of str or pathlib.Path, optional [[]]
                Validation data directories.
            restart_directory: str or pathlib.Path, optional [None]
                Directory name to be used for restarting.
            pretrain_diectory: str or pathlib.Path, optional [None]
                Pretrained directory name.
            loss_function: chainer.FunctionNode,
                    optional [chainer.functions.mean_squared_error]
                Loss function to be used for training.
            optimizer: chainer.Optimizer,
                    optional [chainer.optimizers.Adam]
                Optimizer to be used for training.
            compute_accuracy: bool, optional [False]
                If True, compute accuracy.
            batch_size: int, optional [10]
                Batch size.
            n_epoch: int, optional [1000]
                The number of epochs.
            gpu_id: int, optional [-1]
                GPU ID. Specify non negative value to use GPU. -1 Meaning CPU.
            log_trigger_epoch: int, optional [1]
                The interval of logging of training. It is used for logging,
                plotting, and saving snapshots.
            stop_trigger_epoch: int, optional [10]
                The interval to check if training should be stopped. It is used
                for early stopping and pruning.
            optuna_trial: optuna.Trial [None]
                Trial object used to perform optuna hyper parameter tuning.
            prune: bool [False]
                If True and optuna_trial is given, prining would be performed.
        Returns:
            None
        """

        # Define model
        self.model = model
        self.classifier = ch.links.Classifier(
            self.model, lossfun=loss_function)
        self.classifier.compute_accuracy = compute_accuracy

        # Manage settings
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.optimizer = optimizer
        self.log_trigger_epoch = log_trigger_epoch
        self.stop_trigger_epoch = stop_trigger_epoch
        self.output_directory = output_directory
        self.restart_directory = restart_directory
        self.pretrain_diectory = pretrain_diectory
        self.optuna_trial = optuna_trial
        self.prune = prune
        if self.optuna_trial is None and self.prune:
            raise ValueError(f"Cannot prune without optuna_trial. Feed it.")
        if self._is_gpu_supporting():
            self.gpu_id = gpu_id
        else:
            if gpu_id != -1:
                print(f"GPU not found. Using CPU.")
            self.gpu_id = -1
            daz.set_daz()
            daz.set_ftz()

        # Generate trainer
        self.train_directories = [Path(d) for d in train_directories]
        self.validation_directories = [Path(d) for d in validation_directories]
        self.trainer = self._generate_trainer(
            x_variable_names, y_variable_names,
            self.train_directories, self.validation_directories)

    def train(self):
        """Perform training.

        Args:
            None
        Returns:
            loss: float
                Overall loss value.
        """
        self.trainer.run()
        loss = self.log_report_extension.log[-1]['validation/main/loss']
        return loss

    def _is_gpu_supporting(self):
        return ch.cuda.available

    def _generate_trainer(
            self, x_variable_names, y_variable_names,
            train_directories, validation_directories):

        x_train = self._load_data(x_variable_names, train_directories)
        y_train = self._load_data(y_variable_names, train_directories)
        train_iter = ch.iterators.SerialIterator(
            ch.datasets.TupleDataset(x_train, y_train),
            batch_size=self.batch_size, shuffle=True)

        optimizer = self.optimizer()
        optimizer.setup(self.classifier)
        updater = ch.training.StandardUpdater(
            train_iter, optimizer, device=self.gpu_id)
        stop_trigger = ch.training.triggers.EarlyStoppingTrigger(
            monitor='validation/main/loss', check_trigger=(
                self.stop_trigger_epoch, 'epoch'),
            max_trigger=(self.n_epoch, 'epoch'))

        trainer = ch.training.Trainer(
            updater, stop_trigger, out=self.output_directory)

        self.log_report_extension = ch.training.extensions.LogReport(
            trigger=(self.log_trigger_epoch, 'epoch'))
        trainer.extend(self.log_report_extension)
        trainer.extend(ch.training.extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss']))
        trainer.extend(
            ch.training.extensions.PlotReport(
                ['main/loss', 'validation/main/loss'],
                'epoch', trigger=(self.log_trigger_epoch, 'epoch')))
        trainer.extend(
            ch.training.extensions.snapshot(
                filename='snapshot_epoch_{.updater.epoch}'),
            trigger=(self.log_trigger_epoch, 'epoch'))
        trainer.extend(ch.training.extensions.ProgressBar())

        if self.prune:
            trainer.extend(
                optuna.integration.ChainerPruningExtension(
                    self.trial, 'validation/main/loss',
                    (self.stop_trigger_epoch, 'epoch')))

        # Manage validation
        if len(validation_directories) > 0:
            x_validation = self._load_data(
                x_variable_names, validation_directories)
            y_validation = self._load_data(
                y_variable_names, validation_directories)
            validation_iter = ch.iterators.SerialIterator(
                ch.datasets.TupleDataset(x_validation, y_validation),
                batch_size=self.batch_size, shuffle=False, repeat=False)
            trainer.extend(ch.training.extensions.Evaluator(
                validation_iter, self.classifier))
        return trainer

    def _load_data(self, variable_names, directories):
        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=['*.npy'])

        data = [
            self._concatenate_variable([
                util.load_variable(data_directory, variable_name)
                for variable_name in variable_names])
            for data_directory in data_directories]
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
