import chainer as ch
import daz
import numpy as np
import optuna

# from femio import FEMData
from . import util
from . import networks
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
            self.setting.trainer.inputs, self.setting.trainer.outputs,
            self.setting.data.train, self.setting.data.validation)

    def train(self):
        """Perform training.

        Args:
            None
        Returns:
            loss: float
                Overall loss value.
        """
        self.setting.trainer.output_directory.mkdir(parents=True)
        self.setting.write_yaml(
            self.setting.trainer.output_directory / 'settings.yaml')

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
