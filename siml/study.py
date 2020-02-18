import enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import setting
from . import trainer
from . import util


class Study():

    def __init__(
            self, settings, *, scale_conversion_function=None):
        if isinstance(settings, Path):
            self.original_setting = setting.MainSetting.read_settings_yaml(
                settings)
        elif isinstance(settings, setting.MainSetting):
            self.original_setting = settings
        else:
            raise ValueError(
                f"Unknown type for settings: {settings.__class__}")

        if scale_conversion_function is None:
            def scale_conversion_function(x):
                return x
        self.scale_conversion_function = scale_conversion_function

        if self.original_setting.study.root_directory is None:
            self.original_setting.study.root_directory = \
                self.original_setting.trainer.name

        self.relative_train_sizes = np.linspace(
            *self.original_setting.study.relative_train_size_linspace)
        self.log_file_path = self.original_setting.study.root_directory \
            / 'study_log.csv'
        if self.log_file_path.exists():
            self.study_setting = setting.MainSetting.read_settings_yaml(
                list(self.original_setting.study.root_directory.glob(
                    'study_setting.yml'))[0])
        else:
            self.initialize_log_file()
            self.study_setting = self.initialize_study_setting()
            setting.write_yaml(
                self.study_setting,
                self.original_setting.study.root_directory
                / 'study_setting.yml')

        self.total_data_directories = self.study_setting.data.train
        self.total_data_size = len(self.total_data_directories)

        return

    def initialize_log_file(self):
        self.original_setting.study.root_directory.mkdir(parents=True)

        all_relative_train_sizes = [
            size
            for size in self.relative_train_sizes
            for i_fold in range(self.original_setting.study.n_fold)]
        all_fold_ids = [
            i_fold
            for size in self.relative_train_sizes
            for i_fold in range(self.original_setting.study.n_fold)]
        length = len(all_relative_train_sizes)
        nones = [None] * length
        statuses = [Status.NOT_YET.value] * length

        data_frame = pd.DataFrame({
            'id': range(length),
            'relative_train_size': all_relative_train_sizes,
            'fold_id': all_fold_ids,
            'train_loss': nones,
            'validation_loss': nones,
            'test_loss': nones,
            'status': statuses,
        })
        data_frame.to_csv(self.log_file_path, index=False, header=True)

    def initialize_study_setting(self):
        template_trainer = trainer.Trainer(self.original_setting)
        template_trainer.prepare_training()
        total_data_directories = np.unique(np.concatenate([
            template_trainer.train_loader.dataset.data_directories,
            template_trainer.validation_loader.dataset.data_directories]))

        update_dict = {
            'data': {
                'train': total_data_directories,
                'validation': [],
            },
            'trainer': {
                'output_directory':
                self.original_setting.study.root_directory,
            },
        }
        new_setting = self.original_setting.update_with_dict(update_dict)
        return new_setting

    def run(self):
        df = pd.read_csv(self.log_file_path, header=0)
        total_trial_length = len(df)
        for _ in range(total_trial_length):
            df = pd.read_csv(self.log_file_path, header=0)
            df = pd.read_csv(self.log_file_path, header=0)
            conditions = df[df.status != Status('FINISHED').value]
            if len(conditions) == 0:
                break
            condition = conditions.iloc[0]
            self.run_single(condition)

        self.finalize_study()

    def run_single(self, condition):
        print('Training started with the following condition:')
        print(condition)
        main_setting = self._create_setting(condition)
        try:
            tr = trainer.Trainer(main_setting)
            tr.train()
        except Exception as err:
            condition.status = Status.ERROR.value
            df = pd.read_csv(self.log_file_path, header=0)
            df.iloc[condition.id] = condition
            df.to_csv(self.log_file_path, index=False, header=True)
            raise err

        train_state, validation_state, test_state = tr.evaluate(
            evaluate_test=True, load_best_model=True)
        train_loss = train_state.metrics['loss']
        validation_loss = validation_state.metrics['loss']
        test_loss = test_state.metrics['loss']
        condition.train_loss = train_loss
        condition.validation_loss = validation_loss
        condition.test_loss = test_loss
        condition.status = Status.FINISHED.value
        print('Training finished with the following condition:')
        print(condition)
        print('--')

        df = pd.read_csv(self.log_file_path, header=0)
        df.iloc[condition.id] = condition
        df.to_csv(self.log_file_path, index=False, header=True)

        return

    def finalize_study(self):
        df = pd.read_csv(self.log_file_path, header=0)

        train_loss_means, train_loss_stds = self._calculate_stats(
            df, 'train_loss')
        validation_loss_means, validation_loss_stds = self._calculate_stats(
            df, 'validation_loss')
        test_loss_means, test_loss_stds = self._calculate_stats(
            df, 'test_loss')
        sizes = self.relative_train_sizes * self.total_data_size

        cmap = plt.get_cmap('tab10')
        plt.plot(
            sizes, train_loss_means, '.:',
            label='training loss', color=cmap(0))
        plt.fill_between(
            sizes, train_loss_means - train_loss_stds,
            train_loss_means + train_loss_stds, alpha=.15, color=cmap(0))
        if self.original_setting.study.plot_validation:
            plt.plot(
                sizes, validation_loss_means, '.-',
                label='validation loss', color=cmap(1))
            plt.fill_between(
                sizes, validation_loss_means - validation_loss_stds,
                validation_loss_means + validation_loss_stds, alpha=.15,
                color=cmap(1))
        plt.plot(
            sizes, test_loss_means, '*-',
            label='test loss', color=cmap(2))
        plt.fill_between(
            sizes, test_loss_means - test_loss_stds,
            test_loss_means + test_loss_stds, alpha=.15, color=cmap(2))

        plt.xlabel('Training sample size [-]')
        plt.ylabel(f"Loss [{self.original_setting.study.unit_error}]")
        plt.legend()
        plt.savefig(
            self.original_setting.study.root_directory / 'learning_curve.png')
        return

    def _calculate_stats(self, df, key):
        means = self.scale_conversion_function(np.array([
            np.mean(df[abs(df.relative_train_size - s) < 1e-5][key])
            for s in self.relative_train_sizes]))
        stds = self.scale_conversion_function(np.array([
            np.std(df[abs(df.relative_train_size - s) < 1e-5][key])
            for s in self.relative_train_sizes]))
        return means, stds

    def _create_setting(self, condition):
        fold_id = condition.fold_id
        if fold_id == 0:
            self._determine_develop_data_directories(condition)

        develop_data_directories = self._load_develop_data_directories(
            condition)

        return self._initialize_setting(condition, develop_data_directories)

    def _load_develop_data_directories(self, condition):
        develop_data_size = self._determine_develop_data_size(condition)
        directories = np.loadtxt(str(
            self.original_setting.study.root_directory
            / f"develop_data_size{develop_data_size}.dat"), dtype=str)
        return [Path(d) for d in directories]

    def _determine_develop_data_directories(self, condition):
        develop_data_directories = \
            self._extract_develop_data_data_directories(condition)
        develop_data_size = self._determine_develop_data_size(condition)
        np.savetxt(
            self.original_setting.study.root_directory
            / f"develop_data_size{develop_data_size}.dat",
            [str(d) for d in develop_data_directories],
            fmt='%s')
        return develop_data_directories

    def _initialize_setting(self, condition, develop_data_directories):
        fold_id = condition.fold_id

        split_data_directories = np.array_split(
            develop_data_directories, self.original_setting.study.n_fold)

        former = split_data_directories[:fold_id]
        latter = split_data_directories[fold_id+1:]
        if len(former) > 0:
            former = np.concatenate(former).tolist()
        if len(latter) > 0:
            latter = np.concatenate(latter).tolist()
        train_data_directories = former + latter
        validation_data_directories = split_data_directories[fold_id].tolist()
        develop_data_size = self._determine_develop_data_size(condition)

        update_dict = {
            'data': {
                'train': train_data_directories,
                'validation': validation_data_directories,
            },
            'trainer': {
                'output_directory':
                self.original_setting.study.root_directory
                / f"size{develop_data_size}_fold{fold_id}_{util.date_string()}"
            },
        }
        new_setting = self.original_setting.update_with_dict(update_dict)
        return new_setting

    def _extract_develop_data_data_directories(self, condition):
        develop_data_size = self._determine_develop_data_size(condition)
        develop_data_directories = np.random.permutation(
            self.total_data_directories)[:develop_data_size]
        return develop_data_directories

    def _determine_develop_data_size(self, condition):
        return int(round(
            self.total_data_size * condition.relative_train_size))


class Status(enum.Enum):
    NOT_YET = 'NOT_YET'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    ERROR = 'ERROR'
