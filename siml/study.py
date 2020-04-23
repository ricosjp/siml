import enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

from . import prepost
from . import setting
from . import trainer
from . import util


class Study():

    def __init__(self, settings):
        if isinstance(settings, Path):
            self.original_setting = setting.MainSetting.read_settings_yaml(
                settings)
        elif isinstance(settings, setting.MainSetting):
            self.original_setting = settings
        else:
            raise ValueError(
                f"Unknown type for settings: {settings.__class__}")

        if self.original_setting.study.root_directory is None:
            self.original_setting.study.root_directory = \
                self.original_setting.trainer.name

        self.relative_develop_sizes = np.linspace(
            *self.original_setting.study.relative_develop_size_linspace)
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

        all_relative_develop_sizes = [
            size
            for size in self.relative_develop_sizes
            for i_fold in range(self.original_setting.study.n_fold)]
        all_fold_ids = [
            i_fold
            for size in self.relative_develop_sizes
            for i_fold in range(self.original_setting.study.n_fold)]
        length = len(all_relative_develop_sizes)
        nones = [None] * length
        statuses = [Status.NOT_YET.value] * length

        data_frame = pd.DataFrame({
            'id': range(length),
            'relative_develop_size': all_relative_develop_sizes,
            'fold_id': all_fold_ids,
            'train_loss': nones,
            'validation_loss': nones,
            'test_loss': nones,
            'status': statuses,
        })
        data_frame.to_csv(self.log_file_path, index=False, header=True)

    def initialize_study_setting(self):
        template_trainer = trainer.Trainer(self.original_setting)
        template_trainer.prepare_training(draw=False)
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
            conditions = df[
                (df.status != Status('FINISHED').value)
                & (df.status != Status('RUNNING').value)]
            if len(conditions) == 0:
                break
            condition = conditions.iloc[0]
            self.run_single(condition)

        self.plot_study(allow_nan=False)

    def run_single(self, condition):
        condition.status = Status.RUNNING.value
        df = pd.read_csv(self.log_file_path, header=0)
        df.iloc[condition.id] = condition
        df.to_csv(self.log_file_path, index=False, header=True)
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

        self.plot_study(allow_nan=True)

        return

    def plot_study(self, allow_nan=False):
        original_df = pd.read_csv(self.log_file_path, header=0)
        df_wo_nan = original_df.dropna()
        if not allow_nan and len(original_df) != len(df_wo_nan):
            raise ValueError(f"{self.log_file_path} contains NaN")

        train_loss_means, train_loss_stds = self._calculate_stats(
            df_wo_nan, 'train_loss')
        validation_loss_means, validation_loss_stds = self._calculate_stats(
            df_wo_nan, 'validation_loss')
        test_loss_means, test_loss_stds = self._calculate_stats(
            df_wo_nan, 'test_loss')
        sizes = self.relative_develop_sizes * self.total_data_size * (
            self.original_setting.study.n_fold - 1
        ) / self.original_setting.study.n_fold

        cmap = plt.get_cmap('tab10')
        plt.figure()
        plt.plot(
            sizes, train_loss_means, '.:',
            label='train loss', color=cmap(0))
        plt.fill_between(
            sizes, train_loss_means - train_loss_stds,
            train_loss_means + train_loss_stds, alpha=.15, color=cmap(0))
        if self.original_setting.study.plot_validation:
            plt.plot(
                sizes, validation_loss_means, '+--',
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
        if self.original_setting.study.scale_loss:
            y_name = self.original_setting.trainer.output_names[0]
            unit = self.original_setting.study.unit_error
        else:
            y_name = 'Loss'
            unit = '-'
        plt.ylabel(f"{y_name} [{unit}]")
        if self.original_setting.study.x_from_zero:
            plt.xlim(0., None)
        if self.original_setting.study.y_from_zero:
            plt.ylim(0., None)
        if self.original_setting.study.x_logscale:
            plt.xscale('log')
        if self.original_setting.study.y_logscale:
            plt.yscale('log')
        plt.legend()
        plt.savefig(
            self.original_setting.study.root_directory / 'learning_curve.pdf')

        plt.close()
        return

    def _calculate_stats(self, df, key):
        if self.original_setting.study.scale_loss:
            scale_conversion_function = \
                self._generate_scale_conversion_function()
        else:
            def identity(x):
                return x
            scale_conversion_function = identity

        means = scale_conversion_function(np.array([
            np.mean(df[abs(df.relative_develop_size - s) < 1e-5][key])
            for s in self.relative_develop_sizes]))
        stds = scale_conversion_function(np.array([
            np.std(df[abs(df.relative_develop_size - s) < 1e-5][key])
            for s in self.relative_develop_sizes]))
        return means, stds

    def _generate_scale_conversion_function(self):
        preprocessors_pkl = self.original_setting.data.preprocessed \
            / 'preprocessors.pkl'
        converter = prepost.Converter(preprocessors_pkl)
        output_names = self.original_setting.trainer.output_names
        if len(output_names) != 1:
            raise NotImplementedError(
                f"Output names more than 1 cannot be converted automatically")
        else:
            output_name = output_names[0]

        output_converter = converter.converters[output_name].converter
        # NOTE: Assume loss is mean square error
        if isinstance(output_converter, preprocessing.StandardScaler):
            def scale_conversion_function(x):
                return (x * output_converter.var_)**.5

        elif isinstance(output_converter, preprocessing.MinMaxScaler):
            def scale_conversion_function(x):
                return (x * output_converter.scale_)**.5

        elif isinstance(output_converter, preprocessing.MaxAbsScaler):
            def scale_conversion_function(x):
                return (x * output_converter.max_abs_)**.5

        else:
            raise ValueError(f"Unknown converter type: {output_converter}")

        return scale_conversion_function

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
            self.total_data_size * condition.relative_develop_size))


class Status(enum.Enum):
    NOT_YET = 'NOT_YET'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    ERROR = 'ERROR'
