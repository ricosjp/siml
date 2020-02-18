import enum
from pathlib import Path

import numpy as np
import pandas as pd

from . import setting
from . import trainer


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

        self.relative_train_sizes = np.linspace(
            *self.original_setting.study.relative_train_size_linspace)
        self.log_file_path = self.original_setting.study.root_directory \
            / 'study_long.csv'
        if not self.log_file_path.exists():
            self._initialize_log_file()

        return

    def _initialize_log_file(self):
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
        train_losses = [None] * length
        test_losses = [None] * length
        statuses = [Status.NOT_YET.value] * length
        data_frame = pd.DataFrame({
            'id': range(length),
            'relative_train_size': all_relative_train_sizes,
            'fold_id': all_fold_ids,
            'train_loss': train_losses,
            'test_loss': test_losses,
            'status': statuses,
        })
        data_frame.to_csv(self.log_file_path, index=0)
        return

    def run(self):
        df = pd.read_csv(self.log_file_path)
        condition = df.where(df['status'] != Status('FINISHED')).iloc[0]
        main_setting = self._create_setting(condition)
        self.trainer = trainer.Trainer(main_setting)

    def _create_setting(self, condition):


class Status(enum.Enum):
    NOT_YET = 'NOT_YET'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    ERROR = 'ERROR'
