import pathlib
from typing import Union, Optional

import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Optimizer

from siml.networks import Network
from siml.path_like_objects import SimlFileBuilder
from siml.services.training.logging_items import ILoggingItem, create_logitems
from siml.setting import TrainerSetting


class LogRecordItems:
    def __init__(
        self,
        *,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        train_other_losses: dict[str, float],
        validation_other_losses: dict[str, float],
        elapsed_time: float,
        train_loss_details: Optional[dict[str, float]] = None,
        validation_loss_details: Optional[dict[str, float]] = None
    ) -> None:
        self.epoch = create_logitems(int(epoch), "epoch")
        self.train_loss = create_logitems(train_loss, "train_loss")
        self.validation_loss = create_logitems(
            validation_loss, "validation_loss"
        )
        self.train_other_losses = create_logitems(
            train_other_losses, "train/"
        )
        self.validation_other_losses = create_logitems(
            validation_other_losses, "validation/"
        )
        self.elapsed_time = create_logitems(
            elapsed_time, "elapsed_time"
        )
        train_loss_details = self._create_empty_if_none(train_loss_details)
        self.train_loss_details = create_logitems(
            train_loss_details, "train_loss_details/"
        )

        validation_loss_details = \
            self._create_empty_if_none(validation_loss_details)
        self.validation_loss_details = create_logitems(
            validation_loss_details, "validation_loss_details/"
        )

        self._loss_detail_keys = list(train_loss_details.keys())

    def _create_empty_if_none(self, val: Union[dict, None]) -> dict:
        if val is None:
            return {}
        return val


class SimlTrainingConsoleLogger:
    def __init__(
        self,
        display_margin: int,
        loss_keys: list[str],
        output_names: list[str],
    ) -> None:
        self._display_margin = display_margin
        self._loss_keys = loss_keys
        self._output_names = output_names
        self._train_details_title = r"tr_DTL/"
        self._validation_details_title = r"vl_DTL/"

        self._headers = self._get_headers()

    def _get_headers(self) -> list[ILoggingItem]:
        headers = [
            'epoch',
            'train_loss',
            *[f"train/{k}, " for k in self._loss_keys],
            'validation_loss',
            *[f"validation/{k}" for k in self._loss_keys],
            'elapsed_time',
            *[f"{self._train_details_title}{k}" for k in self._output_names],
            *[
                f"{self._validation_details_title}{k}"
                for k in self._output_names
            ],
        ]
        headers = [create_logitems(v) for v in headers]
        return headers

    def get_header(self) -> str:
        strings = [
            v.format(padding_margin=self._display_margin)
            for v in self._headers
        ]
        return "\n" + "".join(strings)

    def output_header(self) -> None:
        header = self.get_header()
        # HACK: should use logger instead.
        print(header)

    def output(self, log_record: LogRecordItems) -> str:
        strings = [
            log_record.epoch.format(padding_margin=self._display_margin),
            log_record.train_loss.format(
                formatter=".5e", padding_margin=self._display_margin
            ),
            log_record.train_other_losses.format(
                formatter=".5e", padding_margin=self._display_margin
            ),
            log_record.validation_loss.format(
                formatter=".5e", padding_margin=self._display_margin
            ),
            log_record.validation_other_losses.format(
                formatter=".5e", padding_margin=self._display_margin
            ),
            log_record.elapsed_time.format(
                formatter=".2f", padding_margin=self._display_margin
            ),
            log_record.train_loss_details.format(
                formatter=".5e", key_orders=self._output_names,
                padding_margin=self._display_margin,
                title=self._train_details_title
            ),
            log_record.validation_loss_details.format(
                formatter=".5e", key_orders=self._output_names,
                padding_margin=self._display_margin,
                title=self._validation_details_title
            )
        ]
        return "".join(strings)


class SimlTrainingFileLogger:
    def __init__(
        self,
        file_path: pathlib.Path,
        loss_figure_path: pathlib.Path,
        loss_keys: list[str],
        output_names: list[str],
        continue_mode: bool = False,
    ) -> None:
        self._file_path = file_path
        self._loss_figure_path = loss_figure_path
        self._loss_keys = loss_keys
        self._output_names = output_names
        self._continue_mode = continue_mode

        self._headers = self._get_headers()

    def read_offset_start_time(self) -> float:
        if not self._continue_mode:
            return 0

        df = self.read_history()
        offset_start_time = df.tail(1).loc[:, "elapsed_time"].item()
        if offset_start_time is None:
            return 0
        return offset_start_time

    def _get_headers(self) -> list[ILoggingItem]:
        headers = [
            'epoch',
            'train_loss',
            *[f"train/{k}" for k in self._loss_keys],
            'validation_loss',
            *[f"validation/{k}" for k in self._loss_keys],
            'elapsed_time',
            *[f"train_loss_details/{k}" for k in self._output_names],
            *[f"validation_loss_details/{k}" for k in self._output_names],
        ]
        headers = [create_logitems(v) for v in headers]
        return headers

    def _header_strings(self) -> str:
        headers = [v.format() for v in self._headers]
        headers = [v for v in headers if len(v) > 0]
        return ", ".join(headers)

    def write_header_if_needed(self) -> None:
        if self._continue_mode:
            return

        header_str = self._header_strings()
        with open(self._file_path, 'w') as fw:
            fw.write(header_str + '\n')

    def write(self, log_record: LogRecordItems) -> None:
        values = [
            log_record.epoch.format(),
            log_record.train_loss.format(formatter=".5e"),
            log_record.train_other_losses.format(formatter=".5e"),
            log_record.validation_loss.format(formatter=".5e"),
            log_record.validation_other_losses.format(formatter=".5e"),
            log_record.elapsed_time.format(formatter=".2f"),
            log_record.train_loss_details.format(
                formatter=".5e", key_orders=self._output_names),
            log_record.validation_loss_details.format(
                formatter=".5e", key_orders=self._output_names)
        ]
        values = [v for v in values if len(v) > 0]
        with open(self._file_path, 'a') as fw:
            fw.write(", ".join(values) + '\n')

    def read_history(self) -> pd.DataFrame:
        df = pd.read_csv(
            self._file_path,
            header=0,
            index_col=None,
            skipinitialspace=True
        )
        return df

    def save_figure(self) -> None:
        fig = plt.figure(figsize=(16 / 2, 9 / 2))
        df = self.read_history()
        plt.plot(df['epoch'], df['train_loss'], label='train loss')
        plt.plot(
            df['epoch'], df['validation_loss'], label='validation loss')
        for k in self._loss_keys:
            plt.plot(df['epoch'], df[f"train/{k}"], label=f"train/{k}")
        for k in self._loss_keys:
            plt.plot(
                df['epoch'], df[f"validation/{k}"],
                label=f"validation/{k}")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(self._loss_figure_path)
        plt.close(fig)

    def save_model(
        self,
        epoch: int,
        model: Network,
        optimizer: Optimizer,
        validation_loss: float,
        trainer_setting: TrainerSetting
    ) -> None:
        enc_ext = ".enc" if trainer_setting.model_key is not None else ""
        file_path = trainer_setting.output_directory / \
            pathlib.Path(f"snapshot_epoch_{epoch}.pth{enc_ext}")
        siml_file = SimlFileBuilder.checkpoint_file(file_path)

        data = {
            'epoch': epoch,
            'validation_loss': validation_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        siml_file.save(
            data,
            encrypt_key=trainer_setting.model_key
        )


class TrainDataDebugLogger:
    def __init__(self, file_path: pathlib.Path) -> None:
        self._file_path = file_path

    def _header_strings(self) -> list[str]:
        return ["data_directory"]

    def write_epoch(self, epoch: int):
        with open(self._file_path, 'a') as fw:
            fw.write(f"epoch: {epoch} finished ---------- \n")

    def write(
        self, data_directories: Union[list[pathlib.Path], None]
    ) -> None:
        if data_directories is None:
            values = []
        else:
            values = [str(path) for path in data_directories]

        with open(self._file_path, 'a') as fw:
            fw.write(", ".join(values) + '\n')
