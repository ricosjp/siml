import abc
import pathlib

import numpy as np
import pandas as pd

from siml.base.siml_const import SimlConstItems
from siml.base.siml_enums import ModelSelectionType, SimlFileExtType
from siml.path_like_objects import ISimlCheckpointFile, SimlFileBuilder


class IModelSelector(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def select_model(
        dir_path: pathlib.Path,
        **kwards
    ) -> ISimlCheckpointFile:
        raise NotImplementedError()


class ModelSelectorBuilder:
    @staticmethod
    def create(
        selection_name: str
    ) -> IModelSelector:
        class_dict = {
            ModelSelectionType.BEST.value: BestModelSelector,
            ModelSelectionType.LATEST.value: LatestModelSelector,
            ModelSelectionType.SPECIFIED.value: SpecifiedModelSelector,
            ModelSelectionType.TRAIN_BEST.value: TrainBestModelSelector,
            ModelSelectionType.DEPLOYED.value: DeployedModelSelector
        }
        return class_dict[selection_name]


class BestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        **kwards
    ) -> ISimlCheckpointFile:
        snapshots = [
            SimlFileBuilder.checkpoint_file(p) for p in
            list(dir_path.glob('snapshot_epoch_*'))
        ]
        if len(snapshots) == 0:
            raise FileNotFoundError(
                f"snapshot file does not exist in {dir_path}"
            )

        df = pd.read_csv(
            dir_path / 'log.csv',
            header=0,
            index_col=None,
            skipinitialspace=True)
        if np.any(np.isnan(df['validation_loss'])):
            # HACK: BETTER TO RAISE ERROR
            # raise ValueError("NaN value is found in validation result.")
            print(
                "'best' model not found: NaN value is found in "
                "validation result."
                "'train_best' options is used instead."
            )
            return TrainBestModelSelector.select_model(dir_path, **kwards)

        best_epoch = df['epoch'].iloc[df['validation_loss'].idxmin()]
        target_snapshot: ISimlCheckpointFile = \
            [p for p in snapshots if p.epoch == best_epoch][0]
        return target_snapshot


class LatestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        **kwards
    ) -> ISimlCheckpointFile:
        snapshots = [
            SimlFileBuilder.checkpoint_file(p) for p in
            list(dir_path.glob('snapshot_epoch_*'))
        ]
        if len(snapshots) == 0:
            raise FileNotFoundError(
                f"snapshot file does not exist in {dir_path}"
            )

        max_epoch = max([p.epoch for p in snapshots])
        target_snapshot: ISimlCheckpointFile = \
            [p for p in snapshots if p.epoch == max_epoch][0]
        return target_snapshot


class TrainBestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        **kwards
    ) -> ISimlCheckpointFile:
        snapshots = [
            SimlFileBuilder.checkpoint_file(p) for p in
            list(dir_path.glob('snapshot_epoch_*'))
        ]
        if len(snapshots) == 0:
            raise FileNotFoundError(
                f"snapshot file does not exist in {dir_path}"
            )

        df = pd.read_csv(
            dir_path / 'log.csv',
            header=0,
            index_col=None,
            skipinitialspace=True)
        best_epoch = df['epoch'].iloc[df['train_loss'].idxmin()]
        target_snapshot: ISimlCheckpointFile = \
            [p for p in snapshots if p.epoch == best_epoch][0]
        return target_snapshot


class SpecifiedModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        *,
        infer_epoch: int,
        **kwards
    ) -> ISimlCheckpointFile:
        snapshots = [
            SimlFileBuilder.checkpoint_file(p) for p in
            list(dir_path.glob('snapshot_epoch_*'))
        ]
        target_snapshots: ISimlCheckpointFile = \
            [p for p in snapshots if p.epoch == infer_epoch]

        if len(target_snapshots) == 0:
            raise FileNotFoundError(
                f"snapshot_epoch_{infer_epoch} does not exist in "
                f"{dir_path}"
            )

        return target_snapshots[0]


class DeployedModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        **kwards
    ) -> ISimlCheckpointFile:
        model_name = SimlConstItems.DEPLOYED_MODEL_NAME
        extensions = [
            SimlFileExtType.PTH, SimlFileExtType.PTHENC
        ]
        for ext in extensions:
            p = dir_path / f"{model_name}{ext.value}"
            if p.exists():
                return SimlFileBuilder.checkpoint_file(p)

        raise FileNotFoundError(
            f"Deployed model file does not exist in {dir_path}"
        )
