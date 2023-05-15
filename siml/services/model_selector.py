import abc
import pathlib

import numpy as np
import pandas as pd

from siml.base.siml_const import SimlConstItems
from siml.base.siml_enums import ModelSelectionType, SimlFileExtType
from siml.path_like_objects import ISimlCheckpointFile, SimlFileBulider


class IModelSelector(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def select_model(
        dir_path: pathlib.Path,
        **args
    ) -> ISimlCheckpointFile:
        raise NotImplementedError()


class ModelSelectorBuilder:
    @staticmethod
    def create(
        selection_type: ModelSelectionType
    ) -> IModelSelector:
        class_dict = {
            ModelSelectionType.BEST: BestModelSelector,
            ModelSelectionType.LATEST: LatestModelSelector,
            ModelSelectionType.SPECIFIED: SpecifiedModelSelector,
            ModelSelectionType.TRAIN_BEST: TrainBestModelSelector,
            ModelSelectionType.DEPLOYED: DeployedModelSelector
        }
        return class_dict[selection_type]


class BestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        **args
    ) -> ISimlCheckpointFile:
        snapshots = [
            SimlFileBulider.checkpoint_file(p) for p in
            list(dir_path.glob('snapshot_epoch_*'))
        ]

        df = pd.read_csv(dir_path / 'log.csv',
                         header=0,
                         index_col=None,
                         skipinitialspace=True)
        if np.any(np.isnan(df['validation_loss'])):
            raise ValueError("NaN value is found in validation result.")

        best_epoch = df['epoch'].iloc[df['validation_loss'].idxmin()]
        target_snapshot: ISimlCheckpointFile = \
            [p for p in snapshots if p.epoch == best_epoch][0]
        return target_snapshot


class LatestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        **args
    ) -> ISimlCheckpointFile:
        snapshots = [
            SimlFileBulider.checkpoint_file(p) for p in
            list(dir_path.glob('snapshot_epoch_*'))
        ]
        max_epoch = max([p.epoch for p in snapshots])
        target_snapshot: ISimlCheckpointFile = \
            [p for p in snapshots if p.epoch == max_epoch][0]
        return target_snapshot


class TrainBestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        **args
    ) -> ISimlCheckpointFile:
        snapshots = [
            SimlFileBulider.checkpoint_file(p) for p in
            list(dir_path.glob('snapshot_epoch_*'))
        ]

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
        target_epoch: int,
        **args
    ) -> ISimlCheckpointFile:
        snapshots = [
            SimlFileBulider.checkpoint_file(p) for p in
            list(dir_path.glob('snapshot_epoch_*'))
        ]
        target_snapshot: ISimlCheckpointFile = \
            [p for p in snapshots if p.epoch == target_epoch][0]
        return target_snapshot


class DeployedModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        dir_path: pathlib.Path,
        **kwards
    ) -> pathlib.Path:
        model_name = SimlConstItems.DEPLOYED_MODEL_NAME
        extensions = [
            SimlFileExtType.PTH, SimlFileExtType.PTHENC
        ]
        for ext in extensions:
            p = dir_path / f"{model_name}{ext.value}"
            if p.exists():
                return SimlFileBulider.checkpoint_file(p)

        raise FileNotFoundError(
            f"Deployed model file does not exist in {dir_path}"
        )
