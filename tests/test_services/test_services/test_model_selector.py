import pathlib
import shutil
import random
import os

import pandas as pd
import pytest

from siml.services import ModelSelectorBuilder
from siml.base.siml_const import SimlConstItems
from siml.services.model_selector import (
    BestModelSelector, TrainBestModelSelector, SpecifiedModelSelector,
    LatestModelSelector, DeployedModelSelector
)

TEST_DATA_DIR = pathlib.Path(__file__).parent / "tmp_data"


@pytest.mark.parametrize("select_type, expect_cls", [
    ('best', BestModelSelector),
    ('train_best', TrainBestModelSelector),
    ('specified', SpecifiedModelSelector),
    ('latest', LatestModelSelector),
    ('deployed', DeployedModelSelector)
])
def test__selctor_builder(select_type, expect_cls):
    actual_cls = ModelSelectorBuilder.create(select_type)

    assert type(actual_cls) == type(expect_cls)


@pytest.fixture(scope="module")
def prepare_snapshots():
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)

    TEST_DATA_DIR.mkdir()

    for i in range(10):
        file_path = TEST_DATA_DIR / f"snapshot_epoch_{i}.pth"
        file_path.touch()

    epochs = [i + 1 for i in range(i)]
    train_loss = [random.random() for i in range(i)]
    validation_loss = [random.random() for i in range(i)]

    df = pd.DataFrame(data={
        "epoch": epochs,
        "train_loss": train_loss,
        "validation_loss": validation_loss
    }, index=None)
    df.to_csv(TEST_DATA_DIR / "log.csv")


def test__best_select_model(prepare_snapshots):
    actual_path = BestModelSelector.select_model(TEST_DATA_DIR)

    df = pd.read_csv(TEST_DATA_DIR / "log.csv", index_col=None, header=0)
    idx = df.loc[:, "validation_loss"].idxmin()
    epoch = df.loc[idx, "epoch"]

    assert actual_path.epoch == epoch


def test__latest_select_model(prepare_snapshots):
    actual_path = LatestModelSelector.select_model(TEST_DATA_DIR)
    df = pd.read_csv(TEST_DATA_DIR / "log.csv", index_col=None)
    max_epoch = df.loc[:, "epoch"].max()

    assert actual_path.epoch == max_epoch


def test__train_best_select_model(prepare_snapshots):
    actual_path = TrainBestModelSelector.select_model(TEST_DATA_DIR)

    df = pd.read_csv(TEST_DATA_DIR / "log.csv", index_col=None)
    idx = df.loc[:, "train_loss"].idxmin()
    epoch = df.loc[idx, "epoch"]

    assert actual_path.epoch == epoch


@pytest.mark.parametrize("epoch", [
    1, 5, 6, 8
])
def test__spcified_model_selector(epoch, prepare_snapshots):
    actual_path = SpecifiedModelSelector.select_model(
        TEST_DATA_DIR,
        infer_epoch=epoch
    )

    assert actual_path.epoch == epoch


@pytest.mark.parametrize("epoch", [
    100, 200
])
def test__spcified_model_selector_not_existed(epoch, prepare_snapshots):
    with pytest.raises(FileNotFoundError):
        _ = SpecifiedModelSelector.select_model(
            TEST_DATA_DIR,
            infer_epoch=epoch
        )


@pytest.mark.parametrize("file_name", [
    (f"{SimlConstItems.DEPLOYED_MODEL_NAME}.pth"),
    (f"{SimlConstItems.DEPLOYED_MODEL_NAME}.pth.enc")
])
def test__deployed_model_selector(file_name: str, prepare_snapshots):
    path = (TEST_DATA_DIR / file_name)
    path.touch(exist_ok=True)

    actual_path = DeployedModelSelector.select_model(
        TEST_DATA_DIR
    )

    assert actual_path.file_path == path
    os.remove(path)
