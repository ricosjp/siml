import pathlib
from unittest import mock

import pytest
from torch.utils.data import RandomSampler, SequentialSampler

from siml.services.training.data_loader_builder import DataLoaderBuilder
from siml.setting import MainSetting


@pytest.mark.parametrize("shuffle", [
    True, False
])
def test__consider_shuffle_option(shuffle):
    main_setting = MainSetting.read_settings_yaml(
        pathlib.Path("tests/data/linear/linear.yml")
    )
    main_setting.trainer.lazy = True
    main_setting.trainer.train_data_shuffle = shuffle
    collate_fn = mock.MagicMock()

    builder = DataLoaderBuilder(
        main_setting=main_setting,
        collate_fn=collate_fn
    )
    train_loader, _, _ = builder.create()

    if shuffle:
        assert isinstance(train_loader.sampler, RandomSampler)
    else:
        assert isinstance(train_loader.sampler, SequentialSampler)
