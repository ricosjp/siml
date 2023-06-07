from unittest import mock

import pytest

import siml
from siml.services.inference import InnerInfererSetting
from siml.services.inference.postprocessing import SaveProcessor


@pytest.mark.parametrize("save_summary", [
    True, False
])
def test_save_processor_save_x_is_equal_to_save_summary(save_summary):

    inner_setting = InnerInfererSetting(
        main_setting=siml.setting.MainSetting()
    )

    records = []
    save_processor = SaveProcessor(inner_setting)
    save_processor._save_each_results = mock.MagicMock()
    save_processor._save_summary = mock.MagicMock()
    save_processor.run(records, save_summary=save_summary)

    assert save_processor._save_each_results.call_args.kwargs["save_x"] \
        == save_summary
