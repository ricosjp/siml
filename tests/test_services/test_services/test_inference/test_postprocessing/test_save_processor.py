from unittest import mock
import pathlib
import shutil

import pytest
from ignite.engine import State

import siml
from siml.services.inference import InnerInfererSetting
from siml.services.inference.postprocessing import SaveProcessor
from siml.services.inference.postprocessing.save_processor \
    import WrapperResultItems


TEST_OUTPUT_DIR = pathlib.Path(__file__).parent / "tmp_data"


@pytest.fixture(scope="module")
def init_output_dir():
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)


@pytest.mark.parametrize("save_summary", [
    True, False
])
def test_save_processor_save_x_is_equal_to_save_summary(
        save_summary, init_output_dir):

    inner_setting = InnerInfererSetting(
        main_setting=siml.setting.MainSetting()
    )
    state = State(metrics={"post_results": []})
    with mock.patch.object(
        WrapperResultItems,
        "get_item",
        return_value="sample_str"
    ), mock.patch.object(
        InnerInfererSetting,
        "get_output_directory",
        return_value=pathlib.Path(__file__).parent / "tmp_data"
    ):
        save_processor = SaveProcessor(inner_setting)
        save_processor.save_each_results = mock.MagicMock()
        save_processor.run(state, save_summary=save_summary)
        save_processor._save_logs = mock.MagicMock()

        assert save_processor.save_each_results.call_args.kwargs["save_x"] \
            == save_summary
