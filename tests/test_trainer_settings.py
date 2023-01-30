import pathlib

import siml.setting as setting
import siml.trainer as trainer


def test_match_encryption_keys_when_restart():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform/mlp.yml'))
    main_setting.trainer.restart_directory = \
        pathlib.Path("tests/data/deform/mlp")

    test_key = bytes(b'sample_test')
    main_setting.trainer.model_key = test_key
    main_setting.data.encrypt_key = test_key

    tr = trainer.Trainer(main_setting)

    assert tr.setting.trainer.model_key == test_key
    assert tr.setting.data.encrypt_key == test_key
