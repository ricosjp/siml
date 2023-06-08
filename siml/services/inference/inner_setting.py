import pathlib
from typing import Optional, Union, Any

import pydantic

from siml import setting
from siml.preprocessing import ScalersComposition
from siml.services.model_selector import ModelSelectorBuilder
from siml.services.path_rules import SimlPathRules


# HACK
# In the future, this setting is merged to setting.InfererSetting
class InnerInfererSetting(pydantic.BaseModel):
    # HACK: To Avoid recursive validation by pydantic, Any type is used.
    main_setting: Union[setting.MainSetting, Any]
    force_model_path: Optional[pathlib.Path] = None
    force_converter_parameters_pkl: Optional[pathlib.Path] = None
    decrypt_key: Optional[bytes] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def conversion_setting(self) -> setting.ConversionSetting:
        return self.main_setting.conversion

    @property
    def inferer_setting(self) -> setting.InfererSetting:
        return self.main_setting.inferer

    @property
    def trainer_setting(self) -> setting.TrainerSetting:
        return self.main_setting.trainer

    @property
    def model_setting(self) -> setting.ModelSetting:
        return self.main_setting.model

    @property
    def perform_inverse(self) -> bool:
        return self.main_setting.inferer.perform_inverse

    def load_scalers(self) -> Union[ScalersComposition, None]:
        if not self.perform_inverse:
            return None
        pkl_file = self.get_converter_parameters_pkl_path()
        scalers = ScalersComposition.create_from_file(
            pkl_file,
            key=self.get_crypt_key()
        )
        return scalers

    def get_crypt_key(self) -> Union[bytes, None]:
        if self.decrypt_key is not None:
            return self.decrypt_key

        return self.main_setting.get_crypt_key()

    def get_snapshot_file_path(self) -> pathlib.Path:
        if self.force_model_path is not None:
            if self.force_model_path.is_file():
                return self.force_model_path

            model_path = self.force_model_path
        else:
            model_path = self.main_setting.inferer.model

        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} does not exist."
            )

        if model_path.is_file():
            return model_path

        selector = ModelSelectorBuilder.create(
            self.main_setting.trainer.snapshot_choise_method
        )
        siml_file = selector.select_model(
            model_path,
            infer_epoch=self.inferer_setting.infer_epoch
        )
        return siml_file.file_path

    def get_write_simulation_case_dir(
        self,
        data_directory: Union[pathlib.Path, None]
    ) -> Union[pathlib.Path, None]:
        if data_directory is None:
            return None

        if self.main_setting.inferer.perform_preprocess:
            # Assume the given data is raw data
            return data_directory

        path_rules = SimlPathRules()
        write_simulation_base = self.main_setting.inferer.write_simulation_base
        path = path_rules.determine_write_simulation_case_dir(
            data_directory=data_directory,
            write_simulation_base=write_simulation_base
        )
        return path

    def get_converter_parameters_pkl_path(self) -> pathlib.Path:
        if self.force_converter_parameters_pkl is not None:
            return self.force_converter_parameters_pkl

        if self.main_setting.inferer.converter_parameters_pkl is None:
            return self.main_setting.data.preprocessed_root \
                / 'preprocessors.pkl'
        else:
            return self.main_setting.inferer.converter_parameters_pkl

    def get_output_directory(
        self,
        date_string: str,
        *,
        data_directory: Optional[pathlib.Path] = None,
    ) -> pathlib.Path:
        if self.inferer_setting.output_directory is not None:
            return self.inferer_setting.output_directory

        if data_directory is None:
            data_directory = pathlib.Path("")
        dir_name = self._determine_directory_name(date_string)
        base = self.inferer_setting.output_directory_base / dir_name
        rules = SimlPathRules()

        path = rules.determine_output_directory(
            data_directory, output_base_directory=base
        )
        return path

    def _determine_directory_name(self, date_string: str):
        model_name = self.get_model_name()
        return f"{model_name}_{date_string}"

    def get_model_name(self) -> str:
        snapshot_file = self.get_snapshot_file_path()
        model_name = snapshot_file.parent.name
        return model_name

    def skip_fem_data_creation(
        self,
        data_directory: Optional[pathlib.Path] = None
    ) -> bool:
        write_simulation_base = self.get_write_simulation_case_dir(
            data_directory
        )

        if self.inferer_setting.skip_fem_data_creation:
            return True

        if write_simulation_base is None:
            return True

        if not write_simulation_base.exists():
            print(
                f"{write_simulation_base} does not exist."
                "Thus, skip creating fem data."
            )
            return True

        return False
