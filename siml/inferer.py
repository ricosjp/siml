import pathlib
from typing import Callable, Optional

import numpy as np
from torch import Tensor

from siml import datasets, setting
from siml.base.siml_const import SimlConstItems
from siml.loss_operations import LossCalculatorBuilder
from siml.path_like_objects import SimlDirectory, SimlFileBuilder
from siml.preprocessing import ScalersComposition
from siml.preprocessing.converter import ILoadFunction
from siml.services import ModelEnvironmentSetting, ModelSelectorBuilder
from siml.services.inference import (
    CoreInferer, InferenceDataLoaderBuilder, InnerInfererSetting,
    PostPredictionRecord
)
from siml.services.inference.postprocessing import (
    ISaveFunction, SaveProcessor, PostProcessor,
    PostFEMDataConverter, IFEMDataAdditionFunction
)


class Inferer():

    @classmethod
    def read_settings(cls, settings_yaml: pathlib.Path, **kwargs):
        """Read settings.yaml to generate Inferer object.

        Parameters
        ----------
        settings_yaml: str or pathlib.Path
            setting.yaml file name.

        Returns
        --------
        siml.Inferer
        """
        main_setting = setting.MainSetting.read_settings_yaml(settings_yaml)
        return cls(main_setting, **kwargs)

    @classmethod
    def from_model_directory(
        cls,
        model_directory: pathlib.Path,
        decrypt_key: bytes = None,
        infer_epoch: int = None,
        **kwargs
    ):
        """Load model data from a deployed directory.

        Parameters
        ----------
        model_directory: str or pathlib.Path
            Model directory created with Inferer.deploy().
        decrypt_key: bytes, optional
            Key to decrypt model data. If not fed, and the data is encrypted,
            ValueError is raised.
        infer_epoch: int, optional
            If fed, model which corresponds to infer_epoch is used.

        Returns
        --------
        siml.Inferer
        """
        siml_directory = SimlDirectory(model_directory)
        pickle_file = siml_directory.find_pickle_file(
            "preprocessors", allow_missing=True
        )
        if pickle_file is None:
            print(f"Not found pickle file in a directory: {model_directory}")

        setting_file = siml_directory.find_yaml_file('settings')
        method = 'best' if infer_epoch is None else 'specified'
        selector = ModelSelectorBuilder.create(method)
        model_path = selector.select_model(
            model_directory,
            infer_epoch=infer_epoch
        )

        main_setting = setting.MainSetting.read_settings_yaml(
            setting_file.file_path,
            decrypt_key=decrypt_key,
            model_path=model_path,
            converter_parameters_pkl=pickle_file.file_path
        )

        obj = cls(main_setting, **kwargs)
        return obj

    def __init__(
        self,
        settings: setting.MainSetting,
        *,
        model_path: Optional[pathlib.Path] = None,
        converter_parameters_pkl: Optional[pathlib.Path] = None,
        load_function: ILoadFunction = None,
        data_addition_function: IFEMDataAdditionFunction = None,
        postprocess_function=None,
        save_function: ISaveFunction = None,
        user_loss_function_dic:
        dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
        infer_epoch: Optional[int] = None
    ) -> None:
        """Initialize Inferer object.

        Parameters
        ----------
        settings: siml.MainSetting
        model: pathlib.Path, optional
            If fed, overwrite self.setting.inferer.model.
        converter_parameters_pkl: pathlib.Path, optional
            If fed, overwrite self.setting.inferer.converter_parameters_pkl
        save: bool, optional
            If fed, overwrite self.setting.inferer.save
        conversion_function: function, optional
            Conversion function to preprocess raw data. It should receive
            two parameters, fem_data and raw_directory. If not fed,
            no additional conversion occurs.
        load_function: function, optional
            Function to load data, which take list of pathlib.Path objects
            (as required files) and pathlib.Path object (as data directory)
            and returns data_dictionary and fem_data (can be None) to be saved.
        data_addition_function: function, optional
            Function to add some data at simulation data writing phase.
            If not fed, no data addition occurs.
        postprocess_function: function, optional
            Function to make postprocess of the inference data.
            If not fed, no additional postprocess will be performed.
        save_function: function, optional
            Function to save results. If not fed the default save function
            will be used.
        """
        self._inner_setting = InnerInfererSetting(
            main_setting=settings,
            force_model_path=model_path,
            force_converter_parameters_pkl=converter_parameters_pkl,
            infer_epoch=infer_epoch
        )

        self.load_function = load_function
        self.data_addition_function = data_addition_function
        self.postprocess_function = postprocess_function
        self.save_function = save_function

        fem_data_creator = PostFEMDataConverter(
            inferer_setting=self._inner_setting.inferer_setting,
            conversion_setting=self._inner_setting.conversion_setting,
            load_function=load_function,
            data_addition_function=data_addition_function
        )

        self._model_env = self._create_model_env_setting()
        self._collate_fn = self._create_collate_fn()
        self._dataloader_builder = self._create_data_loader_builder()
        self._core_inferer = self._create_core_inferer(
            fem_data_creator,
            user_loss_function_dic
        )
        self._save_processor = SaveProcessor(
            inner_setting=self._inner_setting,
            user_save_function=save_function
        )

    def _create_model_env_setting(self) -> ModelEnvironmentSetting:
        trainer_setting = self._inner_setting.trainer_setting
        _model_env = ModelEnvironmentSetting(
            gpu_id=trainer_setting.gpu_id,
            seed=trainer_setting.seed,
            data_parallel=trainer_setting.data_parallel,
            model_parallel=trainer_setting.model_parallel,
            time_series=trainer_setting.time_series
        )
        return _model_env

    def _create_collate_fn(self):
        trainer_setting = self._inner_setting.trainer_setting
        input_is_dict = trainer_setting.inputs.is_dict
        output_is_dict = trainer_setting.outputs.is_dict

        input_time_series_keys = \
            trainer_setting.inputs.get_time_series_keys()
        output_time_series_keys = \
            trainer_setting.outputs.get_time_series_keys()

        input_time_slices = trainer_setting.inputs.time_slice
        output_time_slices = trainer_setting.outputs.time_slice
        element_wise = trainer_setting.determine_element_wise()

        collate_fn = datasets.CollateFunctionGenerator(
            time_series=trainer_setting.time_series,
            dict_input=input_is_dict,
            dict_output=output_is_dict,
            use_support=trainer_setting.support_inputs,
            element_wise=element_wise,
            data_parallel=trainer_setting.data_parallel,
            input_time_series_keys=input_time_series_keys,
            output_time_series_keys=output_time_series_keys,
            input_time_slices=input_time_slices,
            output_time_slices=output_time_slices
        )
        return collate_fn

    def _create_post_processor(
        self,
        fem_data_creator: PostFEMDataConverter
    ) -> PostProcessor:
        pkl_path = self._inner_setting.get_converter_parameters_pkl_path()
        scalers = ScalersComposition.create_from_file(
            pkl_path,
            key=self._inner_setting.main_setting.get_encrypt_key()
        )
        post_processor = PostProcessor(
            inner_setting=self._inner_setting,
            fem_data_converter=fem_data_creator,
            scalers=scalers
        )
        return post_processor

    def _create_core_inferer(
        self,
        fem_data_creator: PostFEMDataConverter,
        user_loss_function_dic: dict = None
    ) -> CoreInferer:
        post_processor = self._create_post_processor(fem_data_creator)
        loss_function = LossCalculatorBuilder.create(
            trainer_setting=self._inner_setting.trainer_setting,
            allow_no_answer=True,
            user_loss_function_dic=user_loss_function_dic
        )
        _core_inferer = CoreInferer(
            trainer_setting=self._inner_setting.trainer_setting,
            model_setting=self._inner_setting.model_setting,
            env_setting=self._model_env,
            snapshot_file=self._inner_setting.get_snapshot_file_path(),
            prepare_batch_function=self._collate_fn.prepare_batch,
            loss_function=loss_function,
            post_processor=post_processor
        )
        return _core_inferer

    def _create_data_loader_builder(self) -> InferenceDataLoaderBuilder:
        dataloader_builder = InferenceDataLoaderBuilder(
            trainer_setting=self._inner_setting.trainer_setting,
            collate_fn=self._collate_fn,
            decrypt_key=self._inner_setting.main_setting.get_encrypt_key()
        )
        return dataloader_builder

    def infer(
        self,
        *,
        data_directories: list[pathlib.Path] = None,
        output_directory_base: Optional[pathlib.Path] = None,
        output_all: bool = False
    ):
        """Perform infererence.

        Parameters
        ----------
        data_directories: list[pathlib.Path], optional
            List of data directories. Data is searched recursively.
            The default is an empty list.
        model: pathlib.Path, optional
            If fed, overwrite self.setting.inferer.model.
        perform_preprocess: bool, optional
            If fed, overwrite self.setting.inferer.perform_preprocess
        output_directory_base: pathlib.Path, optional
            If fed, overwrite self.setting.inferer.output_directory_base
        output_all: bool, optional. Dafault False
            If True, return all of results \
                including not preprocessed predicted data


        Returns
        -------
        inference_results: list[Dict]
            Inference results contains:
                - dict_x: input and variables
                - dict_y: inferred variables
                - dict_answer: answer variables (None if not found)
                - loss: Loss value (scaled)
                - raw_loss: Loss in a raw scale
                - fem_data: FEMData object
                - output_directory: Output directory path
                - data_directory: Input directory path
                - inference_time: Inference time
        """
        if data_directories is not None:
            if isinstance(data_directories, pathlib.Path):
                data_directories = [data_directories]
        else:
            data_directories = \
                self._inner_setting.inferer_setting.data_directories
        if output_directory_base is not None:
            # HACK: Improve it to make setting class immutable
            self._inner_setting.main_setting.inferer.output_directory_base \
                = output_directory_base

        inference_loader = self._dataloader_builder.create(
            data_directories=data_directories
        )
        inference_state = self._core_inferer.run(inference_loader)

        if self._inner_setting.inferer_setting.save:
            self._save_processor.run(inference_state)

        if output_all:
            return inference_state
        else:
            return self._format_results(inference_state.metrics)

    def infer_dict_data(
        self,
        raw_dict_x: dict,
        *,
        answer_raw_dict_y: Optional[dict] = None
    ):
        """
        Infer with simplified model.

        Parameters
        ----------
        model_path: pathlib.Path
            Model file or directory name.
        raw_dict_x: dict
            Dict of raw x data.
        answer_raw_dict_y: dict, optional
            Dict of answer raw y data.

        Returns
        -------
        inference_result: Dict
            Inference results contains:
                - dict_x: input and answer variables
                - dict_y: inferred variables
                - loss: Loss value (scaled)
                - raw_loss: Loss in a raw scale
                - fem_data: FEMData object
                - output_directory: Output directory path
                - data_directory: Input directory path
                - inference_time: Inference time
        """
        inference_loader = self._dataloader_builder.create(
            raw_dict_x=raw_dict_x,
            answer_raw_dict_y=answer_raw_dict_y
        )
        inference_state = self._core_inferer.run(inference_loader)

        if self._inner_setting.inferer_setting.save:
            self._save_processor.run(inference_state)

        return inference_state.metrics

    def infer_parameter_study(
            self, model, data_directories, *, n_interpolation=100,
            converter_parameters_pkl=None):
        """
        Infer with performing parameter study. Parameter study is done with the
        data generated by interpolating the input data_directories.

        Parameters
        ----------
        model: pathlib.Path or io.BufferedIOBase, optional
            Model directory, file path, or buffer. If not fed,
            TrainerSetting.pretrain_directory will be used.
        data_directories: list[pathlib.Path]
            List of data directories.
        n_interpolation: int, optional
            The number of points used for interpolation.
        Returns
        -------
        interpolated_input_dict: dict
            Input data dict generated by interpolation.
        output_dict: dict
            Output data dict generated by inference.
        """
        if self.setting.trainer.time_series:
            batch_axis = 1
        else:
            batch_axis = 0
        if self.setting.trainer.simplified_model:
            keepdims = False
        else:
            keepdims = True

        input_dict = {
            x_variable_name:
            np.stack([
                np.load(d / f"{x_variable_name}.npy")
                for d in data_directories], axis=batch_axis)
            for x_variable_name in self.setting.trainer.input_names}
        min_input_dict = {
            x_variable_name: np.min(data, axis=batch_axis, keepdims=keepdims)
            for x_variable_name, data in input_dict.items()}
        max_input_dict = {
            x_variable_name: np.max(data, axis=batch_axis, keepdims=keepdims)
            for x_variable_name, data in input_dict.items()}

        interpolated_input_dict = {
            x_variable_name:
            np.concatenate([
                (
                    i * min_input_dict[x_variable_name]
                    + (n_interpolation - i) * max_input_dict[x_variable_name])
                / n_interpolation
                for i in range(n_interpolation + 1)], axis=1)
            for x_variable_name in min_input_dict.keys()}
        output_dict = self.infer_simplified_model(
            model, interpolated_input_dict,
            converter_parameters_pkl=converter_parameters_pkl)[0]
        return interpolated_input_dict, output_dict

    def deploy(
        self,
        output_directory: pathlib.Path,
        encrypt_key: bytes = None
    ):
        """Deploy model information.

        Parameters
        ----------
        output_directory: pathlib.Path
            Output directory path.
        encrypt_key: bytes, optional
            Key to encrypt model data. If not fed, the model data will not be
            encrypted.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        enc_ext = ".enc" if encrypt_key is not None else ""
        key = self._inner_setting.main_setting.get_encrypt_key()

        # TODO: should deep copy to avoid side-effect
        main_setting = self._inner_setting.main_setting

        # output name
        output_model_name = \
            SimlConstItems.DEPLOYED_MODEL_NAME + f".pth{enc_ext}"
        output_model_path = SimlFileBuilder.checkpoint_file(
            output_directory / output_model_name
        )
        output_pkl = SimlFileBuilder.pickle_file(
            output_directory / f'preprocessors.pkl{enc_ext}'
        )
        output_setting_yaml = SimlFileBuilder.yaml_file(
            output_directory / 'settings.yml'
        )
        # Overwrite Setting
        main_setting.inferer.model = output_model_path

        # Model
        snapshot_file = self._inner_setting.get_snapshot_file_path()
        siml_path = SimlFileBuilder.checkpoint_file(snapshot_file)
        model_data = siml_path.load(device="cpu", decrypt_key=key)
        output_model_path.save(model_data, encrypt_key=key)

        # pickle
        pkl_path = self._inner_setting.get_converter_parameters_pkl_path()
        pkl_path = SimlFileBuilder.pickle_file(pkl_path)
        pkl_data = pkl_path.load(decrypt_key=key)
        output_pkl.save(
            pkl_data, encrypt_key=key
        )

        # yaml
        output_setting_yaml.save(
            self._inner_setting.main_setting, encrypt_key=key
        )
        return

    def _format_results(self, metrics: dict) -> list[dict]:
        results = []
        records: list[PostPredictionRecord] = metrics.pop("post_results")
        for i, val in enumerate(records):
            _data = {
                name: getattr(val, name)
                for name in PostPredictionRecord._fields
            }
            for name, item in metrics.items():
                _data.update({name: item[i]})

            results.append(_data)
        return results
