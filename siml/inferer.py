from __future__ import annotations
import pathlib
from typing import Callable, Optional, Union

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
import femio
from ignite.engine import State

from siml import datasets, setting
from siml.base.siml_const import SimlConstItems
from siml.loss_operations import LossCalculatorBuilder
from siml.path_like_objects import SimlDirectory, SimlFileBuilder
from siml.preprocessing import ScalersComposition
from siml.preprocessing.converter import (
    ILoadFunction, IConvertFunction, RawConverter)
from siml.services import ModelEnvironmentSetting, ModelSelectorBuilder
from siml.services.inference import (
    CoreInferer, InferenceDataLoaderBuilder, InnerInfererSetting,
    PredictionRecord, PostPredictionRecord
)
from siml.services.inference.postprocessing import (
    IInfererSaveFunction, SaveProcessor, PostProcessor,
    PostFEMDataConverter, IFEMDataAdditionFunction
)


class WholeInferProcessor:
    def __init__(
        self,
        main_setting: setting.MainSetting,
        model_path: Optional[pathlib.Path] = None,
        converter_parameters_pkl: Optional[pathlib.Path] = None,
        conversion_function: Optional[IConvertFunction] = None,
        load_function: Optional[ILoadFunction] = None,
        data_addition_function: Optional[IFEMDataAdditionFunction] = None,
        save_function: Optional[IInfererSaveFunction] = None,
        user_loss_function_dic:
        dict[str, Callable[[Tensor, Tensor], Tensor]] = None
    ) -> dict:

        self.raw_converter = RawConverter(
            main_setting=main_setting,
            conversion_function=conversion_function,
            load_function=load_function
        )

        _inner_setting = InnerInfererSetting(
            main_setting=main_setting,
            force_model_path=model_path,
            force_converter_parameters_pkl=converter_parameters_pkl
        )
        pkl_path = _inner_setting.get_converter_parameters_pkl_path()
        self.scalers = ScalersComposition.create_from_file(
            converter_parameters_pkl=pkl_path,
            key=main_setting.get_crypt_key()
        )

        self.inferer = Inferer(
            main_setting=main_setting,
            model_path=model_path,
            scalers=self.scalers,
            load_function=load_function,
            data_addition_function=data_addition_function,
            save_function=save_function,
            user_loss_function_dic=user_loss_function_dic
        )

    def run(
        self,
        data_directories: Union[list[pathlib.Path], pathlib.Path],
        output_directory_base: Optional[pathlib.Path] = None,
        perform_preprocess: bool = True,
        save_summary: Optional[bool] = True
    ) -> dict:
        """run whole inference processes.

        Parameters
        ----------
        data_directories : Union[list[pathlib.Path], pathlib.Path]
            pathes to data
        output_directory_base : Optional[pathlib.Path], optional
            path to parent directory of cases, by default None
        perform_preprocess : bool, optional
            If True, perform preprocessing and scaling, by default True
        save : Optional[bool], optional
            If True, save items, by default None

        Returns
        -------
        dict
            dictionary of results
        """
        if perform_preprocess:
            return self._run_with_preprocess(
                data_directories=data_directories,
                output_directory_base=output_directory_base,
                save_summary=save_summary
            )
        else:
            return self.inferer.infer(
                data_directories=data_directories,
                output_directory_base=output_directory_base,
                save_summary=save_summary
            )

    def _run_with_preprocess(
        self,
        data_directories: Union[list[pathlib.Path], pathlib.Path],
        output_directory_base: Optional[pathlib.Path] = None,
        save_summary: Optional[bool] = True
    ) -> dict:
        if isinstance(data_directories, pathlib.Path):
            data_directories = [data_directories]

        inner_setting = self.inferer._inner_setting
        conversion_setting = inner_setting.conversion_setting
        dataset = datasets.PreprocessDataset(
            inner_setting.trainer_setting.input_names,
            inner_setting.trainer_setting.output_names,
            data_directories,
            supports=inner_setting.trainer_setting.support_inputs,
            num_workers=0,
            allow_no_data=True,
            decrypt_key=inner_setting.get_crypt_key(),
            raw_converter=self.raw_converter,
            scalers=self.scalers,
            required_file_names=conversion_setting.required_file_names,
            conversion_setting=conversion_setting
        )
        return self.inferer.infer_dataset(
            dataset,
            output_directory_base=output_directory_base,
            save_summary=save_summary
        )

    def run_dict_data(
        self,
        raw_dict_x: dict,
        *,
        answer_raw_dict_y: Optional[dict] = None,
        perform_preprocess: bool = True
    ) -> dict:
        """_summary_

        Parameters
        ----------
        raw_dict_x : dict
            Dict of raw x data.
        answer_raw_dict_y : Optional[dict], optional
            Dict of raw answer y data, by default None
        perform_preprocess : bool, optional
            If True, perform scaling. by default True

        Returns
        -------
        dict
            dictionary of result
        """

        if perform_preprocess:
            scaled_dict_x = self.scalers.transform_dict(raw_dict_x)
            if answer_raw_dict_y is not None:
                scaled_dict_answer = self.scalers.transform_dict(
                    answer_raw_dict_y
                )
            else:
                scaled_dict_answer = None

            results = self.inferer.infer_dict_data(
                scaled_dict_x, scaled_dict_answer=scaled_dict_answer
            )
            return results
        else:
            return self.inferer.infer_dict_data(
                raw_dict_x, scaled_dict_answer=answer_raw_dict_y
            )


class Inferer():

    @classmethod
    def read_settings_file(
        cls,
        settings_yaml: pathlib.Path,
        model_path: Optional[pathlib.Path] = None,
        converter_parameters_pkl: Optional[pathlib.Path] = None,
        **kwargs
    ) -> Inferer:
        """Read settings.yaml to generate Inferer object.

        Parameters
        ----------
        settings_yaml : pathlib.Path
            Path to yaml file of setting
        model_path : Optional[pathlib.Path], optional
            If fed, overwrite path to model file, by default None
        converter_parameters_pkl : Optional[pathlib.Path], optional
            If fed, overwrite path to pkl file of scaling parameters,
             by default None

        Returns
        -------
        Inferer
            Inferer object
        """
        main_setting = setting.MainSetting.read_settings_yaml(settings_yaml)
        return cls(
            main_setting=main_setting,
            model_path=model_path,
            converter_parameters_pkl=converter_parameters_pkl,
            **kwargs
        )

    @classmethod
    def from_model_directory(
        cls,
        model_directory: pathlib.Path,
        converter_parameters_pkl: Optional[pathlib.Path] = None,
        model_select_method: str = "best",
        decrypt_key: bytes = None,
        infer_epoch: int = None,
        **kwargs
    ):
        """Load model data from a deployed directory.

        Parameters
        ----------
        model_directory: str or pathlib.Path
            Model directory created with Inferer.deploy().
        model_path : Optional[pathlib.Path], optional
            If fed, overwrite path to model file, by default None
        converter_parameters_pkl : Optional[pathlib.Path], optional
            If fed, overwrite path to pkl file of scaling parameters,
             by default None
        decrypt_key: bytes, optional
            Key to decrypt model data. If not fed, and the data is encrypted,
            ValueError is raised.
        model_select_method: str, optional
            method name to select model. By default, best
        infer_epoch: int, optional
            If fed, model which corresponds to infer_epoch is used.

        Returns
        --------
        siml.Inferer
            Inferer object
        """

        siml_directory = SimlDirectory(model_directory)
        if converter_parameters_pkl is None:
            pickle_file = siml_directory.find_pickle_file(
                "preprocessors", allow_missing=True
            )

            if pickle_file is None:
                raise ValueError(
                    f"Not found pickle file in a directory: {model_directory}."
                    "Set converter_parameters_pkl argument explicitly."
                )
            pickle_file_path = pickle_file.file_path
        else:
            pickle_file_path = converter_parameters_pkl

        setting_file = siml_directory.find_yaml_file('settings')
        selector = ModelSelectorBuilder.create(model_select_method)
        model_path = selector.select_model(
            model_directory,
            infer_epoch=infer_epoch
        )

        main_setting = setting.MainSetting.read_settings_yaml(
            setting_file.file_path,
            decrypt_key=decrypt_key,
        )

        obj = cls(
            main_setting=main_setting,
            model_path=model_path.file_path,
            converter_parameters_pkl=pickle_file_path,
            decrypt_key=decrypt_key,
            **kwargs
        )
        return obj

    def __init__(
        self,
        main_setting: setting.MainSetting,
        *,
        scalers: ScalersComposition = None,
        model_path: Optional[pathlib.Path] = None,
        converter_parameters_pkl: Optional[pathlib.Path] = None,
        load_function: ILoadFunction = None,
        data_addition_function: IFEMDataAdditionFunction = None,
        save_function: IInfererSaveFunction = None,
        user_loss_function_dic:
        dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
        decrypt_key: Optional[bytes] = None
    ) -> None:
        """Initialize Inferer object.

        Parameters
        ----------
        main_setting: siml.MainSetting
        model_path : Optional[pathlib.Path], optional
            If fed, overwrite path to model file, by default None
        converter_parameters_pkl : Optional[pathlib.Path], optional
            If fed, overwrite path to pkl file of scaling parameters,
             by default None
        save: bool, optional
            If fed, overwrite self.setting.inferer.save
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
            main_setting=main_setting,
            force_model_path=model_path,
            force_converter_parameters_pkl=converter_parameters_pkl,
            decrypt_key=decrypt_key
        )

        self.load_function = load_function
        self.data_addition_function = data_addition_function
        self.save_function = save_function
        if scalers is None:
            self._scalers = self._inner_setting.load_scalers()
        else:
            self._scalers = scalers

        self._fem_data_converter = PostFEMDataConverter(
            inferer_setting=self._inner_setting.inferer_setting,
            conversion_setting=self._inner_setting.conversion_setting,
            load_function=load_function,
            data_addition_function=data_addition_function
        )

        self._model_env = self._create_model_env_setting()
        self._collate_fn = self._create_collate_fn()
        self._dataloader_builder = self._create_data_loader_builder()
        self._core_inferer = self._create_core_inferer(
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

    def _create_core_inferer(
        self,
        user_loss_function_dic: dict = None
    ) -> CoreInferer:
        post_processor = PostProcessor(
            inner_setting=self._inner_setting,
            scalers=self._scalers
        )
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
            post_processor=post_processor,
            decrypt_key=self._inner_setting.get_crypt_key()
        )
        return _core_inferer

    def _create_data_loader_builder(self) -> InferenceDataLoaderBuilder:
        dataloader_builder = InferenceDataLoaderBuilder(
            trainer_setting=self._inner_setting.trainer_setting,
            collate_fn=self._collate_fn,
            decrypt_key=self._inner_setting.get_crypt_key()
        )
        return dataloader_builder

    def infer(
        self,
        *,
        data_directories: list[pathlib.Path] = None,
        output_directory_base: Optional[pathlib.Path] = None,
        output_all: bool = False,
        save_summary: Optional[bool] = True
    ):
        """Perform infererence.

        Parameters
        ----------
        data_directories: list[pathlib.Path], optional
            List of data directories. Data is searched recursively.
            The default is an empty list.
        output_directory_base: pathlib.Path, optional
            If fed, overwrite self.setting.inferer.output_directory_base
        output_all: bool, optional. Dafault False
            If True, return all of results \
                including not preprocessed predicted data
        save: bool, optional. Default None
            If fed, overwrite save option in main setting
        save_summary: bool, optional. Default True
            If True, save summary information

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

        records = self._create_post_records(inference_state)
        if self._inner_setting.inferer_setting.save:
            self._save_processor.run(
                records, save_summary=save_summary
            )

        if output_all:
            return inference_state
        else:
            return self._format_results(records)

    def infer_dataset(
        self,
        preprocess_dataset: datasets.PreprocessDataset,
        output_directory_base: Optional[pathlib.Path] = None,
        save_summary: Optional[bool] = True
    ) -> list[dict]:
        """Perform inference for datasets

        Parameters
        ----------
        preprocess_dataset : datasets.PreprocessDataset
            dataset of preprocessed data
        output_directory_base : Optional[pathlib.Path], optional
            base output directory, by default None
        save_summary : Optional[bool], optional
            If fed, overwrite save option. by default None

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

        if output_directory_base is not None:
            # HACK: Improve it to make setting class immutable
            self._inner_setting.main_setting.inferer.output_directory_base \
                = output_directory_base

        inference_loader = DataLoader(
            preprocess_dataset,
            collate_fn=self._collate_fn,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        inference_state = self._core_inferer.run(inference_loader)

        records = self._create_post_records(inference_state)
        if self._inner_setting.inferer_setting.save:
            self._save_processor.run(
                records, save_summary=save_summary
            )

        return self._format_results(records)

    def infer_dict_data(
        self,
        scaled_dict_x: dict,
        *,
        data_directory: pathlib.Path = None,
        scaled_dict_answer: Optional[dict] = None,
        save_summary: Optional[bool] = True,
        base_fem_data: Optional[femio.FEMData] = None
    ):
        """
        Infer with dictionary data.

        Parameters
        ----------
        scaled_dict_x: dict
            Dict of scaled x data.
        data_directory: pathlib.Path, optional
            path to directory of simulation files
        scaled_dict_answer: dict, optional
            Dict of answer scaled y data.
        save_summary: bool, default True
            If True, save summary information of inference
        base_fem_data: femio.FEMData, optional
            If fed, inference results are registered to base_fem_data and
             saved as a file.


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
        if data_directory is not None:
            data_directories = [data_directory]
        else:
            data_directories = None

        inference_loader = self._dataloader_builder.create(
            raw_dict_x=scaled_dict_x,
            answer_raw_dict_y=scaled_dict_answer,
            data_directories=data_directories
        )
        inference_state = self._core_inferer.run(inference_loader)

        records = self._create_post_records(
            inference_state,
            base_fem_data=base_fem_data
        )
        if self._inner_setting.inferer_setting.save:
            self._save_processor.run(
                records, save_summary=save_summary
            )

        return self._format_results(records)

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
        main_setting.inferer.model = output_model_path.file_path

        # Model
        snapshot_file = self._inner_setting.get_snapshot_file_path()
        siml_path = SimlFileBuilder.checkpoint_file(snapshot_file)
        model_data = siml_path.load(device="cpu", decrypt_key=encrypt_key)
        output_model_path.save(model_data, encrypt_key=encrypt_key)

        # pickle
        pkl_path = self._inner_setting.get_converter_parameters_pkl_path()
        pkl_path = SimlFileBuilder.pickle_file(pkl_path)
        pkl_data = pkl_path.load(decrypt_key=encrypt_key)
        output_pkl.save(
            pkl_data, encrypt_key=encrypt_key
        )

        # yaml
        setting.write_yaml(
            main_setting,
            output_setting_yaml.file_path,
            key=encrypt_key
        )
        return

    def _format_results(
        self,
        records: list[PostPredictionRecord]
    ) -> list[dict]:
        results = [
            {
                name: getattr(record, name)
                for name in PostPredictionRecord._fields
            }
            for record in records
        ]
        return results

    def _create_post_records(
        self,
        state: State,
        base_fem_data: femio.FEMData = None
    ) -> list[PostPredictionRecord]:
        records: list[PredictionRecord] = state.metrics["post_results"]

        new_records: list[PostPredictionRecord] = []
        for i, _record in enumerate(records):
            fem_data = self._create_fem_data(
                _record, base_fem_data=base_fem_data
            )
            output_directory = \
                self._inner_setting.get_output_directory(
                    _record.inference_start_datetime,
                    data_directory=_record.data_directory,
                )
            _new_record = PostPredictionRecord(
                *_record,
                loss=state.metrics["loss"][i],
                raw_loss=state.metrics["raw_loss"][i],
                output_directory=output_directory,
                fem_data=fem_data
            )
            new_records.append(_new_record)
        return new_records

    def _create_fem_data(
        self,
        record: PredictionRecord,
        base_fem_data: femio.FEMData = None
    ) -> Union[femio.FEMData, None]:

        if self._inner_setting.skip_fem_data_creation(
            record.data_directory
        ):
            return None

        write_simulation_case_dir = \
            self._inner_setting.get_write_simulation_case_dir(
                record.data_directory
            )

        fem_data = self._fem_data_converter.create(
            dict_data_x=record.dict_x,
            dict_data_y=record.dict_y,
            dict_data_answer=record.dict_answer,
            write_simulation_case_dir=write_simulation_case_dir,
            base_fem_data=base_fem_data
        )
        return fem_data
