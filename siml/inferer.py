import pathlib
import time

import ignite
import numpy as np
import torch

from . import collect_results
from . import datasets
from . import networks
from . import prepost
from . import setting
from . import siml_manager


class Inferer(siml_manager.SimlManager):

    @classmethod
    def read_settings(cls, settings_yaml, **kwargs):
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

    def __init__(
            self, settings, *,
            conversion_function=None, load_function=None,
            data_addition_function=None, postprocess_function=None):
        """Initialize Inferer object.

        Parameters
        ----------
        settings: siml.MainSetting
        conversion_function: function, optional [None]
            Conversion function to preprocess raw data. It should receive
            two parameters, fem_data and raw_directory. If not fed,
            no additional conversion occurs.
        load_function: function, optional [None]
            Function to load data, which take list of pathlib.Path objects
            (as required files) and pathlib.Path object (as data directory)
            and returns data_dictionary and fem_data (can be None) to be saved.
        data_addition_function: function, optional [None]
            Function to add some data at simulation data writing phase.
            If not fed, no data addition occurs.
        postprocess_function: function, optional [None]
            Function to make postprocess of the inference data.
            If not fed, no additional postprocess will be performed.
        """
        self.setting = settings
        self.conversion_function = conversion_function
        self.load_function = load_function
        self.data_addition_function = data_addition_function
        self.postprocess_function = postprocess_function
        return

    def infer(self, *, data_directories=None, model=None):
        """Perform infererence.

        Parameters
        ----------
        data_directories: List[pathlib.Path], optional
            List of data directories. Data is searched recursively.
            The default is an empty list.
        model: pathlib.Path, optional
            If fed, overwrite self.setting.trainer.pretrain_directory.

        Returns
        -------
        inference_results: List[Dict]
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
        if data_directories is not None:
            if isinstance(data_directories, pathlib.Path):
                data_directories = [data_directories]
            self.setting.inferer.data_directories = data_directories
        if model is not None:
            self.setting.inferer.model = model

        self._prepare_inference()
        inference_state = self.inferer.run(self.inference_loader)
        if 'results' in inference_state.metrics:
            return inference_state.metrics['results']
        else:
            return None

    def infer_simplified_model(
            self, model_path, raw_dict_x, *, answer_raw_dict_y=None):
        """
        Infer with simplified model.

        Parameters
        ----------
            model_path: pathlib.Path
                Model file or directory name.
            raw_dict_x: dict
                Dict of raw x data.
            answer_raw_dict_y: dict, optional [None]
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
        if model_path is not None:
            self.setting.inferer.model = model_path
        self._prepare_inference(
            raw_dict_x=raw_dict_x, answer_raw_dict_y=answer_raw_dict_y)
        inference_state = self.inferer.run(self.inference_loader)
        if 'results' in inference_state.metrics:
            return inference_state.metrics['results'][0]
        else:
            return None

    def infer_parameter_study(
            self, model, data_directories, *, n_interpolation=100,
            converter_parameters_pkl=None):
        """
        Infer with performing parameter study. Parameter study is done with the
        data generated by interpolating the input data_directories.

        Parameters
        ----------
        model: pathlib.Path or io.BufferedIOBase, optional [None]
            Model directory, file path, or buffer. If not fed,
            TrainerSetting.pretrain_directory will be used.
        data_directories: List[pathlib.Path]
            List of data directories.
        n_interpolation: int, optional [100]
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

    def _prepare_inference(self, *, raw_dict_x=None, answer_raw_dict_y=None):

        # Define model
        if self.setting.inferer.model is None:
            if self.setting.trainer.pretrain_directory is None:
                raise ValueError(
                    f'No pretrain directory is specified for inference.')
            else:
                model = pathlib.Path(self.setting.trainer.pretrain_directory)
        else:
            model = pathlib.Path(self.setting.inferer.model)
        self.setting.trainer.restart_directory = None

        model = pathlib.Path(model)
        if model.is_dir():
            self.setting.trainer.pretrain_directory = model
            self._update_setting_if_needed()
            model_file = None
        elif model.is_file():
            model_file = model
        else:
            raise ValueError(f"Model does not exist: {model}")

        self.model = networks.Network(
            self.setting.model, self.setting.trainer)
        self._select_device()
        self._load_pretrained_model_if_needed(model_file=model_file)

        self.element_wise = self._determine_element_wise()
        self.loss = self._create_loss_function(allow_no_answer=True)
        if self.setting.inferer.converter_parameters_pkl is None:
            self.setting.inferer.converter_parameters_pkl \
                = self.setting.data.preprocessed_root / 'preprocessors.pkl'
        self.prepost_converter = prepost.Converter(
            self.setting.inferer.converter_parameters_pkl)

        self.inference_loader = self._get_inferernce_loader(
            raw_dict_x, answer_raw_dict_y)
        self.inferer = self._create_inferer()
        return

    def _get_inferernce_loader(self, raw_dict_x=None, answer_raw_dict_y=None):
        input_is_dict = isinstance(self.setting.trainer.inputs, dict)
        output_is_dict = isinstance(self.setting.trainer.outputs, dict)
        self.collate_fn = datasets.CollateFunctionGenerator(
            time_series=self.setting.trainer.time_series,
            dict_input=input_is_dict, dict_output=output_is_dict,
            use_support=self.setting.trainer.support_inputs,
            element_wise=self.element_wise,
            data_parallel=self.setting.trainer.data_parallel)
        self.prepare_batch = self.collate_fn.prepare_batch

        setting = self.setting
        if raw_dict_x is not None:
            inference_dataset = datasets.SimplifiedDataset(
                setting.trainer.input_names,
                setting.trainer.output_names,
                raw_dict_x=raw_dict_x, answer_raw_dict_y=answer_raw_dict_y,
                prepost_converter=self.prepost_converter,
                num_workers=0)
        elif setting.inferer.perform_preprocess:
            inference_dataset = datasets.PreprocessDataset(
                setting.trainer.input_names,
                setting.trainer.output_names,
                setting.inferer.data_directories,
                supports=setting.trainer.support_inputs,
                num_workers=0,
                required_file_names=setting.conversion.required_file_names,
                decrypt_key=setting.data.encrypt_key,
                prepost_converter=self.prepost_converter,
                conversion_setting=setting.conversion,
                conversion_function=self.conversion_function,
                load_function=self.load_function)
        else:
            inference_dataset = datasets.LazyDataset(
                setting.trainer.input_names,
                setting.trainer.output_names,
                setting.inferer.data_directories,
                supports=setting.trainer.support_inputs,
                num_workers=0,
                decrypt_key=setting.data.encrypt_key)

        inference_loader = torch.utils.data.DataLoader(
            inference_dataset, collate_fn=self.collate_fn,
            batch_size=1, shuffle=False, num_workers=0)
        return inference_loader

    def _create_inferer(self):

        def _inference(engine, batch):
            self.model.eval()
            x, y = self.prepare_batch(
                batch, device=self.device,
                output_device=self.output_device,
                non_blocking=self.setting.trainer.non_blocking)
            with torch.no_grad():
                start_time = time.time()
                y_pred = self.model(x)
                end_time = time.time()
                elapsed_time = end_time - start_time

            assert len(batch['data_directories']) == 1
            data_directory = batch['data_directories'][0]
            loss = self.loss(y_pred, y)
            print('--')
            print(f"              Data: {data_directory}")
            print(f"Inference time [s]: {elapsed_time:.5e}")
            print(f"              Loss: {loss}")

            return y_pred, y, {
                'x': x, 'original_shapes': x['original_shapes'],
                'data_directory': data_directory,
                'inference_time': elapsed_time}

        evaluator_engine = ignite.engine.Engine(_inference)

        if self.setting.inferer.return_all_results:
            metrics = {
                'results': collect_results.CollectResults(inferer=self),
            }
        else:
            metrics = {
                'loss': ignite.metrics.Loss(self.loss)
            }

        for name, metric in metrics.items():
            metric.attach(evaluator_engine, name)
        return evaluator_engine

    def _separate_data(self, data, descriptions, *, axis=-1):
        if isinstance(data, dict):
            return {
                key:
                self._separate_data(data[key], descriptions[key], axis=axis)
                for key in data.keys()}

        data_dict = {}
        index = 0
        data = np.swapaxes(data, 0, axis)
        for description in descriptions:
            dim = description.get('dim', 1)
            data_dict.update({
                description['name']:
                np.swapaxes(data[index:index+dim], 0, axis)})
            index += dim
        return data_dict
