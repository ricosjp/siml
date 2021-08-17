import io
import pathlib
import shutil
import time

import ignite
import numpy as np
import pandas as pd
import torch

from . import datasets
from . import networks
from . import postprocessor
from . import prepost
from . import setting
from . import siml_manager
from . import util


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

    @classmethod
    def from_model_directory(cls, model_directory, decrypt_key=None, **kwargs):
        """Load model data from a deployed directory.

        Parameters
        ----------
        model_directory: str or pathlib.Path
            Model directory created with Inferer.deploy().
        decrypt_key: bytes, optional
            Key to decrypt model data. If not fed, and the data is encrypted,
            ValueError is raised.

        Returns
        --------
        siml.Inferer
        """
        model_directory = pathlib.Path(model_directory)
        if (model_directory / 'settings.yml').is_file():
            main_setting = setting.MainSetting.read_settings_yaml(
                model_directory / 'settings.yml')
        elif (model_directory / 'settings.yml.enc').is_file():
            main_setting = setting.MainSetting.read_settings_yaml(
                util.decrypt_file(
                    decrypt_key, model_directory / 'settings.yml.enc',
                    return_stringio=True))
        else:
            raise ValueError('No setting yaml file found')

        obj = cls(main_setting, **kwargs)
        obj.setting.inferer.model_key = decrypt_key

        if (model_directory / 'model').is_file():
            obj.setting.inferer.model = model_directory / 'model'
        elif (model_directory / 'model.enc').is_file():
            obj.setting.inferer.model = model_directory / 'model.enc'
        else:
            obj.setting.inferer.model = obj._select_snapshot(model_directory)

        if (model_directory / 'preprocessors.pkl').is_file():
            obj.setting.inferer.converter_parameters_pkl \
                = model_directory / 'preprocessors.pkl'
        elif (model_directory / 'preprocessors.pkl.enc').is_file():
            obj.setting.inferer.converter_parameters_pkl \
                = model_directory / 'preprocessors.pkl.enc'
        else:
            raise ValueError('No preprocessor pickle file found')

        return obj

    def __init__(
            self, settings, *,
            model=None, converter_parameters_pkl=None, save=None,
            conversion_function=None, load_function=None,
            data_addition_function=None, postprocess_function=None,
            save_function=None):
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
        super().__init__(settings)
        self.conversion_function = conversion_function
        self.load_function = load_function
        self.data_addition_function = data_addition_function
        self.postprocess_function = postprocess_function
        self.save_function = save_function

        if model is not None:
            self.setting.inferer.model = model
        if converter_parameters_pkl is not None:
            self.setting.inferer.converter_parameters_pkl \
                = converter_parameters_pkl
        if save is not None:
            self.setting.inferer.save = save

        return

    def infer(
            self, *, data_directories=None, model=None,
            perform_preprocess=None, output_directory_base=None):
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

        Returns
        -------
        inference_results: list[Dict]
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
        if perform_preprocess is not None:
            self.setting.inferer.perform_preprocess = perform_preprocess
        if output_directory_base is not None:
            self.setting.inferer.output_directory_base = output_directory_base

        self._prepare_inference()
        self.date_string = util.date_string()
        inference_state = self.inferer.run(self.inference_loader)

        if self.setting.inferer.save:
            self.save(inference_state.metrics['results'])
        return inference_state.metrics['results']

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
        if model_path is not None:
            self.setting.inferer.model = model_path

        self._prepare_inference(
            raw_dict_x=raw_dict_x, answer_raw_dict_y=answer_raw_dict_y)
        self.date_string = util.date_string()
        inference_state = self.inferer.run(self.inference_loader)

        if self.setting.inferer.save:
            self.save(inference_state.metrics['results'])
        return inference_state.metrics['results'][0]

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

    def save(self, results):
        """Save inference results information.

        Parameters
        ----------
        results: Dict
            Inference results.
        """
        output_directory = self._determine_output_directory()
        output_directory.mkdir(parents=True, exist_ok=True)
        setting.write_yaml(self.setting, output_directory / 'settings.yml')
        self._write_log(output_directory, results)
        return

    def _write_log(self, output_directory, results):
        column_names = [
            'loss', 'raw_loss', 'output_directory', 'data_directory',
            'inference_time']

        log_dict = {}
        for column_name in column_names:
            log_dict.update({column_name: [r[column_name] for r in results]})

        pd.DataFrame(log_dict).to_csv(output_directory / 'log.csv', index=None)
        return

    def deploy(self, output_directory, *, model=None, encrypt_key=None):
        """Deploy model information.

        Parameters
        ----------
        output_directory: pathlib.Path
            Output directory path.
        model: pathlib.Path, optional
            If fed, overwrite self.setting.inferer.model.
        encrypt_key: bytes, optional
            Key to encrypt model data. If not fed, the model data will not be
            encrypted.
        """
        if model is not None:
            self.setting.inferer.model = model
        if model.is_dir():
            snapshot = self._select_snapshot(
                self.setting.inferer.model,
                method=self.setting.trainer.snapshot_choise_method)
        else:
            snapshot = model

        output_directory.mkdir(parents=True, exist_ok=True)

        if encrypt_key is None:
            output_model = output_directory / 'model'
            if output_model.exists():
                raise ValueError(f"{output_model} already exists")
            shutil.copyfile(snapshot, output_model)

            output_pkl = output_directory / 'preprocessors.pkl'
            if output_pkl.exists():
                raise ValueError(f"{output_pkl} already exists")
            shutil.copyfile(
                self.setting.inferer.converter_parameters_pkl,
                output_pkl)

            output_setting = output_directory / 'settings.yml'
            if output_setting.exists():
                raise ValueError(f"{output_setting} already exists")
            setting.write_yaml(self.setting, output_setting)

        else:

            output_model = output_directory / 'model.enc'
            if output_model.exists():
                raise ValueError(f"{output_model} already exists")
            self._encrypt_file(
                encrypt_key, snapshot, output_directory / 'model.enc')

            output_pkl = output_directory / 'preprocessors.pkl.enc'
            if output_pkl.exists():
                raise ValueError(f"{output_pkl} already exists")
            self._encrypt_file(
                encrypt_key, self.setting.inferer.converter_parameters_pkl,
                output_pkl)

            output_setting = output_directory / 'settings.yml.enc'
            if output_setting.exists():
                raise ValueError(f"{output_setting} already exists")
            string = setting.dump_yaml(self.setting, None)
            bio = io.BytesIO(string.encode('utf8'))
            util.encrypt_file(encrypt_key, output_setting, bio)

        return

    def _encrypt_file(self, key, input_file_name, output_file_name):
        with open(input_file_name, "rb") as f:
            data = f.read()
            binary = io.BytesIO(data)
            util.encrypt_file(key, output_file_name, binary)
        return

    def _prepare_inference(
            self, *,
            raw_dict_x=None, answer_raw_dict_y=None, allow_no_data=False):

        # Define model
        if self.setting.inferer.model is None:
            if self.setting.trainer.pretrain_directory is None:
                raise ValueError(
                    'No pretrain directory is specified for inference.')
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
            self.setting.inferer.converter_parameters_pkl,
            key=self.setting.inferer.model_key)

        self.inference_loader = self._get_inferernce_loader(
            raw_dict_x, answer_raw_dict_y, allow_no_data=allow_no_data)
        self.inferer = self._create_inferer()
        return

    def _get_inferernce_loader(
            self, raw_dict_x=None, answer_raw_dict_y=None,
            allow_no_data=False):
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
                allow_no_data=allow_no_data,
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
                allow_no_data=allow_no_data,
                load_function=self.load_function)
        else:
            inference_dataset = datasets.LazyDataset(
                setting.trainer.input_names,
                setting.trainer.output_names,
                setting.inferer.data_directories,
                supports=setting.trainer.support_inputs,
                num_workers=0,
                allow_no_data=allow_no_data,
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
            loss = self.loss(y_pred, y, original_shapes=x['original_shapes'])
            print('--')
            print(f"              Data: {data_directory}")
            print(f"Inference time [s]: {elapsed_time:.5e}")
            if loss is not None:
                print(f"              Loss: {loss}")
            print('--')

            return y_pred, y, {
                'x': x, 'original_shapes': x['original_shapes'],
                'data_directory': data_directory,
                'inference_time': elapsed_time}

        evaluator_engine = ignite.engine.Engine(_inference)

        metrics = {'results': postprocessor.Postprocessor(inferer=self)}

        for name, metric in metrics.items():
            metric.attach(evaluator_engine, name)
        return evaluator_engine

    def _separate_data(self, data, descriptions, *, axis=-1):
        if isinstance(data, dict):
            return {
                key:
                self._separate_data(data[key], descriptions[key], axis=axis)
                for key in data.keys()}
        if len(data) == 0:
            return {}

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

    def _determine_output_directory(self, data_directory=None):
        if data_directory is None:
            data_directory = ''
        if self.setting.inferer.output_directory is not None:
            return self.setting.inferer.output_directory

        subdirectory = self._determine_subdirectory()
        base = self.setting.inferer.output_directory_base / subdirectory
        if 'preprocessed' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory, base, 'preprocessed')
        elif 'interim' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory, base, 'interim')
        elif 'raw' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory, base, 'raw')
        else:
            output_directory = base
        return output_directory

    def _determine_subdirectory(self):
        if self.setting.inferer.model is not None:
            model = pathlib.Path(self.setting.inferer.model)
            if model.is_dir():
                model_name = model.name
            else:
                model_name = model.parent.name
        elif self.setting.trainer.name is not None:
            model_name = self.setting.trainer.name
        else:
            model_name = 'unknown'
        return f"{model_name}_{self.date_string}"

    def _determine_write_simulation_base(self, data_directory):
        if self.setting.inferer.write_simulation_base is None:
            if self.setting.inferer.perform_preprocess:
                # Assume the given data is raw data
                return data_directory
            else:
                if 'preprocessed' in str(data_directory):
                    raw_candidate = pathlib.Path(
                        str(data_directory).replace('preprocessed', 'raw'))
                    if raw_candidate.is_dir():
                        return raw_candidate
                    else:
                        interim_candidate = pathlib.Path(
                            str(data_directory).replace(
                                'preprocessed', 'interim'))
                        if interim_candidate.is_dir():
                            return interim_candidate
                        else:
                            return None
                else:
                    return None

        if 'preprocessed' in str(data_directory):
            write_simulation_base = prepost.determine_output_directory(
                data_directory,
                self.setting.inferer.write_simulation_base,
                'preprocessed')

        elif 'interim' in str(data_directory):
            write_simulation_base = prepost.determine_output_directory(
                data_directory,
                self.setting.inferer.write_simulation_base,
                'interim')
        elif 'raw' in str(data_directory):
            write_simulation_base = data_directory
        else:
            write_simulation_base \
                = self.setting.inferer.write_simulation_base
        return write_simulation_base
