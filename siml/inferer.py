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
from . import util


class Inferer(siml_manager.SimlManager):

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

        Returns
        -------
        inference_results: list
            Inference results contains:
                - input variables
                - output variables
                - loss
        data_directories: List[pathlib.Path] or pathlib.Path
            Data directories to infer.
        model: pathlib.Path
            Model directory or file path.
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

    def _prepare_inference(self):

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

        self.inference_loader = self._get_inferernce_loader()
        self.inferer = self._create_inferer()
        return

    def _get_inferernce_loader(self):
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
        if setting.inferer.perform_preprocess:
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

    def infer_simplified_model(
            self, model_path, raw_dict_x, *,
            answer_raw_dict_y=None, model_directory=None,
            converter_parameters_pkl=None, accomodate_length=None):
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
            model_directory: pathlib.Path
                Model directory name.
            converter_parameters_pkl: pathlib.Path
                Converter parameters pkl data.
            accomodate_length: int
                If specified, duplicate initial state to initialize RNN state.
        """
        self._prepare_inference(
            pathlib.Path(model_path), model_directory=model_directory,
            converter_parameters_pkl=converter_parameters_pkl)

        # Preprocess data
        preprocessed_x = self.prepost_converter.preprocess(raw_dict_x)
        x = np.concatenate(
            [
                preprocessed_x[variable_name]
                for variable_name in self.setting.trainer.input_names],
            axis=-1).astype(np.float32)

        if answer_raw_dict_y is not None:
            answer_preprocessed_y = self.prepost_converter.preprocess(
                answer_raw_dict_y)
            answer_y = np.concatenate(
                [
                    answer_preprocessed_y[variable_name]
                    for variable_name in self.setting.trainer.output_names],
                axis=-1).astype(np.float32)
        else:
            answer_y = None

        _, inversed_dict_y, _, loss, _ = self._infer_single_data(
            self.prepost_converter, x, answer_y=answer_y,
            accomodate_length=accomodate_length)
        return inversed_dict_y, loss

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

    def _infer_single_data(
            self, postprocessor, x, *, answer_y=None,
            overwrite=False, supports=None,
            output_directory=None, write_simulation=False, write_npy=True,
            write_simulation_base=None, write_simulation_stem=None,
            write_simulation_type='fistr', read_simulation_type='fistr',
            data_addition_function=None, accomodate_length=None,
            load_function=None, required_file_names=[],
            convert_to_order1=False):

        if supports is not None:
            converted_supports = [
                datasets.merge_sparse_tensors(
                    [datasets.pad_sparse(s)], return_coo=True).to(self.device)
                for s in supports[0]]
        else:
            converted_supports = None

        if accomodate_length:
            x = np.concatenate([x[:accomodate_length], x])

        if self.setting.trainer.time_series:
            shape_length = 2
        else:
            shape_length = 1
        if isinstance(x, dict):
            shape = list(x.values())[0].shape
            original_shapes = {
                key: [value.shape[:shape_length]] for key, value in x.items()}
        else:
            shape = x.shape
            original_shapes = [shape[:shape_length]]

        # Inference
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            inferred_y = self.model({
                'x': x, 'supports': converted_supports,
                'original_shapes': original_shapes})
            end_time = time.time()
            elapsed_time = end_time - start_time
        if accomodate_length:
            inferred_y = inferred_y[accomodate_length:]
            x = x[accomodate_length:]

        if isinstance(x, dict):
            x = {key: value.cpu().numpy() for key, value in x.items()}
        else:
            x = x.cpu().numpy()

        if isinstance(inferred_y, dict):
            np_inferred_y = {
                key: value.cpu().numpy() for key, value in inferred_y.items()}
        else:
            np_inferred_y = inferred_y.cpu().numpy()

        dict_var_x = self._separate_data(
            x, self.setting.trainer.inputs)
        dict_var_inferred_y = self._separate_data(
            np_inferred_y, self.setting.trainer.outputs)
        if answer_y is not None:
            if isinstance(answer_y, np.ndarray):
                np_answer_y = answer_y
                answer_y = torch.from_numpy(answer_y).to(self.device)
            else:
                if isinstance(answer_y, dict):
                    np_answer_y = {
                        key: value.numpy() for key, value in answer_y.items()}
                    answer_y = {
                        key: value.to(self.device)
                        for key, value in answer_y.items()}
                else:
                    np_answer_y = answer_y.numpy()
                    answer_y = answer_y.to(self.device)

            dict_var_answer_y = self._separate_data(
                np_answer_y, self.setting.trainer.outputs)
            if isinstance(list(dict_var_x.values())[0], dict) \
                    and not isinstance(
                        list(dict_var_answer_y.values())[0], dict):
                dict_var_x.update({'t': dict_var_answer_y})
            else:
                dict_var_x.update(dict_var_answer_y)

        # Postprocess
        if not hasattr(self, 'perform_postprocess'):
            self.perform_postprocess = True
        inversed_dict_x, inversed_dict_y, fem_data = postprocessor.postprocess(
            dict_var_x, dict_var_inferred_y,
            output_directory=output_directory, overwrite=overwrite,
            write_simulation=write_simulation, write_npy=write_npy,
            write_simulation_base=write_simulation_base,
            write_simulation_stem=write_simulation_stem,
            write_simulation_type=write_simulation_type,
            read_simulation_type=read_simulation_type,
            skip_femio=self.setting.conversion.skip_femio,
            load_function=load_function, convert_to_order1=convert_to_order1,
            required_file_names=required_file_names,
            data_addition_function=data_addition_function,
            perform_inverse=self.perform_postprocess)

        # Compute loss
        if answer_y is not None:
            with torch.no_grad():
                loss = self.loss(inferred_y, answer_y).cpu().numpy()
        else:
            # Answer data does not exist
            loss = None

        return inversed_dict_x, inversed_dict_y, fem_data, loss, elapsed_time

    def _infer_single_directory(
            self, postprocessor, directory, x, dict_dir_y, *, save=True,
            overwrite=False,
            output_directory=None, write_simulation=False, write_npy=True,
            write_yaml=True, convert_to_order1=False,
            write_simulation_base=None, write_simulation_stem=None,
            write_simulation_type='fistr', read_simulation_type='fistr',
            data_addition_function=None, accomodate_length=False,
            load_function=None, required_file_names=[]):

        if isinstance(x, list):
            x, supports = x
        else:
            supports = None

        if directory in dict_dir_y:
            # Answer data exists
            answer_y = dict_dir_y[directory]
        else:
            answer_y = None

        if save:
            if output_directory is None:
                try:
                    output_directory = prepost.determine_output_directory(
                        directory, self.setting.data.inferred_root,
                        self.setting.data.preprocessed_root.name) \
                        / f"{self.setting.trainer.name}_{util.date_string()}"
                except ValueError:
                    output_directory = prepost.determine_output_directory(
                        directory, self.setting.data.inferred_root,
                        self.setting.data.raw_root.name) \
                        / f"{self.setting.trainer.name}_{util.date_string()}"

            output_directory.mkdir(parents=True, exist_ok=overwrite)
        else:
            output_directory = None

        inversed_dict_x, inversed_dict_y, fem_data, loss, inference_time = \
            self._infer_single_data(
                postprocessor, x, answer_y=answer_y, overwrite=overwrite,
                output_directory=output_directory, supports=supports,
                write_simulation=write_simulation, write_npy=write_npy,
                write_simulation_base=write_simulation_base,
                write_simulation_stem=write_simulation_stem,
                write_simulation_type=write_simulation_type,
                read_simulation_type=read_simulation_type,
                data_addition_function=data_addition_function,
                accomodate_length=accomodate_length,
                load_function=load_function,
                required_file_names=required_file_names,
                convert_to_order1=convert_to_order1)

        print(f"data: {directory}")
        print(f"Inference time: {inference_time}")
        if loss is not None:
            print(f"loss: {loss}")

        if save:
            if write_yaml:
                setting.write_yaml(
                    self.setting, output_directory / 'settings.yml',
                    overwrite=overwrite)
            with open(output_directory / 'log.dat', 'w') as f:
                f.write(f"inference time: {inference_time}\n")
                if loss is not None:
                    f.write(f"loss: {loss}\n")
            print(f"Inferred data saved in: {output_directory}")

        return {
            'dict_x': inversed_dict_x, 'dict_y': inversed_dict_y,
            'fem_data': fem_data, 'loss': loss,
            'output_directory': output_directory, 'data_directory': directory,
            'inference_time': inference_time}

    def _load_data(
            self, variable_names, directories, *,
            return_dict=False, supports=None, allow_missing=False):
        if isinstance(variable_names, dict):
            first_variable_name = list(variable_names.values())[0][0]
        else:
            first_variable_name = variable_names[0]

        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=[f"{first_variable_name}.npy"])
        data_directories = np.unique(data_directories)

        if len(data_directories) == 0:
            if allow_missing:
                return None
            else:
                raise ValueError(f"No data found in {directories}")

        if supports is None:
            supports = []

        dataset = datasets.BaseDataset(
            variable_names, [],
            [], supports=supports, allow_no_data=True,
            decrypt_key=self.setting.data.encrypt_key)
        data = [
            dataset._load_from_names(
                data_directory, variable_names) for data_directory
            in data_directories]
        support_data = [
            [
                util.load_variable(data_directory, support)
                for support in supports]
            for data_directory in data_directories]
        if len(data) == 0:
            raise ValueError(f"No data found for: {directories}")
        if self.setting.trainer.element_wise \
                or self.setting.trainer.simplified_model:
            if len(support_data[0]) > 0:
                raise ValueError(
                    'Cannot use support_input if '
                    'element_wise or simplified_model is True')
            if return_dict:
                return {
                    data_directory: d for data_directory, d
                    in zip(data_directories, data)}
            else:
                return np.concatenate(data), None
        if return_dict:
            if len(supports) > 0:
                return {
                    data_directory: [d, [s]] for data_directory, d, s
                    in zip(data_directories, data, support_data)}
            else:
                return {
                    data_directory: d for data_directory, d
                    in zip(data_directories, data)}
        else:
            return data, support_data

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
