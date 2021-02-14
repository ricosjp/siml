
from ignite.metrics.metric import Metric, reinit__is_reduced
import numpy as np
import torch

from . import datasets
from . import prepost


class CollectResults(Metric):

    def __init__(self, inferer):
        super().__init__()
        self.inferer = inferer
        return

    @reinit__is_reduced
    def reset(self):
        self._results = []
        return

    @reinit__is_reduced
    def update(self, data):

        y_pred, y = data[0], data[1]
        x = data[2]['x']
        data_directory = data[2]['data_directory']
        inference_time = data[2]['inference_time']
        loss = self.inferer.loss(y_pred, y, x['original_shapes'])

        dict_var_x = self.inferer._separate_data(
            self._to_numpy(x['x']), self.inferer.setting.trainer.inputs)
        dict_var_y = self.inferer._separate_data(
            self._to_numpy(y), self.inferer.setting.trainer.outputs)
        dict_var_y_pred = self.inferer._separate_data(
            self._to_numpy(y_pred), self.inferer.setting.trainer.outputs)

        output_directory = self._determine_output_directory(data_directory)
        write_simulation_base = self._determine_write_simulation_base(
            data_directory)

        setting = self.inferer.setting
        inversed_dict_x, inversed_dict_y, fem_data \
            = self.inferer.prepost_converter.postprocess(
                dict_var_x, dict_var_y_pred,
                output_directory=output_directory,
                dict_data_y_answer=dict_var_y,
                skip_femio=setting.conversion.skip_femio,
                load_function=self.inferer.load_function,
                data_addition_function=self.inferer.data_addition_function,
                overwrite=setting.inferer.overwrite,
                save_x=setting.inferer.save,
                write_simulation=setting.inferer.write_simulation,
                write_npy=setting.inferer.write_npy,
                write_simulation_stem=setting.inferer.write_simulation_stem,
                write_simulation_base=write_simulation_base,
                read_simulation_type=setting.inferer.read_simulation_type,
                write_simulation_type=setting.inferer.write_simulation_type,
                convert_to_order1=setting.inferer.convert_to_order1,
                required_file_names=setting.conversion.required_file_names,
                perform_inverse=setting.inferer.perform_inverse)
        raw_loss = self._compute_raw_loss(
            inversed_dict_x, inversed_dict_y, x['original_shapes'])

        if self.inferer.postprocess_function is not None:
            inversed_dict_x, inversed_dict_y, fem_data \
                = self.inferer.postprocess_function(
                    inversed_dict_x, inversed_dict_y, fem_data)

        self._results.append({
            'dict_x': inversed_dict_x, 'dict_y': inversed_dict_y,
            'fem_data': fem_data,
            'loss': loss,
            'raw_loss': raw_loss,
            'output_directory': output_directory,
            'data_directory': data_directory,
            'inference_time': inference_time})
        return

    def _to_numpy(self, x):
        if isinstance(x, (datasets.DataDict, dict)):
            return {key: value.detach().numpy() for key, value in x.items()}
        else:
            return x.detach().numpy()

    def _determine_output_directory(self, data_directory):
        if self.inferer.setting.inferer.output_directory is not None:
            return self.inferer.setting.inferer.output_directory

        if 'preprocessed' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory,
                self.inferer.setting.inferer.output_directory_base,
                'preprocessed')
        elif 'interim' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory,
                self.inferer.setting.inferer.output_directory_base,
                'interim')
        elif 'raw' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory,
                self.inferer.setting.inferer.output_directory_base,
                'raw')
        else:
            output_directory \
                = self.inferer.setting.inferer.output_directory_base
        return output_directory

    def _determine_write_simulation_base(self, data_directory):
        if self.inferer.setting.inferer.write_simulation_base is None:
            return None

        if 'preprocessed' in str(data_directory):
            write_simulation_base = prepost.determine_output_directory(
                data_directory,
                self.inferer.setting.inferer.write_simulation_base,
                'preprocessed')

        elif 'interim' in str(data_directory):
            write_simulation_base = prepost.determine_output_directory(
                data_directory,
                self.inferer.setting.inferer.write_simulation_base,
                'interim')
        elif 'raw' in str(data_directory):
            write_simulation_base = data_directory
        else:
            write_simulation_base \
                = self.inferer.setting.inferer.write_simulation_base
        return write_simulation_base

    def compute(self):
        return self._results

    def _compute_raw_loss(self, dict_x, dict_y, original_shapes=None):
        y_keys = dict_y.keys()
        if not np.all([y_key in dict_x for y_key in y_keys]):
            return None  # No answer

        if isinstance(self.inferer.setting.trainer.output_names, dict):
            output_names = self.inferer.setting.trainer.output_names
            y_raw_pred = self._reshape_dict(output_names, dict_y)
            y_raw_answer = self._reshape_dict(output_names, dict_x)
        else:
            y_raw_pred = torch.from_numpy(
                np.concatenate([dict_y[k] for k in dict_y.keys()]))
            y_raw_answer = torch.from_numpy(
                np.concatenate([dict_x[k] for k in dict_y.keys()]))

        raw_loss = self.inferer.loss(
            y_raw_pred, y_raw_answer, original_shapes=original_shapes)
        if raw_loss is None:
            return None
        else:
            return raw_loss.numpy()

    def _reshape_dict(self, dict_names, data_dict):
        return {
            key:
            torch.from_numpy(np.concatenate([
                data_dict[variable_name] for variable_name in value]))
            for key, value in dict_names.items()}
