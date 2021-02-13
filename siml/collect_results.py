
from ignite.metrics.metric import Metric, reinit__is_reduced
import numpy as np
import torch

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

        y_pred, y = data[0].detach(), data[1].detach()
        x = data[2]['x']
        data_directory = data[2]['data_directory']
        inference_time = data[2]['inference_time']

        dict_var_x = self.inferer._separate_data(
            x['x'].numpy(), self.inferer.setting.trainer.inputs)
        dict_var_y = self.inferer._separate_data(
            y.numpy(), self.inferer.setting.trainer.outputs)
        dict_var_y_pred = self.inferer._separate_data(
            y_pred.numpy(), self.inferer.setting.trainer.outputs)

        output_directory = self._determine_output_directory(data_directory)
        write_simulation_base = self._determine_write_simulation_base(
            data_directory)

        setting = self.inferer.setting
        inversed_dict_x, inversed_dict_y, fem_data \
            = self.inferer.prepost_converter.postprocess(
                dict_var_x, dict_var_y_pred,
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
                required_file_names=setting.inferer.required_file_names,
                perform_inverse=setting.inferer.perform_inverse)
        loss = self._compute_raw_loss(inversed_dict_x, inversed_dict_y)

        if self.inferer.postprocess_function is not None:
            inversed_dict_x, inversed_dict_y, fem_data \
                = self.inferer.postprocess_function(
                    inversed_dict_x, inversed_dict_y, fem_data)

        self._results.append({
            'dict_x': inversed_dict_x, 'dict_y': inversed_dict_y,
            'fem_data': fem_data,
            'loss': loss,
            'output_directory': output_directory,
            'data_directory': data_directory,
            'inference_time': inference_time})
        return

    def _determine_output_directory(self, data_directory):
        if 'preprocessed' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory,
                self.inferer.setting.inferer.output_directory_root,
                'preprocessed')
        elif 'interim' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory,
                self.inferer.setting.inferer.output_directory_root,
                'interim')
        elif 'raw' in str(data_directory):
            output_directory = prepost.determine_output_directory(
                data_directory,
                self.inferer.setting.inferer.output_directory_root,
                'raw')
        else:
            output_directory \
                = self.inferer.setting.inferer.output_directory_root
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

    def _compute_raw_loss(self, dict_x, dict_y):
        y_keys = dict_y.keys()
        if not np.all([y_key in dict_x for y_key in y_keys]):
            return None  # No answer
        y_raw_pred = np.concatenate([dict_y[k] for k in dict_y.keys()])
        y_raw_answer = np.concatenate([dict_x[k] for k in dict_y.keys()])
        return self.inferer.loss(
            torch.from_numpy(y_raw_pred), torch.from_numpy(y_raw_answer))
