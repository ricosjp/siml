
import chainer as ch

from . import util


class LazyDataSet(ch.dataset.DatasetMixin):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None):
        self.x_variable_names = x_variable_names
        self.y_variable_names = y_variable_names
        self.supports = supports

        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=[f"{x_variable_names[0]}.npy"])
        self.data_directories = data_directories

    def __len__(self):
        return len(self.data_directories)

    def get_example(self, i):
        data_directory = self.data_directories[i]
        x_data = util.concatenate_variable([
            util.load_variable(data_directory, x_variable_name)
            for x_variable_name in self.x_variable_names])
        y_data = util.concatenate_variable([
            util.load_variable(data_directory, y_variable_name)
            for y_variable_name in self.y_variable_names])
        if self.supports is None:
            return {'x': x_data, 't': y_data}
        else:
            support_data = [
                util.load_variable(data_directory, support)
                for support in self.supports]
            return {'x': x_data, 't': y_data, 'supports': support_data}
