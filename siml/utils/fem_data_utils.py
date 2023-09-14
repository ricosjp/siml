import femio
import numpy as np


class FemDataWrapper:
    def __init__(self, fem_data: femio.FEMData) -> None:
        self.fem_data = fem_data

    def extract_variables(
        self,
        mandatory_variables: list[str],
        *,
        optional_variables: list[str] = None
    ) -> dict[str, np.ndarray]:
        """Extract variables from FEMData object to convert to data dictionary.

        Parameters
        ----------
        mandatory_variables: list[str]
            Mandatory variable names.
        optional_variables: list[str], optional
            Optional variable names.

        Returns
        -------
            dict_data: dict
                Data dictionary.
        """
        dict_data = {
            mandatory_variable: self._extract_single_variable(
                mandatory_variable, mandatory=True, ravel=True)
            for mandatory_variable in mandatory_variables}

        if optional_variables is None:
            return dict_data

        for optional_variable in optional_variables:
            optional_variable_data = self._extract_single_variable(
                optional_variable, mandatory=False, ravel=True)
            if optional_variable_data is not None:
                dict_data.update({optional_variable: optional_variable_data})
        return dict_data

    def _extract_single_variable(
        self,
        variable_name: str,
        *,
        mandatory: bool = True,
        ravel: bool = True
    ) -> np.ndarray:
        if variable_name in self.fem_data.nodal_data:
            return self.fem_data.nodal_data.get_attribute_data(variable_name)
        elif variable_name in self.fem_data.elemental_data:
            return self.fem_data.elemental_data.get_attribute_data(
                variable_name
            )
        else:
            if mandatory:
                raise ValueError(
                    f"{variable_name} not found in "
                    f"{self.fem_data.nodal_data.keys()}, "
                    f"{self.fem_data.elemental_data.keys()}"
                )
            else:
                return None

    def update_fem_data(
        self,
        dict_data: dict,
        prefix: str = '',
        *,
        allow_overwrite=False,
        answer_keys=None,
        answer_prefix=''
    ):
        for key, value in dict_data.items():

            variable_name = self._get_variable_name(
                key=key,
                prefix=prefix,
                answer_prefix=answer_prefix,
                answer_keys=answer_keys
            )

            if not isinstance(value, np.ndarray):
                print(f"{variable_name} is skipped to include in fem_data")
                continue

            value = reshape_data_if_needed(value)
            shape = value.shape

            if shape[0] == len(self.fem_data.nodes.ids):
                # Nodal data
                dict_data_to_update = {
                    variable_name: value}
                self.fem_data.nodal_data.update_data(
                    self.fem_data.nodes.ids, dict_data_to_update,
                    allow_overwrite=allow_overwrite)
            elif shape[1] == len(self.fem_data.nodes.ids):
                # Nodal data with time series
                if shape[0] == 1:
                    dict_data_to_update = {
                        variable_name: reshape_data_if_needed(value[0])}
                else:
                    dict_data_to_update = {
                        f"{variable_name}_{i}": reshape_data_if_needed(v)
                        for i, v in enumerate(value)}
                self.fem_data.nodal_data.update_data(
                    self.fem_data.nodes.ids, dict_data_to_update,
                    allow_overwrite=allow_overwrite)
            elif shape[0] == len(self.fem_data.elements.ids):
                # Elemental data
                dict_data_to_update = {
                    variable_name: value}
                self.fem_data.elemental_data.update_data(
                    self.fem_data.elements.ids, dict_data_to_update,
                    allow_overwrite=allow_overwrite)
            elif shape[1] == len(self.fem_data.elements.ids):
                # Elemental data with time series
                if shape[0] == 1:
                    dict_data_to_update = {
                        variable_name: reshape_data_if_needed(value[0])}
                else:
                    dict_data_to_update = {
                        f"{variable_name}_{i}": reshape_data_if_needed(v)
                        for i, v in enumerate(value)}
                self.fem_data.elemental_data.update_data(
                    self.fem_data.nodes.ids, dict_data_to_update,
                    allow_overwrite=allow_overwrite)
            else:
                print(f"{variable_name} is skipped to include in fem_data")
                continue

    def _get_variable_name(
        self,
        key: str,
        prefix: str,
        answer_prefix: str,
        answer_keys: list[str]
    ) -> str:
        if answer_keys is not None:
            if key in answer_keys:
                variable_name = answer_prefix + key
            else:
                variable_name = prefix + key
        else:
            variable_name = prefix + key

        return variable_name

    def add_difference(
        self,
        dict_data: dict,
        reference_dict_data: dict,
        prefix: str = 'difference'
    ) -> None:
        if reference_dict_data is None:
            return

        intersections = \
            set(dict_data.keys()).intersection(reference_dict_data.keys())
        if len(intersections) == 0:
            return

        difference_dict_data = {
            intersection:
            np.reshape(
                dict_data[intersection],
                reference_dict_data[intersection].shape
            )
            - reference_dict_data[intersection]
            for intersection in intersections
        }

        self.update_fem_data(
            difference_dict_data,
            prefix=prefix
        )

    def add_abs_difference(
        self,
        dict_data: dict,
        reference_dict_data: dict,
        prefix: str = 'difference_abs'
    ) -> None:
        if reference_dict_data is None:
            return

        intersections = \
            set(dict_data.keys()).intersection(reference_dict_data.keys())
        if len(intersections) == 0:
            return

        difference_dict_data = {
            intersection:
            np.abs(
                np.reshape(
                    dict_data[intersection],
                    reference_dict_data[intersection].shape
                )
                - reference_dict_data[intersection]
            )
            for intersection in intersections
        }

        self.update_fem_data(
            difference_dict_data,
            prefix=prefix
        )


def reshape_data_if_needed(value: np.ndarray):
    """Reshape numpy.ndarray-like to be writable to visualization files.

    Parameters
    ----------
    value: numpy.ndarray
        Data to be processed.

    Returns
    -------
    reshaped_data: numpy.ndarray
    """
    if len(value.shape) > 2 and value.shape[-1] == 1:
        if len(value.shape) == 4 and value.shape[1] == 3 \
                and value.shape[2] == 3:
            # NOTE: Assume this is symmetric matrix
            reshaped_value \
                = femio.functions.convert_symmetric_matrix2array(
                    value[..., 0])
        else:
            reshaped_value = value[..., 0]
    elif len(value.shape) == 1:
        reshaped_value = value[:, None]
    else:
        reshaped_value = value
    return reshaped_value
