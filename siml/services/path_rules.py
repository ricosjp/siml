import pathlib
from typing import Union, Optional
from types import MappingProxyType
import dataclasses as dc

import numpy as np

from siml.base.siml_enums import DirectoryType

# Reference: Rules of directory
# https://drivendata.github.io/cookiecutter-data-science/


@dc.dataclass(init=True, frozen=True)
class SimlPathRules:
    _type_to_name: MappingProxyType = MappingProxyType(
        {
            DirectoryType.RAW: "raw",
            DirectoryType.INTERIM: "interim",
            DirectoryType.PREPROCESSED: "preprocessed"
        }
    )

    def is_target_directory_type(
        self,
        data_directory: pathlib.Path,
        dir_type: DirectoryType
    ) -> bool:

        target_name = self._type_to_name[dir_type]

        # search from leaf to root
        for name in data_directory.parts[::-1]:
            if name == target_name:
                return True

        return False

    def detect_directory_type(
        self,
        data_directory: pathlib.Path
    ) -> Union[DirectoryType, None]:
        for dir_type in self._type_to_name.keys():
            if self.is_target_directory_type(
                data_directory,
                dir_type
            ):
                return dir_type

        return None

    def determine_output_directory(
        self,
        data_directory: pathlib.Path,
        output_base_directory: pathlib.Path,
        *,
        allowed_type: Optional[DirectoryType] = None
    ) -> pathlib.Path:
        """Determine output directory by replacing a string
         in the input_directory according to directory type of
         data directory.

        Parameters
        ----------
        input_directory: pathlib.Path
            Input directory path.
        output_base_directory: pathlib.Path
            Output base directory path. The output directry name is under that
            directory.

        Returns
        -------
        output_directory: pathlib.Path
            Detemined output directory path.

        Examples
        --------
        input_direcotry = "./aaa/bbb/interim/ddd"
        output_base_directory = "./aaa/ccc"

        output_directory = "./aaa/ccc/bbb/ddd"
        """

        dir_type = self.detect_directory_type(data_directory)
        if dir_type is None:
            return output_base_directory

        if allowed_type is not None:
            if dir_type != allowed_type:
                raise ValueError(
                    f"allowed directory type is {allowed_type.value}."
                    f"Input: {data_directory}, DirType: {dir_type.value}"
                )

        output_directory = self._determine_output_directory(
            data_directory,
            output_base_directory,
            self._type_to_name[dir_type]
        )
        return output_directory

    def determine_write_simulation_case_dir(
        self,
        data_directory: pathlib.Path,
        write_simulation_base: Optional[pathlib.Path] = None
    ) -> Union[pathlib.Path, None]:

        dir_type = self.detect_directory_type(data_directory)
        if write_simulation_base is None:
            if dir_type != DirectoryType.PREPROCESSED:
                return None
            for dir_type in [DirectoryType.RAW, DirectoryType.INTERIM]:
                case_dir = self.switch_directory_type(
                    data_directory,
                    dir_type_to=dir_type
                )
                if case_dir.is_dir():
                    return case_dir
            return None

        if dir_type == DirectoryType.RAW:
            return data_directory

        path = self.determine_output_directory(
            data_directory,
            output_base_directory=write_simulation_base
        )
        return path

    def _determine_output_directory(
        self,
        input_directory: pathlib.Path,
        output_base_directory: pathlib.Path,
        str_replace_from: str,
        *,
        str_replace_to: Optional[str] = ""
    ) -> pathlib.Path:
        """Determine output directory by replacing a string (str_replace)
         in the input_directory.

        Parameters
        ----------
        input_directory: pathlib.Path
            Input directory path.
        output_base_directory: pathlib.Path
            Output base directory path. The output directry name is under that
            directory.
        str_replace_from: str
            The string to be replaced.
        str_replace_to: str
            The string to replace.

        Returns
        -------
        output_directory: pathlib.Path
            Detemined output directory path.

        Examples
        --------
        input_direcotry = "./aaa/bbb/interim/ddd"
        output_base_directory = "./aaa/ccc"
        str_replace = "interim"

        output_directory = "./aaa/ccc/bbb/ddd"
        """
        common_parent = self.common_parent(
            input_directory,
            output_base_directory
        )
        relative_input_path = input_directory.relative_to(common_parent)

        parts = self._replace_parts(
            relative_input_path,
            from_item=str_replace_from,
            to_item=str_replace_to
        )

        output_directory = output_base_directory.joinpath(*parts)
        return output_directory

    def switch_directory_type(
        self,
        directory: pathlib.Path,
        dir_type_to: DirectoryType
    ) -> pathlib.Path:

        dir_type = self.detect_directory_type(directory)
        if dir_type is None:
            return None

        parts = self._replace_parts(
            directory,
            from_item=self._type_to_name[dir_type],
            to_item=self._type_to_name[dir_type_to]
        )

        path = pathlib.Path()
        return path.joinpath(*parts)

    def common_parent(
        self,
        directory_1: pathlib.Path,
        directory_2: pathlib.Path
    ) -> pathlib.Path:
        """Search common parent directory

        Parameters
        ----------
        directory_1 : pathlib.Path
        directory_2 : pathlib.Path

        Returns
        -------
        common_parent: pathlib.Path
            Path to common parent directory
        """
        parts_1 = directory_1.parts
        parts_2 = directory_2.parts
        min_len = min(len(parts_1), len(parts_2))

        common_parts = []
        for i in range(min_len):
            if parts_1[i] == parts_2[i]:
                common_parts.append(parts_1[i])
            else:
                break
        return pathlib.Path("").joinpath(*common_parts)

    def _replace_parts(
        self,
        directory: pathlib.Path,
        from_item: str,
        to_item: str
    ) -> list[str]:

        parts = list(directory.parts)
        replace_indices = np.where(
            np.array(parts) == from_item
        )[0]

        if len(replace_indices) > 1:
            raise ValueError(
                f"Input directory {directory} contains several "
                f"{from_item} parts, thus ambiguous.")

        for idx in replace_indices:
            parts[idx] = to_item

        parts = [p for p in parts if len(p) != 0]
        return parts
