import os

import dill
import yaml


class ColumnStorageMixin:

    _write_together: bool = True

    @property
    def write_together(self):
        return self._write_together

    def write(
        self,
        path: str,
        write_together: bool = None,
    ) -> None:

        # If unspecified, use the column's property to decide whether to write together
        if write_together is None:
            write_together = self.write_together

        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the column state
        state = self.get_state()

        # Remove the actual data and put into a metadata dict
        del state["_data"]
        metadata = {
            "dtype": type(self),
            "data_dtypes": list(map(type, self.data)),
            "len": len(self),
            "write_together": write_together,
            "state": state,
            **self.metadata,
        }

        if write_together:
            # Get the paths where metadata and data should be stored
            data_path = os.path.join(path, "data.dill")

            # Saving all column data in a single dill file
            dill.dump(
                [element.get_state() for element in self.data], open(data_path, "wb")
            )
        else:
            # Save all the elements of the column separately
            data_paths = []
            for index, element in enumerate(self.data):
                data_path = os.path.join(path, f"element_{index}")
                element.write(data_path)
                data_paths.append(data_path)
            metadata["data_paths"] = data_paths

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        yaml.dump(metadata, open(metadata_path, "w"))

    @classmethod
    def read(cls, path: str, *args, **kwargs):
        # Load in the metadata
        metadata = dict(
            yaml.load(
                open(os.path.join(path, "meta.yaml")),
                Loader=yaml.FullLoader,
            )
        )

        # Load the data
        if metadata["write_together"]:
            data = dill.load(open(os.path.join(path, "data.dill"), "rb"))
            data = [
                dtype.from_state(state, *args, **kwargs)
                for dtype, state in zip(metadata["data_dtypes"], data)
            ]
        else:
            data = [
                dtype.read(path, *args, **kwargs)
                for dtype, path in zip(metadata["data_dtypes"], metadata["data_paths"])
            ]

        # Load in the column from the state
        state = metadata["state"]
        state["_data"] = data
        return cls.from_state(state, *args, **kwargs)
