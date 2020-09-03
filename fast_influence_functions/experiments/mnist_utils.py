import torch
import torchvision
from typing import List, Optional


class CustomMNIST(torchvision.datasets.MNIST):
    """Customizied MNIST Dataset that optionally skip examples"""

    def __init__(self,
                 indices_to_skip: Optional[List[int]] = None,
                 *args, **kwargs) -> None:
        super(CustomMNIST, self).__init__(*args, **kwargs)

        skipped_data: List[torch.Tensor] = []
        skipped_targets: List[torch.Tensor] = []
        if indices_to_skip is not None:
            if not isinstance(indices_to_skip, (list, tuple)):
                raise TypeError

            # https://discuss.pytorch.org/t/how-to-remove-an-element-from-a-1-d-tensor-by-index/23109
            for index_to_skip in indices_to_skip:
                skipped_data.append(self.data[index_to_skip])
                skipped_targets.append(self.targets[index_to_skip])

                self.data: torch.Tensor = torch.cat([
                    self.data[: index_to_skip],
                    self.data[index_to_skip + 1:]], dim=0)

                self.targets: torch.Tensor = torch.cat([
                    self.targets[: index_to_skip],
                    self.targets[index_to_skip + 1:]], dim=0)

        self._skipped_data = skipped_data
        self._skipped_targets = skipped_targets
        self._indices_to_skip = len(self.data)
