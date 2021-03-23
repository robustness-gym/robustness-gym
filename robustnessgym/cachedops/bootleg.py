from typing import Dict, List, Optional

from torch import cuda

from robustnessgym.core.cachedops import SingleColumnCachedOperation
from robustnessgym.core.decorators import singlecolumn

try:
    from bootleg.end2end.bootleg_annotator import BootlegAnnotator
except ImportError:
    _bootleg_available = False
else:
    _bootleg_available = True


class Bootleg(SingleColumnCachedOperation):
    def __init__(
        self,
        config: Optional[Dict] = None,
        device: Optional[int] = None,
        cand_map: Optional[int] = None,
        threshold: Optional[float] = 0.5,
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        *args,
        **kwargs
    ):
        if not _bootleg_available:
            raise ImportError(
                "Bootleg not available for import. Please see "
                "https://bootleg.readthedocs.io/en/latest/gettingstarted/install.html "
                "for help getting started."
            )
        # Pass threshold to SingleColumnCachedOperation to use as identifier
        super(Bootleg, self).__init__(threshold=threshold, *args, **kwargs)

        # Set the device
        if not device:
            device = "cuda" if cuda.is_available() else "cpu"

        # Create the annotator
        self.annotator = BootlegAnnotator(
            config=config,
            device=device,
            cand_map=cand_map,
            threshold=threshold,
            cache_dir=cache_dir,
            model_name=model_name,
        )

    @singlecolumn
    def apply(self, batch: Dict[str, List], columns: List[str], **kwargs) -> List:

        # Extract the test - Bootleg will automatically window sentences
        batch_text = batch[columns[0]]

        list_bootleg_results = [
            self.annotator.label_mentions(text) for text in batch_text
        ]

        return list_bootleg_results

    def encode(cls, obj) -> str:
        """Custom encode to allow for np.ndarray to be passed."""
        return obj

    def decode(cls, s: str):
        return s
