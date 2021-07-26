from typing import Dict, List, Optional

from meerkat.tools.lazy_loader import LazyLoader
from torch import cuda

from robustnessgym.core.operation import Operation
from robustnessgym.core.slice import SliceDataPanel as DataPanel

bootleg_annotator = LazyLoader(
    "bootleg.end2end.bootleg_annotator",
    warning="Bootleg not available for import. Please see "
    "https://bootleg.readthedocs.io/en/latest/gettingstarted/install.html "
    "for help getting started.",
)


class BootlegAnnotatorOp(Operation):
    def __init__(
        self,
        config: Optional[Dict] = None,
        device: Optional[int] = None,
        cand_map: Optional[int] = None,
        threshold: Optional[float] = 0.5,
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):

        # Pass threshold to Operation to use as identifier
        super(BootlegAnnotatorOp, self).__init__(threshold=threshold)

        # Set the device
        if not device:
            device = "cuda" if cuda.is_available() else "cpu"

        # Create the annotator
        self.annotator = bootleg_annotator.BootlegAnnotator(
            config=config,
            device=device,
            cand_map=cand_map,
            threshold=threshold,
            cache_dir=cache_dir,
            model_name=model_name,
        )

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        """Annotate text samples using a Bootleg Annotator.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            **kwargs: optional keyword arguments

        Returns:
            Tuple with single output, a list of Bootleg annotations.
        """
        return ([self.annotator.label_mentions(text) for text in dp[columns[0]]],)
