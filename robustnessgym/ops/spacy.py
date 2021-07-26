"""Spacy Operation."""
from __future__ import annotations

from typing import List

import torch
from meerkat import SpacyCell
from meerkat.tools.lazy_loader import LazyLoader

from robustnessgym.core.operation import Operation
from robustnessgym.core.slice import SliceDataPanel as DataPanel

spacy = LazyLoader("spacy", warning="Please `pip install spacy`.")
spacy_tokens = LazyLoader("spacy.tokens")
nc = LazyLoader(
    "neuralcoref",
    error="Can't import neuralcoref. Please install neuralcoref using:\n"
    "git clone https://github.com/huggingface/neuralcoref.git\n"
    "cd neuralcoref\n"
    "pip install -r requirements.txt\n"
    "pip install -e .",
)


class SpacyOp(Operation):
    """Operation that runs the Spacy pipeline."""

    def __init__(
        self,
        lang: str = "en_core_web_sm",
        nlp: spacy.language.Language = None,
        neuralcoref: bool = False,
        device: str = None,
    ):
        # Set all the parameters
        self.lang = lang
        self.neuralcoref = neuralcoref
        self._prebuilt = True

        # Set the device
        self._on_gpu = False
        if device and (device == "gpu" or device.startswith("cuda")):
            spacy.prefer_gpu(
                gpu_id=0 if ":" not in device else int(device.split(":")[1])
            )
            # Spacy sets the default torch float Tensor to torch.cuda.FloatTensor,
            # which causes other GPU operations to crash.
            torch.set_default_tensor_type("torch.FloatTensor")
            self._on_gpu = True

        # Load up the Spacy module
        self.nlp = nlp
        if not nlp:
            self.nlp = spacy.load(lang)
            self._prebuilt = False

        # Add neuralcoref
        if self.neuralcoref:
            nc.add_to_pipe(self.nlp)

        if not nlp:
            super(SpacyOp, self).__init__(
                lang=lang,
                neuralcoref=neuralcoref,
            )
        else:
            super(SpacyOp, self).__init__(
                lang=nlp.lang,
                # No need to pass in neuralcoref separately, it's already in the
                # pipeline if neuralcoref=True
                pipeline=nlp.pipe_names,
            )

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        """Process text examples by running them through the Spacy pipeline.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            **kwargs: optional keyword arguments

        Returns:
            Tuple with single output, a list of SpacyCell objects.
        """
        if self._on_gpu:
            # Adjust the default Tensor type: this is instantaneous
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Apply Spacy's pipe method to process the examples
        cells = [SpacyCell(doc) for doc in list(self.nlp.pipe(dp[columns[0]]))]

        if self._on_gpu:
            # Reset the default Tensor type: this is instantaneous
            torch.set_default_tensor_type("torch.FloatTensor")

        return (cells,)


"""
if obj._.has("huggingface_neuralcoref"):
    # Create a helper function that turns a Span into a dictionary
    span_to_dict = lambda span: {
        "start": span.start,
        "end": span.end,
        "text": span.text,
    }

    # Create a helper function that converts a Cluster (output of
    # neuralcoref) into a dictionary
    cluster_to_dict = lambda cluster: {
        "i": cluster.i,
        "main": span_to_dict(cluster.main),
        "mentions": [span_to_dict(span) for span in cluster.mentions],
    }

    # Apply the helper functions to construct a dictionary for the
    # neuralcoref information
    neuralcoref_dict = {
        "neuralcoref": [
            cluster_to_dict(cluster) for cluster in obj._.coref_clusters
        ]
    }
"""
