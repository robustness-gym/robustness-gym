"""Cachedop with Spacy."""
import json
from typing import List

import cytoolz as tz
import spacy
import torch
from spacy.tokens import Doc

from robustnessgym.core.cachedops import SingleColumnCachedOperation
from robustnessgym.core.dataset import BatchOrDataset


class Spacy(SingleColumnCachedOperation):
    """Class for running the Spacy pipeline using a CachedOperation."""

    def __init__(
        self,
        lang: str = "en_core_web_sm",
        nlp: spacy.language.Language = None,
        neuralcoref: bool = False,
        device: str = None,
        *args,
        **kwargs
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
            # which causes other GPU cachedops to crash.
            torch.set_default_tensor_type("torch.FloatTensor")
            self._on_gpu = True

        # Load up the Spacy module
        self._nlp = nlp
        if not nlp:
            self._nlp = self._load_spacy(lang=lang)
            self._prebuilt = False

        # Add neuralcoref
        self._add_neuralcoref()

        if not nlp:
            super(Spacy, self).__init__(
                lang=lang,
                neuralcoref=neuralcoref,
                *args,
                **kwargs,
            )
        else:
            super(Spacy, self).__init__(
                lang=nlp.lang,
                # No need to pass in neuralcoref separately, it's already in the
                # pipeline if neuralcoref=True
                pipeline=nlp.pipe_names,
                *args,
                **kwargs,
            )
            print(
                "Warning: Spacy.encode does not support arbitrary nlp pipelines so "
                "information stored in the Doc object may be lost in encoding."
            )

    @staticmethod
    def _load_spacy(lang: str = "en_core_web_sm"):
        """Load the Spacy nlp pipeline."""
        return spacy.load(lang)

    def _add_neuralcoref(self):
        """Add the neuralcoref pipeline to Spacy."""
        if self.neuralcoref:
            try:
                import neuralcoref as nc

                nc.add_to_pipe(self.nlp)
            except ImportError:
                print(
                    "Can't import neuralcoref. Please install neuralcoref using:\n"
                    "git clone https://github.com/huggingface/neuralcoref.git\n"
                    "cd neuralcoref\n"
                    "pip install -r requirements.txt\n"
                    "pip install -e ."
                )

    def __call__(
        self,
        batch_or_dataset: BatchOrDataset,
        columns: List[str],
        batch_size: int = 8192,
    ) -> BatchOrDataset:
        return super().__call__(batch_or_dataset, columns, batch_size)

    @property
    def nlp(self):
        """Return the nlp pipeline."""
        return self._nlp

    @classmethod
    def encode(cls, obj: Doc) -> str:
        """Encode the Doc object.

        Args:
            obj:

        Returns:
        """
        # JSON dump the Doc
        doc_json = obj.to_json()

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

            # Combine the neuralcoref dictionary with the doc_json
            doc_json = tz.merge(doc_json, neuralcoref_dict)

        # Convert the Spacy Doc to json before caching
        return json.dumps(doc_json)

    def single_column_apply(self, column_batch: List, *args, **kwargs) -> List:
        """Apply to a single column.

        Args:
            column_batch:
            *args:
            **kwargs:

        Returns:
        """
        if self._on_gpu:
            # Adjust the default Tensor type: this is instantaneous
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Apply Spacy's pipe method to process the examples
        docs = list(self.nlp.pipe(column_batch))

        if self._on_gpu:
            # Reset the default Tensor type: this is instantaneous
            torch.set_default_tensor_type("torch.FloatTensor")

        return docs

    @classmethod
    def tokens(cls, decoded_batch: List) -> List[List[str]]:
        """For each example, returns the list of tokens extracted by Spacy for
        each key.

        Spacy stores the span of each token under the "tokens" key. This
        function extracts the tokens from the text using the span of
        each token.
        """

        token_batch = []
        # Iterate over each decoded Doc dictionary
        for doc_dict in decoded_batch:
            tokens = []
            for token_dict in doc_dict["tokens"]:
                tokens.append(doc_dict["text"][token_dict["start"] : token_dict["end"]])

            token_batch.append(tokens)

        return token_batch

    @classmethod
    def entities(cls, decoded_batch: List) -> List[List[dict]]:
        """For each example, returns the list of entity extracted by Spacy for
        each column."""
        return [doc_dict["ents"] for doc_dict in decoded_batch]

    @classmethod
    def sentences(cls, decoded_batch: List) -> List[List[str]]:
        """For each example, returns the list of sentences extracted by Spacy
        for each column."""
        return [
            [
                doc_dict["text"][sent["start"] : sent["end"]]
                for sent in doc_dict["sents"]
            ]
            for doc_dict in decoded_batch
        ]

    @classmethod
    def num_tokens(cls, decoded_batch: List) -> List[int]:
        """For each example, returns the length or the number of tokens
        extracted by Spacy for each column."""
        return [len(doc_dict["tokens"]) for doc_dict in decoded_batch]
