import json
from typing import List, Dict

import cytoolz as tz
import spacy
from spacy.tokens import Doc

from robustnessgym.cached_ops.cached_ops import SingleColumnCachedOperation
from robustnessgym.dataset import BatchOrDataset


class Spacy(SingleColumnCachedOperation):

    def __init__(self,
                 lang: str = 'en_core_web_sm',
                 nlp: spacy.language.Language = None,
                 neuralcoref: bool = False,
                 *args,
                 **kwargs):

        # Set all the parameters
        self.lang = lang
        self.neuralcoref = neuralcoref
        self._prebuilt = True

        # Load up the Spacy module
        spacy.prefer_gpu()
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
                # No need to pass in neuralcoref separately, it's already in the pipeline if neuralcoref=True
                pipeline=nlp.pipe_names,
                *args,
                **kwargs,
            )
            print("Warning: Spacy.encode does not support arbitrary nlp pipelines so information "
                  "stored in the Doc object may be lost in encoding.")

    def _load_spacy(self,
                    lang: str = 'en_core_web_sm'):
        """
        Load the Spacy nlp pipeline.
        """
        return spacy.load(lang)

    def _add_neuralcoref(self):
        """
        Add the neuralcoref pipeline to Spacy.
        """
        if self.neuralcoref:
            try:
                import neuralcoref as nc
                nc.add_to_pipe(self.nlp)
            except ImportError:
                print("Can't import neuralcoref. Please install neuralcoref using:\n"
                      "git clone https://github.com/huggingface/neuralcoref.git\n"
                      "cd neuralcoref\n"
                      "pip install -r requirements.txt\n"
                      "pip install -e .")

    def __call__(self,
                 batch_or_dataset: BatchOrDataset,
                 columns: List[str],
                 batch_size: int = 8192) -> BatchOrDataset:
        return super().__call__(batch_or_dataset, columns, batch_size)

    @property
    def nlp(self):
        return self._nlp

    @classmethod
    def encode(cls, obj: Doc) -> str:

        # JSON dump the Doc
        doc_json = obj.to_json()

        if obj._.has('huggingface_neuralcoref'):
            # Create a helper function that turns a Span into a dictionary
            span_to_dict = lambda span: {'start': span.start, 'end': span.end, 'text': span.text}

            # Create a helper function that converts a Cluster (output of neuralcoref) into a dictionary
            cluster_to_dict = lambda cluster: {'i': cluster.i,
                                               'main': span_to_dict(cluster.main),
                                               'mentions': [span_to_dict(span) for span in cluster.mentions]}

            # Apply the helper functions to construct a dictionary for the neuralcoref information
            neuralcoref_dict = {'neuralcoref': [cluster_to_dict(cluster) for cluster in obj._.coref_clusters]}

            # Combine the neuralcoref dictionary with the doc_json
            doc_json = tz.merge(doc_json, neuralcoref_dict)

        # Convert the Spacy Doc to json before caching
        return json.dumps(doc_json)

    def single_column_apply(self,
                            column_batch: List,
                            *args,
                            **kwargs) -> List:
        # Apply Spacy's pipe method to process the examples
        return list(self.nlp.pipe(column_batch))

    @classmethod
    def tokens(cls,
               decoded_batch: List) -> List[List[str]]:
        """
        For each example, returns the list of tokens extracted by Spacy for each key.

        Spacy stores the span of each token under the "tokens" key.
        This function extracts the tokens from the text using the span of each token.
        """

        token_batch = []
        # Iterate over each decoded Doc dictionary
        for doc_dict in decoded_batch:
            tokens = []
            for token_dict in doc_dict['tokens']:
                tokens.append(doc_dict['text'][token_dict['start']:token_dict['end']])

            token_batch.append(tokens)

        return token_batch
