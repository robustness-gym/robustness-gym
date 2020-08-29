from typing import *

import cytoolz as tz
import pytorch_lightning.metrics.functional as lightning_metrics
import torch
from transformers import *

from robustness_gym import *


class Model:

    def __init__(self,
                 identifier: str,
                 task: Task,
                 evaluation_fn=None):
        # TODO(karan): improve this wrapper around models
        self.identifier = identifier
        self.task = task

        if evaluation_fn is not None:
            self.evaluate = evaluation_fn

        self.outputs = {
            'probs',
            'logits',
            'pred',
            # 'embeddings',
            # TODO(karan): other information from the model e.g. embeddings which aren't task related?
        }

    def __call__(self,
                 dataset: Dataset,
                 input_keys: List[str],
                 output_keys: List[str],
                 batch_size: int = 32,
                 coerce_fn: Callable = None,
                 *args,
                 **kwargs):

        return self.evaluate(dataset,
                             input_keys,
                             output_keys,
                             batch_size,
                             coerce_fn,
                             *args,
                             **kwargs)

    @classmethod
    def huggingface(cls,
                    identifier: str,
                    task: Task,
                    model: Optional[AutoModel] = None,
                    tokenizer: Optional[AutoTokenizer] = None):
        """

        Args:
            identifier:
            task:
            model:
            tokenizer:

        Returns:

        Examples:
            >>> Model.huggingface(identifier='', task=TernaryNaturalLanguageInference())
            >>> Model.huggingface(identifier='', model=AutoModelForSequenceClassification.from_pretrained(''), tokenizer=AutoTokenizer.from_pretrained(''))

        """

        return HuggingfaceModel(identifier=identifier, task=task, model=model, tokenizer=tokenizer)

    def forward(self, input_batch: Dict) -> Dict:
        raise NotImplementedError

    def evaluate(self,
                 dataset: Dataset,
                 input_keys: List[str],
                 output_keys: List[str],
                 batch_size: int = 32,
                 coerce_fn: Callable = None):
        raise NotImplementedError

    @staticmethod
    def remap_labels(output_dict: Dict, label_map: List[int]) -> Dict:
        """
        Map the output labels of the model.

        Example: 3-way classificaiton, with label_map = [1, 2, 0]
        => (model label 0 -> dataset label 1, model label 1 -> dataset label 2, ...).
        """

        # Check the number of classes
        num_classes = len(label_map)

        # Remap the columns of all outputs that have # columns = num_classes
        for key in output_dict:
            if output_dict[key].shape[-1] == num_classes:
                output_dict[key] = output_dict[key][..., label_map]

        # Remap the pred key
        inverse_label_map = [t[1] for t in sorted([(label, i) for i, label in enumerate(label_map)])]
        output_dict['pred'] = torch.tensor(inverse_label_map)[output_dict['pred']]

        return output_dict


class HuggingfaceModel(Model):

    def __init__(self,
                 identifier: str,
                 task: Task,
                 model: Optional[AutoModel] = None,
                 tokenizer: Optional[AutoTokenizer] = None):

        super(HuggingfaceModel, self).__init__(
            identifier=identifier,
            task=task,
        )

        self.tokenizer = tokenizer
        if tokenizer is None:
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.identifier)

        self.model = model
        if model is None:
            # Load the model
            if task.classification():
                self.model = AutoModelForSequenceClassification.from_pretrained(self.identifier)
            else:
                raise NotImplementedError

    def forward(self, input_batch: Dict) -> Dict:
        # Create the required outputs
        output_dict = {k: None for k in self.outputs}

        # Run the model on the input_batch
        # TODO(karan): allow outputs to generically contain side information (embeddings, attention, etc.)
        outputs = self.model(**input_batch)

        # The logits are at the 0th index
        logits = outputs[0]

        # TODO(karan): these are still on GPU, do metric computation on GPU then move to CPU
        # TODO(karan): incrementally compute metrics?
        if 'logits' in self.outputs:
            output_dict['logits'] = logits

        if 'probs' in self.outputs:
            output_dict['probs'] = torch.nn.functional.softmax(logits, dim=-1)

        if 'pred' in self.outputs:
            output_dict['pred'] = logits.argmax(dim=-1)

        return output_dict

    def encode_batch(self,
                     batch: Dict[str, List],
                     keys: Collection[str],
                     **kwargs):
        # TODO(karan): Automatically writing this encoder for a variety of tasks
        return self.tokenizer(*[batch[key] for key in keys], truncation=True, padding=True, **kwargs)

    def predict_batch(self,
                      batch: Dict[str, List],
                      input_keys: Collection[str]):

        # Tokenize the batch
        input_batch = self.encode_batch(batch=batch,
                                        keys=input_keys)

        # Convert the batch to torch.Tensor
        input_batch = tz.valmap(lambda v: torch.tensor(v), input_batch)

        # Apply the model to the batch
        return self.forward(input_batch)

    def evaluate(self,
                 dataset: Dataset,
                 input_keys: List[str],
                 output_keys: List[str],
                 batch_size: int = 32,
                 coerce_fn: Callable = None):

        # TODO(karan): generalize to TF2

        # Reset the dataset format
        dataset.reset_format()

        # TODO(karan): check that the Dataset conforms to the task definition
        # TODO(karan): figure out how the output_keys will be used by the metrics
        pass

        predictions = []
        targets = []

        # Loop and apply the prediction function
        # TODO(karan): not using .map() here in order to get more fine-grained control over devices
        for idx in range(0, len(dataset), batch_size):
            # Create the batch
            batch = dataset[idx: idx + batch_size]

            # Predict on the batch
            prediction_dict = self.predict_batch(batch=batch,
                                                 input_keys=input_keys)

            # Coerce the predictions
            if coerce_fn:
                prediction_dict = coerce_fn(prediction_dict)

            # Grab the raw target key/values
            target_dict = tz.keyfilter(lambda k: k in output_keys, batch)

            # TODO(karan): general version for non-classification problems
            # TODO(karan): move this to the right device
            target_dict = tz.valmap(lambda v: torch.tensor(v), target_dict)

            # TODO(karan): incremental metric computation here
            # Append the predictions and targets
            predictions.append(prediction_dict)
            targets.append(target_dict)

        # Consolidate the predictions and targets
        # TODO(karan): Need to store predictions and outputs from the model
        predictions = tz.merge_with(lambda v: torch.cat(v), *predictions)
        targets = tz.merge_with(lambda v: torch.cat(v), *targets)

        # Compute the metrics
        # TODO(karan): generalize this code to support metric computation for any task

        # Assumes classification, so the output_keys contains a single key for the label
        assert len(output_keys) == 1, "Only supports classification."
        labels = targets[list(targets.keys())[0]]

        # TODO(karan): make this easier
        # TODO(karan): move to the right device
        print(predictions['pred'])
        print(labels)
        evaluation_dict = {}
        for metric in self.task.metrics:
            if metric == 'accuracy':
                # Calculate the accuracy
                evaluation_dict[metric] = lightning_metrics.accuracy(predictions['pred'], labels)
            elif metric == 'f1':
                # Calculate the f1
                evaluation_dict[metric] = lightning_metrics.f1_score(predictions['pred'], labels)

        return evaluation_dict
