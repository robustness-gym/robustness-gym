import itertools
import statistics
from typing import *

import cytoolz as tz
import pytorch_lightning.metrics.functional as lightning_metrics
import torch
from transformers import *
from rouge_score import rouge_scorer

from robustnessgym.tasks.task import Task
from robustnessgym.dataset import Dataset


class Model:

    def __init__(self,
                 identifier: str,
                 task: Task,
                 model=None,
                 evaluation_fn=None,
                 device: str = None):

        # TODO(karan): improve this wrapper around models
        # TODO(karan): add some human-readble identifier to this as optional
        self.identifier = identifier
        self.task = task
        self.model = model

        if evaluation_fn is not None:
            self.evaluate = evaluation_fn

        if self.task.classification():
            self.outputs = {
                'probs',
                'logits',
                'pred',
                # 'embeddings',
                # TODO(karan): other information from the model e.g. embeddings which aren't task related?
            }
        else:
            self.outputs = {
                'pred',
                # 'embeddings',
                # TODO(karan): other information from the model e.g. embeddings which aren't task related?
            }


        if not device:
            self.device = 'cpu'
            if torch.cuda.is_available():
                self.device = 'cuda:0'

    def to(self, device: str):
        self.device = device
        return self.model.to(device)

    def __call__(self,
                 dataset: Dataset,
                 input_columns: List[str],
                 output_columns: List[str],
                 batch_size: int = 32,
                 coerce_fn: Callable = None,
                 *args,
                 **kwargs):

        return self.evaluate(dataset,
                             input_columns,
                             output_columns,
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
                 input_columns: List[str],
                 output_columns: List[str],
                 batch_size: int = 32,
                 coerce_fn: Callable = None):
        raise NotImplementedError

    @staticmethod
    def remap_labels(output_dict: Dict,
                     label_map: List[int]) -> Dict:
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
                 tokenizer: Optional[AutoTokenizer] = None,
                 device: str = None):

        super(HuggingfaceModel, self).__init__(
            identifier=identifier,
            task=task,
            device=device,
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
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.identifier)

        self.task = task

        # Move the model to device
        self.to(self.device)

    def forward(self, input_batch: Dict) -> Dict:
        # Create the required outputs
        output_dict = {k: None for k in self.outputs}

        if self.task.classification():
            # Run the model on the input_batch
            # TODO(karan): allow outputs to generically contain side information (embeddings, attention, etc.)
            with torch.no_grad():
                outputs = self.model(**input_batch)

            # The logits are at the 0th index
            logits = outputs[0]

            # TODO(karan): these are still on GPU, do metric computation on GPU then move to CPU
            # TODO(karan): incrementally compute metrics?
            if 'logits' in self.outputs:
                output_dict['logits'] = logits.to('cpu')

            if 'probs' in self.outputs:
                output_dict['probs'] = torch.nn.functional.softmax(logits, dim=-1).to('cpu')

            if 'pred' in self.outputs:
                output_dict['pred'] = logits.argmax(dim=-1).to('cpu')
        else:
            with torch.no_grad():
                summary_token_ids = self.model.generate(**input_batch)
                summaries = [self.tokenizer.decode(token_id_list, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False) for token_id_list in
                       summary_token_ids]
                output_dict['pred'] = summaries

        return output_dict

    def encode_batch(self,
                     batch: Dict[str, List],
                     columns: Collection[str],
                     **kwargs):
        # TODO(karan): Automatically writing this encoder for a variety of tasks
        return self.tokenizer(*[batch[key] for key in columns],
                              truncation=True,
                              padding=True,
                              **kwargs)

    def predict_batch(self,
                      batch: Dict[str, List],
                      input_columns: Collection[str]):

        # Tokenize the batch
        input_batch = self.encode_batch(batch=batch,
                                        columns=input_columns)

        # Convert the batch to torch.Tensor
        input_batch = tz.valmap(lambda v: torch.tensor(v).to(device=self.device), input_batch)

        # Apply the model to the batch
        return self.forward(input_batch)

    def evaluate(self,
                 dataset: Dataset,
                 input_columns: List[str],
                 output_columns: List[str],
                 batch_size: int = 32,
                 coerce_fn: Callable = None):

        # TODO(karan): generalize to TF2

        # Reset the dataset format
        dataset.reset_format()
        dataset.set_format(columns=input_columns + output_columns)

        # TODO(karan): check that the Dataset conforms to the task definition
        # TODO(karan): figure out how the output_columns will be used by the metrics
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
                                                 input_columns=input_columns)

            # Coerce the predictions
            if coerce_fn:
                prediction_dict = coerce_fn(prediction_dict)

            # Grab the raw target key/values
            target_dict = tz.keyfilter(lambda k: k in output_columns, batch)

            # TODO(karan): general version for non-classification problems
            # TODO(karan): move this to the right device
            if self.task.classification():
                target_dict = tz.valmap(lambda v: torch.tensor(v), target_dict)

            # TODO(karan): incremental metric computation here
            # Append the predictions and targets
            predictions.append(prediction_dict)
            targets.append(target_dict)

        # Consolidate the predictions and targets
        if self.task.classification():
            # TODO(karan): Need to store predictions and outputs from the model
            predictions = tz.merge_with(lambda v: torch.cat(v).to('cpu'), *predictions)
            targets = tz.merge_with(lambda v: torch.cat(v).to('cpu'), *targets)
        else:
            predictions = tz.merge_with(lambda x: list(itertools.chain.from_iterable(x)), *predictions)
            targets = tz.merge_with(lambda x: list(itertools.chain.from_iterable(x)), *targets)


        # Compute the metrics
        # TODO(karan): generalize this code to support metric computation for any task

        # Assumes classification, so the output_columns contains a single key for the label
        if self.task.classification():
            assert len(output_columns) == 1#, "Only supports classification."
            num_classes = self.task.output_schema.features[list(self.task.output_schema.keys())[0]].num_classes

        labels = targets[list(targets.keys())[0]]

        # TODO(karan): make this easier
        # TODO(karan): move to the right device
        evaluation_dict = {}
        for metric in self.task.metrics:
            if metric == 'accuracy':
                # Calculate the accuracy
                evaluation_dict[metric] = lightning_metrics.accuracy(
                    pred=predictions['pred'].to(self.device),
                    target=labels.to(self.device),
                    num_classes=num_classes
                ).item()

            elif metric == 'f1':
                # Calculate the f1
                evaluation_dict[metric] = lightning_metrics.f1_score(
                    pred=predictions['pred'].to(self.device),
                    target=labels.to(self.device),
                    num_classes=num_classes
                ).item()

            elif metric in ('rouge1', 'rouge2', 'rougeLsum'):
                scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
                evaluation_dict[metric] = statistics.mean(scorer.score(reference, pred)[metric].fmeasure for \
                                                          reference, pred in zip(labels, predictions['pred']))

            elif metric == 'class_dist':
                # Calculate class distribution
                evaluation_dict[metric] = lightning_metrics.to_onehot(
                    tensor=labels,
                    num_classes=num_classes
                ).double().mean(dim=0).tolist()

            elif metric == 'pred_dist':
                # Calculate predicted class distribution
                evaluation_dict[metric] = lightning_metrics.to_onehot(
                    tensor=predictions['pred'],
                    num_classes=num_classes
                ).double().mean(dim=0).tolist()

            else:
                raise NotImplementedError


        # Reset the data format
        dataset.reset_format()

        return evaluation_dict
