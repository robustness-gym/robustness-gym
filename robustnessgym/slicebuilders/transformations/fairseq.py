from typing import List

import cytoolz as tz
import torch

from robustnessgym.core.identifier import Identifier
from robustnessgym.slicebuilders.transformation import SingleColumnTransformation

try:
    import fastBPE  # noqa
except ImportError:
    _fastbpe_available = False
else:
    _fastbpe_available = True


# TODO(karan): spec requirements (fastBPE)
class FairseqBacktranslation(SingleColumnTransformation):
    """Class for performing backtranslation using torchhub fairseq models."""

    def __init__(
        self,
        n_src2tgt: int = 1,
        n_tgt2src: int = 1,
        langs: str = "en2de",
        torchhub_dir: str = None,
        device: str = "cuda",
        src2tgt_topk: int = 1000,
        src2tgt_temp: float = 1.0,
        tgt2src_topk: int = 1000,
        tgt2src_temp: float = 1.0,
    ):

        if not _fastbpe_available:
            raise ImportError(
                "fastBPE not available for import. Please install fastBPE with pip "
                "install fastBPE."
            )

        super(FairseqBacktranslation, self).__init__(
            identifiers=Identifier.range(
                n=n_src2tgt * n_tgt2src,
                _name=self.__class__.__name__,
                langs=langs,
                src2tgt_topk=src2tgt_topk,
                src2tgt_temp=src2tgt_temp,
                tgt2src_topk=tgt2src_topk,
                tgt2src_temp=tgt2src_temp,
            )
        )

        # Set the parameters
        self.n_src2tgt = n_src2tgt
        self.n_tgt2src = n_tgt2src
        self.src2tgt_topk = src2tgt_topk
        self.src2tgt_temp = src2tgt_temp
        self.tgt2src_topk = tgt2src_topk
        self.tgt2src_temp = tgt2src_temp

        # Setup the backtranslation models
        self.src2tgt, self.tgt2src = self.load_models(
            langs=langs,
            torchhub_dir=torchhub_dir,
            # self.logdir if not torchhub_dir else torchhub_dir,
            device=device,
        )

    @staticmethod
    def load_models(
        langs: str,
        torchhub_dir: str = None,
        device: str = "cuda",
        half_precision: bool = False,
    ):
        if torchhub_dir:
            # Set the directory where the models will be stored.
            torch.hub.set_dir(torchhub_dir)

        if langs == "en2de":
            # Round-trip translations between English and German
            src2tgt = torch.hub.load(
                "pytorch/fairseq",
                "transformer.wmt19.en-de.single_model",
                tokenizer="moses",
                bpe="fastbpe",
            )

            tgt2src = torch.hub.load(
                "pytorch/fairseq",
                "transformer.wmt19.de-en.single_model",
                tokenizer="moses",
                bpe="fastbpe",
            )

        elif langs == "en2ru":
            # Round-trip translations between English and Russian
            src2tgt = torch.hub.load(
                "pytorch/fairseq",
                "transformer.wmt19.en-ru.single_model",
                tokenizer="moses",
                bpe="fastbpe",
            )

            tgt2src = torch.hub.load(
                "pytorch/fairseq",
                "transformer.wmt19.ru-en.single_model",
                tokenizer="moses",
                bpe="fastbpe",
            )
        else:
            raise NotImplementedError

        # Convert to half precision
        if half_precision:
            return src2tgt.to(device).half(), tgt2src.to(device).half()
        return src2tgt.to(device), tgt2src.to(device)

    def single_column_apply(self, column_batch: List) -> List[List]:
        """Perform backtranslation using the fairseq pretrained translation
        models."""
        # Encode the source sentences
        src_sentences = column_batch
        src_sentences_bin = [self.src2tgt.encode(e)[:1024] for e in src_sentences]

        # Translate it
        tgt_sentences = self.src2tgt.generate(
            src_sentences_bin,
            beam=self.n_src2tgt,
            sampling=True,
            sampling_topk=self.src2tgt_topk,
            temperature=self.src2tgt_temp,
            skip_invalid_size_inputs=True,
        )

        # Back-translate: moving tokens to CPU because of an error otherwise
        src_paraphrases = self.tgt2src.generate(
            [e["tokens"].cpu() for l in tgt_sentences for e in l],
            beam=self.n_tgt2src,
            sampling=True,
            sampling_topk=self.tgt2src_topk,
            temperature=self.tgt2src_temp,
            skip_invalid_size_inputs=True,
        )

        # Flatten out all the translations into one giant list
        flat_src_paraphrases = list(
            tz.concat(
                map(
                    lambda l: list(map(lambda e: self.tgt2src.decode(e["tokens"]), l)),
                    src_paraphrases,
                )
            )
        )

        # Partition so that we get n_src2tgt * n_tgt2src paraphrases per input sentence
        return list(
            tz.partition_all(
                len(flat_src_paraphrases) // len(src_sentences), flat_src_paraphrases
            )
        )
