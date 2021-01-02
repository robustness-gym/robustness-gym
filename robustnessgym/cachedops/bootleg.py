import tarfile
import urllib.request
from typing import Dict, List

from torch import cuda

from robustnessgym.cachedops.textblob import TextBlob
from robustnessgym.core.cachedops import SingleColumnCachedOperation
from robustnessgym.core.decorators import singlecolumn
from robustnessgym.core.tools import DownloadProgressBar

try:
    from bootleg.annotator import Annotator
    from bootleg.utils.parser_utils import get_full_config
except ImportError:
    _bootleg_available = False
else:
    _bootleg_available = True


class Bootleg(SingleColumnCachedOperation):
    def __init__(self, threshold: float = 0.3, device: str = None, *args, **kwargs):

        if not _bootleg_available:
            # TODO(karan): add instructions to install bootleg
            raise ImportError(
                "Bootleg not available for import. Please install Bootleg."
            )

        super(Bootleg, self).__init__(threshold=threshold, *args, **kwargs)

        # Set the device
        if not device:
            device = "cuda" if cuda.is_available() else "cpu"

        # Fetch sources for Bootleg
        self._fetch_sources()

        # Create the annotator
        self.annotator = Annotator(
            config_args=self._create_config(),
            device=device,
            cand_map=self.logdir / "entity_db/entity_mappings/alias2qids_wiki.json",
        )
        self.annotator.set_threshold(threshold)

    @classmethod
    def _fetch_sources(cls):
        if not (cls.logdir / "bootleg_wiki").exists():
            print("bootleg_wiki not found. Downloading..")
            urllib.request.urlretrieve(
                "https://bootleg-emb.s3.amazonaws.com/models/2020_08_25/bootleg_wiki"
                ".tar.gz",
                filename=str(cls.logdir / "bootleg_wiki.tar.gz"),
                reporthook=DownloadProgressBar(),
            )

            tar = tarfile.open(str(cls.logdir / "bootleg_wiki.tar.gz"), "r:gz")
            tar.extractall()
            tar.close()

        if not (cls.logdir / "emb_data").exists():
            print("emb_data not found. Downloading..")
            urllib.request.urlretrieve(
                "https://bootleg-emb.s3.amazonaws.com/emb_data.tar.gz",
                filename=str(cls.logdir / "emb_data.tar.gz"),
                reporthook=DownloadProgressBar(),
            )

            tar = tarfile.open(str(cls.logdir / "emb_data.tar.gz"), "r:gz")
            tar.extractall()
            tar.close()

        if not (cls.logdir / "entity_db").exists():
            print("entity_db not found. Downloading..")
            urllib.request.urlretrieve(
                "https://bootleg-emb.s3.amazonaws.com/entity_db.tar.gz",
                filename=str(cls.logdir / "entity_db.tar.gz"),
                reporthook=DownloadProgressBar(),
            )

            tar = tarfile.open(str(cls.logdir / "entity_db.tar.gz"), "r:gz")
            tar.extractall()
            tar.close()

    @classmethod
    def _create_config(cls):
        # load a config for Bootleg
        config_args = get_full_config(cls.logdir / "bootleg_wiki/bootleg_config.json")

        # set the model checkpoint path
        config_args.run_config.init_checkpoint = (
            cls.logdir / "bootleg_wiki/bootleg_model.pt"
        )

        # set the path for the entity db and candidate map
        config_args.data_config.entity_dir = cls.logdir / "entity_db"
        config_args.data_config.alias_cand_map = "alias2qids_wiki.json"

        # set the embedding paths
        config_args.data_config.emb_dir = cls.logdir / "emb_data"
        config_args.data_config.word_embedding.cache_dir = cls.logdir / "emb_data"

        return config_args

    @singlecolumn
    def apply(self, batch: Dict[str, List], columns: List[str], **kwargs) -> List:

        # Use TextBlob to split the column into sentences
        blobs = TextBlob.retrieve(batch=batch, columns=columns)[columns[0]]

        # Annotate each example
        return [
            [
                (
                    self.annotator.extract_mentions(str(text)),
                    self.annotator.label_mentions(str(text)),
                )
                for text in blob.sentences
            ]
            for blob in blobs
        ]
