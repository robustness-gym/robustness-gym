import urllib.request
import zipfile

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import DownloadProgressBar
from robustnessgym.slicebuilders.subpopulations.phrase import HasAnyPhrase


class HasCategoryPhrase(HasAnyPhrase):
    def __init__(self):

        # Fetch wordlists
        self._fetch_sources()
        self.categories_to_words = self._load_all()

        super(HasCategoryPhrase, self).__init__(
            phrase_groups=[
                self.categories_to_words[supercategory][category]
                for (supercategory, category) in self.categories
            ],
            identifiers=[
                Identifier(
                    _name=self.__class__.__name__,
                    supercategory=supercategory,
                    category=category,
                )
                for (supercategory, category) in self.categories
            ],
        )

    @property
    def supercategories(self):
        return list(self.categories_to_words.keys())

    @property
    def categories(self):
        return sorted(
            [
                (supercategory, category)
                for supercategory in self.categories_to_words.keys()
                for category in self.categories_to_words[supercategory]
            ]
        )

    @classmethod
    def _fetch_sources(cls):
        if not (cls.logdir / "wordlists-master").exists():
            print("wordlists not found. Downloading..")
            urllib.request.urlretrieve(
                "https://github.com/imsky/wordlists/archive/master.zip",
                filename=str(cls.logdir / "wordlists.zip"),
                reporthook=DownloadProgressBar(),
            )

            with zipfile.ZipFile(str(cls.logdir / "wordlists.zip")) as zip_ref:
                zip_ref.extractall(str(cls.logdir))

    @classmethod
    def _load_all(cls):
        """Loads wordlists.

        Returns:
        """
        category_to_words = {
            supercategory: {} for supercategory in ["nouns", "verbs", "adjectives"]
        }
        for supercategory in ["nouns", "verbs", "adjectives"]:
            for path in (cls.logdir / "wordlists-master" / supercategory).glob("*"):
                with open(str(path)) as f:
                    category_to_words[supercategory][path.stem] = set(
                        f.read().splitlines()
                    )

        return category_to_words
