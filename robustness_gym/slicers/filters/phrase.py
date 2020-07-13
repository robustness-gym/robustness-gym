from __future__ import annotations
from robustness_gym.slicer import *
import numpy as np
from itertools import compress
from ahocorasick import Automaton


class AhoCorasickMixin:

    def __init__(self):
        # Use the Aho-Corasick search algorithm to speed up phrase lookups
        self.automaton = Automaton()

    def populate_automaton(self,
                           phrases: Dict[Any, str],
                           reset_automaton: bool = False):
        if reset_automaton:
            # Create a new automaton
            self.automaton = Automaton()

        # Add all the phrases we want to search for
        for key, phrase in phrases.items():
            # As values, we add the key of the phrase
            self.automaton.add_word(phrase, key)

        # Initialize Aho-Corasick
        self.automaton.make_automaton()


class SpacyMixin:

    @classmethod
    def get_tokens(cls, batch: Dict[str, List], keys: List[str]) -> List[Dict[str, List[str]]]:
        """
        For each example, returns the list of tokens extracted by spacy for each key.
        """
        return [
            {key: cls.tokens_from_spans(doc_json=cache['spacy'][key]) for key in keys}
            for cache in batch['cache']
        ]

    @classmethod
    def tokens_from_spans(cls, doc_json: Dict) -> List[str]:
        """
        Spacy stores the span of each token under the "tokens" key.

        Use this function to actually extract the tokens from the text using the span of each token.
        """
        tokens = []
        for token_dict in doc_json['tokens']:
            tokens.append(doc_json['text'][token_dict['start']:token_dict['end']])

        return tokens


class HasPhrase(Slicer,
                FilterMixin,
                AhoCorasickMixin,
                SpacyMixin):

    def __init__(self,
                 phrases=None):

        super(HasPhrase, self).__init__()

        # This is the list of phrases that will be searched
        self.phrases = phrases
        if self.phrases is None:
            self.phrases = []

        # Populate the Aho-Corasick automaton
        self.populate_automaton({i: phrase for i, phrase in enumerate(self.phrases)})

        # Set the headers
        self.headers = [f"{self.__class__.__name__}('{phrase}')" for phrase in self.phrases]

    @classmethod
    def from_file(cls, path: str):
        """
        Load phrases from a file, one per line.
        """
        with open(path) as f:
            phrases = [line.strip() for line in f.readlines()]
        return cls(phrases=phrases)

    @classmethod
    def default(cls):
        """
        A default vocabulary of phrases to search.
        """
        return cls(phrases=[])

    def alias(self) -> str:
        return self.__class__.__name__

    def slice_batch(self,
                    batch: Dict[str, List],
                    keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:

        # Use the spacy cache to grab the tokens in each example (for each key)
        tokenized_batch = self.get_tokens(batch, keys)

        # Construct the matrix of slice labels: (batch_size x n_phrases)
        slice_labels = np.zeros((len(tokenized_batch), len(self.phrases)),
                                dtype=np.int32)

        for i, example in enumerate(tokenized_batch):
            for key, tokens in example.items():
                # Get the values (indices) of all the matched tokens
                matched_indices = [self.automaton.get(token) for token in tokens if self.automaton.exists(token)]

                # Fill in the slice labels for slices that are present
                slice_labels[i, matched_indices] = 1

        # Store these slice labels
        batch = self.store_slice_labels(batch, slice_labels.tolist(), self.alias())

        return batch, self.slice_batch_with_slice_labels(batch, slice_labels), slice_labels


class HasAnyPhrase(HasPhrase):

    def __init__(self,
                 phrases=None,
                 alias=None):
        super(HasAnyPhrase, self).__init__(phrases=phrases)

        # Set the headers
        self.headers = [f"{self.__class__.__name__}({set(self.phrases) if alias is None else alias})"]

    def alias(self) -> str:
        return self.__class__.__name__

    def slice_batch(self,
                    batch: Dict[str, List],
                    keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:
        # Run the phrase matcher
        batch, _, slice_labels = super(HasAnyPhrase, self).slice_batch(batch=batch, keys=keys)

        # Check if any of the slice labels is 1
        slice_labels = np.any(slice_labels, axis=1)[:, np.newaxis]

        # Store these slice labels
        batch = self.store_slice_labels(batch, slice_labels.tolist(), self.alias())

        return batch, self.slice_batch_with_slice_labels(batch, slice_labels), slice_labels


class HasAllPhrases(HasPhrase):

    def __init__(self,
                 phrases=None,
                 alias=None):
        super(HasAllPhrases, self).__init__(phrases=phrases)

        # Set the headers
        self.headers = [f"{self.__class__.__name__}({set(self.phrases) if alias is None else alias})"]

    def alias(self) -> str:
        return self.__class__.__name__

    def slice_batch(self,
                    batch: Dict[str, List],
                    keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:
        # Run the phrase matcher
        batch, _, slice_labels = super(HasAllPhrases, self).slice_batch(batch=batch, keys=keys)

        # Check if all of the slice labels are 1
        slice_labels = np.all(slice_labels, axis=1)[:, np.newaxis]

        # Store these slice labels
        batch = self.store_slice_labels(batch, slice_labels.tolist(), self.alias())

        return batch, self.slice_batch_with_slice_labels(batch, slice_labels), slice_labels


# Taken from https://github.com/tommccoy1/hans/blob/master/templates.py
nouns_sg = ["professor", "student", "president", "judge", "senator", "secretary", "doctor", "lawyer", "scientist",
            "banker", "tourist", "manager", "artist", "author", "actor", "athlete"]
nouns_pl = ["professors", "students", "presidents", "judges", "senators", "secretaries", "doctors", "lawyers",
            "scientists", "bankers", "tourists", "managers", "artists", "authors", "actors", "athletes"]

transitive_verbs = ["recommended", "called", "helped", "supported", "contacted", "believed", "avoided", "advised",
                    "saw", "stopped", "introduced", "mentioned", "encouraged", "thanked", "recognized", "admired"]
passive_verbs = ["recommended", "helped", "supported", "contacted", "believed", "avoided", "advised", "stopped",
                 "introduced", "mentioned", "encouraged", "thanked", "recognized", "admired"]
intransitive_verbs = ["slept", "danced", "ran", "shouted", "resigned", "waited", "arrived", "performed"]

nps_verbs = ["believed", "knew", "heard", "forgot", "preferred", "claimed", "wanted", "needed",
             "found", "suggested", "expected"]
npz_verbs = ["hid", "moved", "presented", "paid", "studied", "stopped"]
npz_verbs_plural = ["fought", "paid", "changed", "studied", "answered", "stopped", "grew", "moved", "returned",
                    "left", "improved", "lost", "visited", "ate", "played"]
understood_argument_verbs = ["paid", "explored", "won", "wrote", "left", "read", "ate"]
nonentailing_quot_vebs = ["hoped", "claimed", "thought", "believed", "said", "assumed"]
question_embedding_verbs = ["wondered", "understood", "knew", "asked", "explained", "realized"]

preps = ["near", "behind", "by", "in front of", "next to"]
conjs = ["while", "after", "before", "when", "although", "because", "since"]
past_participles = ["studied", "paid", "helped", "investigated", "presented"]
called_objects = ["coward", "liar", "hero", "fool"]
told_objects = ["story", "lie", "truth", "secret"]
food_words = ["fruit", "salad", "broccoli", "sandwich", "rice", "corn", "ice cream"]
location_nouns = ["neighborhood", "region", "country", "town", "valley", "forest", "garden", "museum", "desert",
                  "island", "town"]
location_nouns_b = ["museum", "school", "library", "office", "laboratory"]
won_objects = ["race", "contest", "war", "prize", "competition", "election", "battle", "award", "tournament"]
read_wrote_objects = ["book", "column", "report", "poem", "letter", "novel", "story", "play", "speech"]
adjs = ["important", "popular", "famous", "young", "happy", "helpful", "serious",
        "angry"]
adj_comp_nonent = ["afraid", "sure", "certain"]
adj_comp_ent = ["sorry", "aware", "glad"]
advs = ["quickly", "slowly", "happily", "easily", "quietly", "thoughtfully"]
const_adv = ["after", "before", "because", "although", "though", "since", "while"]
const_quot_entailed = ["forgot", "learned", "remembered", "knew"]
advs_nonentailed = ["supposedly", "probably", "maybe", "hopefully"]
advs_entailed = ["certainly", "definitely", "clearly", "obviously", "suddenly"]
rels = ["who", "that"]
quest = ["why", "how"]
nonent_complement_nouns = ["feeling", "evidence", "idea", "belief"]
ent_complement_nouns = ["fact", "reason", "news", "time"]



# if __name__ == '__main__':
#
#     # sf_slicers = [....]
#     # augmentations = [....]
#     # attacks = [....]
#
#     # apply_attacks(dataset, attacks)


