from __future__ import annotations
from robustness_gym.slicemaker import *
from robustness_gym.dataset import Spacy
import numpy as np
from itertools import compress
from ahocorasick import Automaton


class AhoCorasickMixin:

    def __init__(self):
        # Use the Aho-Corasick search algorithm to speed up phrase lookups
        self.automaton = Automaton()

    def populate_automaton(self,
                           phrases: Dict[Any, str],
                           reset_automaton: bool = False) -> None:
        if reset_automaton:
            # Create a new automaton
            self.automaton = Automaton()

        # Add all the phrases we want to search for
        for key, phrase in phrases.items():
            # As values, we add the key of the phrase
            self.automaton.add_word(phrase, key)

        # Initialize Aho-Corasick
        self.automaton.make_automaton()


class HasPhrase(Subpopulation,
                AhoCorasickMixin,
                Spacy):

    def __init__(self,
                 phrases=None,
                 *args,
                 **kwargs):

        super(HasPhrase, self).__init__(
            # One slice per phrase
            num_slices=len(self.phrases),
            identifiers=[
                f"{self.__class__.__name__}('{phrase}')" for phrase in phrases
            ]
        )

        # This is the list of phrases that will be searched
        self.phrases = phrases
        if self.phrases is None:
            self.phrases = []

        # Populate the Aho-Corasick automaton
        self.populate_automaton(
            {i: phrase for i, phrase in enumerate(self.phrases)})

    @classmethod
    def from_file(cls, path: str) -> SliceMaker:
        """
        Load phrases from a file, one per line.
        """
        with open(path) as f:
            phrases = [line.strip() for line in f.readlines()]
        return cls(phrases=phrases)

    @classmethod
    def default(cls) -> SliceMaker:
        """
        A default vocabulary of phrases to search.
        """
        return cls(phrases=[])

    @classmethod
    def join(cls, *slicers: HasPhrase) -> Sequence[HasPhrase]:
        """
        Join to combine multiple HasPhrase slicers.
        """
        return [HasPhrase(phrases=list(tz.concat([slicer.phrases for slicer in slicers])))]

    def apply(self,
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> np.ndarray:

        # Use the spacy cache to grab the tokens in each example (for each key)
        tokenized_batch = self.get_tokens(batch, keys)

        for i, example in enumerate(tokenized_batch):
            for key, tokens in example.items():
                # Get the values (indices) of all the matched tokens
                matched_indices = [
                    self.automaton.get(token) for token in tokens if self.automaton.exists(token)
                ]

                # Fill in the slice labels for slices that are present
                slice_membership[i, matched_indices] = 1

        return slice_membership


# HasAnyPhrase = FilterMixin.union()

class HasAnyPhrase(HasPhrase):

    def __init__(self,
                 phrases=None,
                 alias=None):
        super(HasAnyPhrase, self).__init__(phrases=phrases)

        # Set the headers
        self.headers = [
            f"{self.__class__.__name__}({set(self.phrases) if alias is None else alias})"]

    def alias(self) -> str:
        return self.__class__.__name__

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:
        # Run the phrase matcher
        batch, _, slice_labels = super(
            HasAnyPhrase, self).process_batch(batch=batch, keys=keys)

        # Check if any of the slice labels is 1
        slice_labels = np.any(slice_labels, axis=1)[:, np.newaxis]

        # Store these slice labels
        batch = self.store_slice_labels(
            batch, slice_labels.tolist(), self.alias())

        return batch, self.filter_batch_by_slice_membership(batch, slice_labels), slice_labels


class HasAllPhrases(HasPhrase):

    def __init__(self,
                 phrases=None,
                 alias=None):
        super(HasAllPhrases, self).__init__(phrases=phrases)

        # Set the headers
        self.headers = [
            f"{self.__class__.__name__}({set(self.phrases) if alias is None else alias})"]

    def alias(self) -> str:
        return self.__class__.__name__

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:
        # Run the phrase matcher
        batch, _, slice_labels = super(
            HasAllPhrases, self).process_batch(batch=batch, keys=keys)

        # Check if all of the slice labels are 1
        slice_labels = np.all(slice_labels, axis=1)[:, np.newaxis]

        # Store these slice labels
        batch = self.store_slice_labels(
            batch, slice_labels.tolist(), self.alias())

        return batch, self.filter_batch_by_slice_membership(batch, slice_labels), slice_labels


# class HANSPhrases(HasPhrase):


# Taken from https://github.com/tommccoy1/hans/blob/master/templates.py
class SingularNouns(HasPhrase):

    def __init__(self):
        super(SingularNouns, self).__init__(phrases=[
            "professor", "student", "president", "judge", "senator", "secretary", "doctor", "lawyer", "scientist",
            "banker", "tourist", "manager", "artist", "author", "actor", "athlete",
        ])


class PluralNouns(HasPhrase):

    def __init__(self):
        super(PluralNouns, self).__init__(phrases=[
            "professors", "students", "presidents", "judges", "senators", "secretaries", "doctors", "lawyers",
            "scientists", "bankers", "tourists", "managers", "artists", "authors", "actors", "athletes",
        ])


class TransitiveVerbs(HasPhrase):

    def __init__(self):
        super(TransitiveVerbs, self).__init__(phrases=[
            "recommended", "called", "helped", "supported", "contacted", "believed", "avoided", "advised",
            "saw", "stopped", "introduced", "mentioned", "encouraged", "thanked", "recognized", "admired"
        ])


class PassiveVerbs(HasPhrase):

    def __init__(self):
        super(PassiveVerbs, self).__init__(phrases=[
            "recommended", "helped", "supported", "contacted", "believed", "avoided", "advised", "stopped",
            "introduced", "mentioned", "encouraged", "thanked", "recognized", "admired"
        ])


class IntransitiveVerbs(HasPhrase):

    def __init__(self):
        super(IntransitiveVerbs, self).__init__(phrases=[
            "slept", "danced", "ran", "shouted", "resigned", "waited", "arrived", "performed"
        ])


class NPSVerbs(HasPhrase):

    def __init__(self):
        super(NPSVerbs, self).__init__(phrases=[
            "believed", "knew", "heard", "forgot", "preferred", "claimed", "wanted", "needed",
            "found", "suggested", "expected"
        ])


class NPZVerbs(HasPhrase):

    def __init__(self):
        super(NPZVerbs, self).__init__(phrases=[
            "hid", "moved", "presented", "paid", "studied", "stopped"
        ])


class PluralNPZVerbs(HasPhrase):

    def __init__(self):
        super(PluralNPZVerbs, self).__init__(phrases=[
            "fought", "paid", "changed", "studied", "answered", "stopped", "grew", "moved", "returned",
            "left", "improved", "lost", "visited", "ate", "played"
        ])


class Prepositions(HasPhrase):

    def __init__(self):
        super(Prepositions, self).__init__(phrases=[
            "near", "behind", "by", "in front of", "next to"
        ])


class Conjs(HasPhrase):

    def __init__(self):
        super(Conjs, self).__init__(phrases=[
            "while", "after", "before", "when", "although", "because", "since"
        ])


class PastParticiples(HasPhrase):

    def __init__(self):
        super(PastParticiples, self).__init__(phrases=[
            "studied", "paid", "helped", "investigated", "presented"
        ])


class UnderstoodArgumentVerbs(HasPhrase):

    def __init__(self):
        super(UnderstoodArgumentVerbs, self).__init__(phrases=[
            "paid", "explored", "won", "wrote", "left", "read", "ate"
        ])


class NonEntQuotVerbs(HasPhrase):

    def __init__(self):
        super(NonEntQuotVerbs, self).__init__(phrases=[
            "hoped", "claimed", "thought", "believed", "said", "assumed"
        ])


class QuestionEmbeddingVerbs(HasPhrase):

    def __init__(self):
        super(QuestionEmbeddingVerbs, self).__init__(phrases=[
            "wondered", "understood", "knew", "asked", "explained", "realized"
        ])


class CalledObjects(HasPhrase):

    def __init__(self):
        super(CalledObjects, self).__init__(phrases=[
            "coward", "liar", "hero", "fool"
        ])


class ToldObjects(HasPhrase):

    def __init__(self):
        super(ToldObjects, self).__init__(phrases=[
            "story", "lie", "truth", "secret"
        ])


class FoodWords(HasPhrase):

    def __init__(self):
        super(FoodWords, self).__init__(phrases=[
            "fruit", "salad", "broccoli", "sandwich", "rice", "corn", "ice cream"
        ])


class LocationNounsA(HasPhrase):

    def __init__(self):
        super(LocationNounsA, self).__init__(phrases=[
            "neighborhood", "region", "country", "town", "valley", "forest", "garden", "museum", "desert",
            "island", "town"
        ])


class LocationNounsB(HasPhrase):

    def __init__(self):
        super(LocationNounsB, self).__init__(phrases=[
            "museum", "school", "library", "office", "laboratory"
        ])


class WonObjects(HasPhrase):

    def __init__(self):
        super(WonObjects, self).__init__(phrases=[
            "race", "contest", "war", "prize", "competition", "election", "battle", "award", "tournament"
        ])


class ReadWroteObjects(HasPhrase):

    def __init__(self):
        super(ReadWroteObjects, self).__init__(phrases=[
            "book", "column", "report", "poem", "letter", "novel", "story", "play", "speech"
        ])


class Adjectives(HasPhrase):

    def __init__(self):
        super(Adjectives, self).__init__(phrases=[
            "important", "popular", "famous", "young", "happy", "helpful", "serious", "angry"
        ])


class AdjectivesCompNonEnt(HasPhrase):

    def __init__(self):
        super(AdjectivesCompNonEnt, self).__init__(phrases=[
            "afraid", "sure", "certain"
        ])


class AdjectivesCompEnt(HasPhrase):

    def __init__(self):
        super(AdjectivesCompEnt, self).__init__(phrases=[
            "sorry", "aware", "glad"
        ])


class Adverbs(HasPhrase):

    def __init__(self):
        super(Adverbs, self).__init__(phrases=[
            "quickly", "slowly", "happily", "easily", "quietly", "thoughtfully"
        ])


class ConstAdv(HasPhrase):

    def __init__(self):
        super(ConstAdv, self).__init__(phrases=[
            "after", "before", "because", "although", "though", "since", "while"
        ])


class ConstQuotEntailed(HasPhrase):

    def __init__(self):
        super(ConstQuotEntailed, self).__init__(phrases=[
            "forgot", "learned", "remembered", "knew"
        ])


class Relations(HasPhrase):

    def __init__(self):
        super(Relations, self).__init__(phrases=[
            "who", "that"
        ])


class Questions(HasPhrase):

    def __init__(self):
        super(Questions, self).__init__(phrases=[
            "why", "how"
        ])


class NonEntComplementNouns(HasPhrase):

    def __init__(self):
        super(NonEntComplementNouns, self).__init__(phrases=[
            "feeling", "evidence", "idea", "belief"
        ])


class EntComplementNouns(HasPhrase):

    def __init__(self):
        super(EntComplementNouns, self).__init__(phrases=[
            "fact", "reason", "news", "time"
        ])


class AdvsNonEntailed(HasPhrase):

    def __init__(self):
        super(AdvsNonEntailed, self).__init__(phrases=[
            "supposedly", "probably", "maybe", "hopefully"
        ])


class AdvsEntailed(HasPhrase):

    def __init__(self):
        super(AdvsEntailed, self).__init__(phrases=[
            "certainly", "definitely", "clearly", "obviously", "suddenly"
        ])
