from __future__ import annotations

from typing import List, Dict, Tuple, Any, Sequence, Optional

import cytoolz as tz
import numpy as np
from ahocorasick import Automaton

from robustness_gym.cached_ops.spacy.spacy import Spacy
from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.subpopulation import Subpopulation


class AhoCorasickMixin:

    def __init__(self, *args, **kwargs):
        super(AhoCorasickMixin, self).__init__(*args, **kwargs)
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


class HasPhrase(AhoCorasickMixin,
                Subpopulation,
                Spacy):

    def __init__(self,
                 phrases=None,
                 *args,
                 **kwargs):

        super(HasPhrase, self).__init__(
            # One slice per phrase
            identifiers=[
                Identifier(name=self.__class__.__name__,
                           phrase=phrase)
                for phrase in phrases
            ],
            *args,
            **kwargs
        )

        # This is the list of phrases that will be searched
        self.phrases = phrases
        if self.phrases is None:
            self.phrases = []

        # Populate the Aho-Corasick automaton
        self.populate_automaton(
            {i: phrase for i, phrase in enumerate(self.phrases)})

    @classmethod
    def from_file(cls, path: str) -> Subpopulation:
        """
        Load phrases from a file, one per line.
        """
        with open(path) as f:
            phrases = [line.strip() for line in f.readlines()]
        return cls(phrases=phrases)

    @classmethod
    def default(cls) -> Subpopulation:
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


class HasAnyPhrase(Subpopulation):

    def __init__(self,
                 phrases=None):
        # Take the union of the phrases
        subpopulation = Subpopulation.union(
            HasPhrase(phrases=phrases),
            identifier=Identifier(
                name=self.__class__.__name__,
                phrases=set(phrases),
            )
        )

        super(HasAnyPhrase, self).__init__(identifiers=subpopulation.identifiers,
                                           apply_fn=subpopulation.apply)


class HasAllPhrases(Subpopulation):

    def __init__(self,
                 phrases=None):
        # Take the intersection of the phrases
        subpopulation = Subpopulation.intersection(
            HasPhrase(phrases=phrases),
            identifier=Identifier(
                name=self.__class__.__name__,
                phrases=set(phrases),
            )
        )

        super(HasAllPhrases, self).__init__(identifiers=subpopulation.identifiers,
                                            apply_fn=subpopulation.apply)
