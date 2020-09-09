from __future__ import annotations

from typing import List, Dict, Any, Sequence

import cytoolz as tz
import numpy as np
from ahocorasick import Automaton

from robustness_gym.cached_ops.spacy.spacy import Spacy
from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.subpopulation import Subpopulation


class AhoCorasick:

    def __init__(self, *args, **kwargs):
        super(AhoCorasick, self).__init__(*args, **kwargs)

        # Use the Aho-Corasick search algorithm to speed up phrase lookups
        self.automaton = Automaton()

    @classmethod
    def from_phrases(cls,
                     phrases: Dict[Any, str]) -> AhoCorasick:
        # Create a new automaton
        ahocorasick = cls()

        # Add all the phrases we want to search for
        for key, phrase in phrases.items():
            # As values, we add the key of the phrase
            ahocorasick.automaton.add_word(phrase, key)

        # Initialize Aho-Corasick
        ahocorasick.automaton.make_automaton()

        return ahocorasick


class HasPhrase(Subpopulation,
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

        # Create and populate Aho-Corasick automatons for words and phrases
        self.word_ahocorasick = AhoCorasick.from_phrases({
            i: phrase for i, phrase in enumerate(self.phrases) if " " not in phrase
        })
        self.phrase_ahocorasick = AhoCorasick.from_phrases({
            i: phrase for i, phrase in enumerate(self.phrases) if " " in phrase
        })

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
        tokenized_batch = Spacy.retrieve(
            batch=batch,
            keys=[[key] for key in keys],
            proc_fns='tokens',
        )

        # Search for words
        if len(self.word_ahocorasick.automaton) > 0:
            for key, tokens_batch in tokenized_batch.items():
                for i, tokens in enumerate(tokens_batch):
                    # Get the values (indices) of all the matched tokens
                    matched_indices = [
                        self.word_ahocorasick.automaton.get(token) for token in tokens
                        if self.word_ahocorasick.automaton.exists(token)
                    ]

                    # Fill in the slice labels for slices that are present
                    slice_membership[i, matched_indices] = 1

        # Search for phrases
        if len(self.phrase_ahocorasick.automaton) > 0:
            for key in keys:
                for i, example in enumerate(batch[key]):
                    # Get the values (indices) of all the matched phrases
                    matched_indices = [index for _, index in self.phrase_ahocorasick.automaton.iter(example)]

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


class HasArticle(HasAnyPhrase):
    pass
