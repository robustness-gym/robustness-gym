from __future__ import annotations

from typing import Any, Dict, List, Sequence

import cytoolz as tz
import numpy as np
from ahocorasick import Automaton

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.operation import lookup
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.ops.spacy import SpacyOp
from robustnessgym.slicebuilders.subpopulation import Subpopulation


class AhoCorasick:
    def __init__(self, *args, **kwargs):
        super(AhoCorasick, self).__init__(*args, **kwargs)

        # Use the Aho-Corasick search algorithm to speed up phrase lookups
        self.automaton = Automaton()

    @classmethod
    def from_phrases(cls, phrases: Dict[Any, str]) -> AhoCorasick:
        # Create a new automaton
        ahocorasick = cls()

        # Add all the phrases we want to search for
        for key, phrase in phrases.items():
            # As values, we add the key of the phrase
            ahocorasick.automaton.add_word(phrase, key)

        # Initialize Aho-Corasick
        ahocorasick.automaton.make_automaton()

        return ahocorasick


class HasPhrase(Subpopulation):
    def __init__(
        self, phrases=None, identifiers: List[Identifier] = None, *args, **kwargs
    ):

        super(HasPhrase, self).__init__(
            # One slice per phrase
            identifiers=[
                Identifier(_name=self.__class__.__name__, phrase=phrase)
                for phrase in phrases
            ]
            if not identifiers
            else identifiers,
            *args,
            **kwargs,
        )

        # This is the list of phrases that will be searched
        self.phrases = phrases
        if self.phrases is None:
            self.phrases = []

        # Create and populate Aho-Corasick automatons for words and phrases
        self.word_ahocorasick = AhoCorasick.from_phrases(
            {i: phrase for i, phrase in enumerate(self.phrases) if " " not in phrase}
        )
        self.phrase_ahocorasick = AhoCorasick.from_phrases(
            {i: phrase for i, phrase in enumerate(self.phrases) if " " in phrase}
        )

    @classmethod
    def from_file(cls, path: str) -> Subpopulation:
        """Load phrases from a file, one per line."""
        with open(path) as f:
            phrases = [line.strip() for line in f.readlines()]
        return cls(phrases=phrases)

    @classmethod
    def default(cls) -> Subpopulation:
        """A default vocabulary of phrases to search."""
        return cls(phrases=[])

    @classmethod
    def join(cls, *slicers: HasPhrase) -> Sequence[HasPhrase]:
        """Join to combine multiple HasPhrase slicers."""
        return [
            HasPhrase(phrases=list(tz.concat([slicer.phrases for slicer in slicers])))
        ]

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        slice_membership: np.ndarray = None,
        *args,
        **kwargs,
    ) -> np.ndarray:

        # Search for words
        if len(self.word_ahocorasick.automaton) > 0:
            for col in columns:
                try:
                    docs = lookup(batch, SpacyOp, [col])
                except AttributeError:
                    docs = [text.split() for text in batch[col]]
                for i, doc in enumerate(docs):
                    # Get the values (indices) of all the matched tokens
                    matched_indices = [
                        self.word_ahocorasick.automaton.get(str(token))
                        for token in doc
                        if self.word_ahocorasick.automaton.exists(str(token))
                    ]

                    # Fill in the slice labels for slices that are present
                    slice_membership[i, matched_indices] = 1

        # Search for phrases
        if len(self.phrase_ahocorasick.automaton) > 0:
            for col in columns:
                for i, example in enumerate(batch[col]):
                    # Get the values (indices) of all the matched phrases
                    matched_indices = [
                        index
                        for _, index in self.phrase_ahocorasick.automaton.iter(example)
                    ]

                    # Fill in the slice labels for slices that are present
                    slice_membership[i, matched_indices] = 1

        return slice_membership


class HasAnyPhrase(Subpopulation):
    def __init__(
        self,
        phrase_groups: List[List[str]] = None,
        identifiers: List[Identifier] = None,
        *args,
        **kwargs,
    ):

        # Keep track of the phrase groups
        self.phrase_groups = phrase_groups

        if identifiers:
            assert len(identifiers) == len(
                phrase_groups
            ), "Must have one identifier per phrase group."

        self.subpopulations = []
        # For every phrase group
        for i, phrases in enumerate(phrase_groups):
            # Take the union of the phrases
            self.subpopulations.append(
                Subpopulation.union(
                    HasPhrase(phrases=phrases),
                    identifier=Identifier(
                        _name=self.__class__.__name__,
                        phrases=set(phrases),
                    )
                    if not identifiers
                    else identifiers[i],
                )
            )

        super(HasAnyPhrase, self).__init__(
            identifiers=list(
                tz.concat(
                    [subpopulation.identifiers for subpopulation in self.subpopulations]
                )
            ),
            *args,
            **kwargs,
        )

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        slice_membership: np.ndarray = None,
        *args,
        **kwargs,
    ) -> np.ndarray:

        # Run all the subpopulations in sequence to update the slice membership matrix
        for i, subpopulation in enumerate(self.subpopulations):
            slice_membership[:, i : i + 1] = subpopulation.apply(
                slice_membership=slice_membership[:, i : i + 1],
                batch=batch,
                columns=columns,
                *args,
                **kwargs,
            )

        return slice_membership

    @classmethod
    def join(cls, *slicebuilders: HasAnyPhrase) -> Sequence[HasAnyPhrase]:
        # Join all the slicebuilders
        return [
            HasAnyPhrase(
                phrase_groups=[
                    phrases
                    for slicebuilder in slicebuilders
                    for phrases in slicebuilder.phrase_groups
                ],
                identifiers=[
                    identifier
                    for slicebuilder in slicebuilders
                    for identifier in slicebuilder.identifiers
                ],
            )
        ]


class HasAllPhrases(Subpopulation):
    def __init__(
        self,
        phrase_groups: List[List[str]] = None,
        identifiers: List[Identifier] = None,
        *args,
        **kwargs,
    ):

        # Keep track of the phrase groups
        self.phrase_groups = phrase_groups

        if identifiers:
            assert len(identifiers) == len(
                phrase_groups
            ), "Must have one identifier per phrase group."

        self.subpopulations = []
        # For every phrase group
        for i, phrases in enumerate(phrase_groups):
            # Take the union of the phrases
            self.subpopulations.append(
                Subpopulation.intersection(
                    HasPhrase(phrases=phrases),
                    identifier=Identifier(
                        _name=self.__class__.__name__,
                        phrases=set(phrases),
                    )
                    if not identifiers
                    else identifiers[i],
                )
            )

        super(HasAllPhrases, self).__init__(
            identifiers=list(
                tz.concat(
                    [subpopulation.identifiers for subpopulation in self.subpopulations]
                )
            ),
            *args,
            **kwargs,
        )

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        slice_membership: np.ndarray = None,
        *args,
        **kwargs,
    ) -> np.ndarray:

        # Run all the subpopulations in sequence to update the slice membership matrix
        for i, subpopulation in enumerate(self.subpopulations):
            slice_membership[:, i : i + 1] = subpopulation.apply(
                slice_membership=slice_membership[:, i : i + 1],
                batch=batch,
                columns=columns,
                *args,
                **kwargs,
            )

        return slice_membership

    @classmethod
    def join(cls, *slicebuilders: HasAllPhrases) -> Sequence[HasAllPhrases]:
        # Join all the slicebuilders
        return [
            HasAllPhrases(
                phrase_groups=[
                    phrases
                    for slicebuilder in slicebuilders
                    for phrases in slicebuilder.phrase_groups
                ],
                identifiers=[
                    identifier
                    for slicebuilder in slicebuilders
                    for identifier in slicebuilder.identifiers
                ],
            )
        ]


class HasIndefiniteArticle(HasAnyPhrase):
    def __init__(self):
        super(HasIndefiniteArticle, self).__init__(
            phrase_groups=[["a", "an"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HasDefiniteArticle(HasAnyPhrase):
    def __init__(self):
        super(HasDefiniteArticle, self).__init__(
            phrase_groups=[["the"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HasTemporalPreposition(HasAnyPhrase):
    def __init__(self):
        super(HasTemporalPreposition, self).__init__(
            phrase_groups=[["after", "before", "past"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HasPosessivePreposition(HasAnyPhrase):
    def __init__(self):
        super(HasPosessivePreposition, self).__init__(
            phrase_groups=[["inside of", "with", "within"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HasComparison(HasAnyPhrase):
    def __init__(self):
        super(HasComparison, self).__init__(
            phrase_groups=[["more", "less", "better", "worse", "bigger", "smaller"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HasQuantifier(HasAnyPhrase):
    def __init__(self):
        super(HasQuantifier, self).__init__(
            phrase_groups=[["all", "some", "none"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HasNegation(HasAnyPhrase):
    def __init__(self):
        super(HasNegation, self).__init__(
            phrase_groups=[
                [
                    "no",
                    "not",
                    "none",
                    "noone ",
                    "nobody",
                    "nothing",
                    "neither",
                    "nowhere",
                    "never",
                    "hardly",
                    "scarcely",
                    "barely",
                    "doesnt",
                    "isnt",
                    "wasnt",
                    "shouldnt",
                    "wouldnt",
                    "couldnt",
                    "wont",
                    "cant",
                    "dont",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )
