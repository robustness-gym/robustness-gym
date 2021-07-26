"""Taken from https://github.com/tommccoy1/hans/blob/master/templates.py."""

from robustnessgym.core.identifier import Identifier
from robustnessgym.slicebuilders.subpopulation_collection import SubpopulationCollection
from robustnessgym.slicebuilders.subpopulations.phrase import HasAnyPhrase


class HansAllPhrases(SubpopulationCollection):
    def __init__(self, *args, **kwargs):
        super(HansAllPhrases, self).__init__(
            subpopulations=HasAnyPhrase.join(
                *[
                    HansSingularNouns(),
                    HansPluralNouns(),
                    HansTransitiveVerbs(),
                    HansPassiveVerbs(),
                    HansIntransitiveVerbs(),
                    HansNPSVerbs(),
                    HansNPZVerbs(),
                    HansPluralNPZVerbs(),
                    HansPrepositions(),
                    HansConjs(),
                    HansPastParticiples(),
                    HansUnderstoodArgumentVerbs(),
                    HansNonEntQuotVerbs(),
                    HansQuestionEmbeddingVerbs(),
                    HansCalledObjects(),
                    HansToldObjects(),
                    HansFoodWords(),
                    HansLocationNounsA(),
                    HansLocationNounsB(),
                    HansWonObjects(),
                    HansReadWroteObjects(),
                    HansAdjectives(),
                    HansAdjectivesCompNonEnt(),
                    HansAdjectivesCompEnt(),
                    HansAdverbs(),
                    HansConstAdv(),
                    HansConstQuotEntailed(),
                    HansRelations(),
                    HansQuestions(),
                    HansNonEntComplementNouns(),
                    HansEntComplementNouns(),
                    HansAdvsNonEntailed(),
                    HansAdvsEntailed(),
                ]
            ),
            *args,
            **kwargs
        )


class HansSingularNouns(HasAnyPhrase):
    def __init__(self):
        super(HansSingularNouns, self).__init__(
            phrase_groups=[
                [
                    "professor",
                    "student",
                    "president",
                    "judge",
                    "senator",
                    "secretary",
                    "doctor",
                    "lawyer",
                    "scientist",
                    "banker",
                    "tourist",
                    "manager",
                    "artist",
                    "author",
                    "actor",
                    "athlete",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansPluralNouns(HasAnyPhrase):
    def __init__(self):
        super(HansPluralNouns, self).__init__(
            phrase_groups=[
                [
                    "professors",
                    "students",
                    "presidents",
                    "judges",
                    "senators",
                    "secretaries",
                    "doctors",
                    "lawyers",
                    "scientists",
                    "bankers",
                    "tourists",
                    "managers",
                    "artists",
                    "authors",
                    "actors",
                    "athletes",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansTransitiveVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansTransitiveVerbs, self).__init__(
            phrase_groups=[
                [
                    "recommended",
                    "called",
                    "helped",
                    "supported",
                    "contacted",
                    "believed",
                    "avoided",
                    "advised",
                    "saw",
                    "stopped",
                    "introduced",
                    "mentioned",
                    "encouraged",
                    "thanked",
                    "recognized",
                    "admired",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansPassiveVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansPassiveVerbs, self).__init__(
            phrase_groups=[
                [
                    "recommended",
                    "helped",
                    "supported",
                    "contacted",
                    "believed",
                    "avoided",
                    "advised",
                    "stopped",
                    "introduced",
                    "mentioned",
                    "encouraged",
                    "thanked",
                    "recognized",
                    "admired",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansIntransitiveVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansIntransitiveVerbs, self).__init__(
            phrase_groups=[
                [
                    "slept",
                    "danced",
                    "ran",
                    "shouted",
                    "resigned",
                    "waited",
                    "arrived",
                    "performed",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansNPSVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansNPSVerbs, self).__init__(
            phrase_groups=[
                [
                    "believed",
                    "knew",
                    "heard",
                    "forgot",
                    "preferred",
                    "claimed",
                    "wanted",
                    "needed",
                    "found",
                    "suggested",
                    "expected",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansNPZVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansNPZVerbs, self).__init__(
            phrase_groups=[["hid", "moved", "presented", "paid", "studied", "stopped"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansPluralNPZVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansPluralNPZVerbs, self).__init__(
            phrase_groups=[
                [
                    "fought",
                    "paid",
                    "changed",
                    "studied",
                    "answered",
                    "stopped",
                    "grew",
                    "moved",
                    "returned",
                    "left",
                    "improved",
                    "lost",
                    "visited",
                    "ate",
                    "played",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansPrepositions(HasAnyPhrase):
    def __init__(self):
        super(HansPrepositions, self).__init__(
            phrase_groups=[["near", "behind", "by", "in front of", "next to"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansConjs(HasAnyPhrase):
    def __init__(self):
        super(HansConjs, self).__init__(
            phrase_groups=[
                ["while", "after", "before", "when", "although", "because", "since"]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansPastParticiples(HasAnyPhrase):
    def __init__(self):
        super(HansPastParticiples, self).__init__(
            phrase_groups=[["studied", "paid", "helped", "investigated", "presented"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansUnderstoodArgumentVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansUnderstoodArgumentVerbs, self).__init__(
            phrase_groups=[["paid", "explored", "won", "wrote", "left", "read", "ate"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansNonEntQuotVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansNonEntQuotVerbs, self).__init__(
            phrase_groups=[
                ["hoped", "claimed", "thought", "believed", "said", "assumed"]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansQuestionEmbeddingVerbs(HasAnyPhrase):
    def __init__(self):
        super(HansQuestionEmbeddingVerbs, self).__init__(
            phrase_groups=[
                ["wondered", "understood", "knew", "asked", "explained", "realized"]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansCalledObjects(HasAnyPhrase):
    def __init__(self):
        super(HansCalledObjects, self).__init__(
            phrase_groups=[["coward", "liar", "hero", "fool"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansToldObjects(HasAnyPhrase):
    def __init__(self):
        super(HansToldObjects, self).__init__(
            phrase_groups=[["story", "lie", "truth", "secret"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansFoodWords(HasAnyPhrase):
    def __init__(self):
        super(HansFoodWords, self).__init__(
            phrase_groups=[
                ["fruit", "salad", "broccoli", "sandwich", "rice", "corn", "ice cream"]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansLocationNounsA(HasAnyPhrase):
    def __init__(self):
        super(HansLocationNounsA, self).__init__(
            phrase_groups=[
                [
                    "neighborhood",
                    "region",
                    "country",
                    "town",
                    "valley",
                    "forest",
                    "garden",
                    "museum",
                    "desert",
                    "island",
                    "town",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansLocationNounsB(HasAnyPhrase):
    def __init__(self):
        super(HansLocationNounsB, self).__init__(
            phrase_groups=[["museum", "school", "library", "office", "laboratory"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansWonObjects(HasAnyPhrase):
    def __init__(self):
        super(HansWonObjects, self).__init__(
            phrase_groups=[
                [
                    "race",
                    "contest",
                    "war",
                    "prize",
                    "competition",
                    "election",
                    "battle",
                    "award",
                    "tournament",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansReadWroteObjects(HasAnyPhrase):
    def __init__(self):
        super(HansReadWroteObjects, self).__init__(
            phrase_groups=[
                [
                    "book",
                    "column",
                    "report",
                    "poem",
                    "letter",
                    "novel",
                    "story",
                    "play",
                    "speech",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansAdjectives(HasAnyPhrase):
    def __init__(self):
        super(HansAdjectives, self).__init__(
            phrase_groups=[
                [
                    "important",
                    "popular",
                    "famous",
                    "young",
                    "happy",
                    "helpful",
                    "serious",
                    "angry",
                ]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansAdjectivesCompNonEnt(HasAnyPhrase):
    def __init__(self):
        super(HansAdjectivesCompNonEnt, self).__init__(
            phrase_groups=[["afraid", "sure", "certain"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansAdjectivesCompEnt(HasAnyPhrase):
    def __init__(self):
        super(HansAdjectivesCompEnt, self).__init__(
            phrase_groups=[["sorry", "aware", "glad"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansAdverbs(HasAnyPhrase):
    def __init__(self):
        super(HansAdverbs, self).__init__(
            phrase_groups=[
                ["quickly", "slowly", "happily", "easily", "quietly", "thoughtfully"]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansConstAdv(HasAnyPhrase):
    def __init__(self):
        super(HansConstAdv, self).__init__(
            phrase_groups=[
                ["after", "before", "because", "although", "though", "since", "while"]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansConstQuotEntailed(HasAnyPhrase):
    def __init__(self):
        super(HansConstQuotEntailed, self).__init__(
            phrase_groups=[["forgot", "learned", "remembered", "knew"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansRelations(HasAnyPhrase):
    def __init__(self):
        super(HansRelations, self).__init__(
            phrase_groups=[["who", "that"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansQuestions(HasAnyPhrase):
    def __init__(self):
        super(HansQuestions, self).__init__(
            phrase_groups=[["why", "how"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansNonEntComplementNouns(HasAnyPhrase):
    def __init__(self):
        super(HansNonEntComplementNouns, self).__init__(
            phrase_groups=[["feeling", "evidence", "idea", "belief"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansEntComplementNouns(HasAnyPhrase):
    def __init__(self):
        super(HansEntComplementNouns, self).__init__(
            phrase_groups=[["fact", "reason", "news", "time"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansAdvsNonEntailed(HasAnyPhrase):
    def __init__(self):
        super(HansAdvsNonEntailed, self).__init__(
            phrase_groups=[["supposedly", "probably", "maybe", "hopefully"]],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )


class HansAdvsEntailed(HasAnyPhrase):
    def __init__(self):
        super(HansAdvsEntailed, self).__init__(
            phrase_groups=[
                ["certainly", "definitely", "clearly", "obviously", "suddenly"]
            ],
            identifiers=[Identifier(_name=self.__class__.__name__)],
        )
