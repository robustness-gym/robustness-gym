"""
Taken from https://github.com/tommccoy1/hans/blob/master/templates.py
"""

from robustnessgym.slicebuilders.subpopulation import SubpopulationCollection
from robustnessgym.slicebuilders.subpopulations.phrase.phrase import HasAnyPhrase


class HansAllPhrases(SubpopulationCollection):

    def __init__(self,
                 *args,
                 **kwargs):
        super(HansAllPhrases, self).__init__(
            subpopulations=[
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
            ],
            *args,
            **kwargs
        )


class HansSingularNouns(HasAnyPhrase):

    def __init__(self):
        super(HansSingularNouns, self).__init__(phrases=[
            "professor", "student", "president", "judge", "senator", "secretary", "doctor", "lawyer", "scientist",
            "banker", "tourist", "manager", "artist", "author", "actor", "athlete",
        ])


class HansPluralNouns(HasAnyPhrase):

    def __init__(self):
        super(HansPluralNouns, self).__init__(phrases=[
            "professors", "students", "presidents", "judges", "senators", "secretaries", "doctors", "lawyers",
            "scientists", "bankers", "tourists", "managers", "artists", "authors", "actors", "athletes",
        ])


class HansTransitiveVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansTransitiveVerbs, self).__init__(phrases=[
            "recommended", "called", "helped", "supported", "contacted", "believed", "avoided", "advised",
            "saw", "stopped", "introduced", "mentioned", "encouraged", "thanked", "recognized", "admired"
        ])


class HansPassiveVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansPassiveVerbs, self).__init__(phrases=[
            "recommended", "helped", "supported", "contacted", "believed", "avoided", "advised", "stopped",
            "introduced", "mentioned", "encouraged", "thanked", "recognized", "admired"
        ])


class HansIntransitiveVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansIntransitiveVerbs, self).__init__(phrases=[
            "slept", "danced", "ran", "shouted", "resigned", "waited", "arrived", "performed"
        ])


class HansNPSVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansNPSVerbs, self).__init__(phrases=[
            "believed", "knew", "heard", "forgot", "preferred", "claimed", "wanted", "needed",
            "found", "suggested", "expected"
        ])


class HansNPZVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansNPZVerbs, self).__init__(phrases=[
            "hid", "moved", "presented", "paid", "studied", "stopped"
        ])


class HansPluralNPZVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansPluralNPZVerbs, self).__init__(phrases=[
            "fought", "paid", "changed", "studied", "answered", "stopped", "grew", "moved", "returned",
            "left", "improved", "lost", "visited", "ate", "played"
        ])


class HansPrepositions(HasAnyPhrase):

    def __init__(self):
        super(HansPrepositions, self).__init__(phrases=[
            "near", "behind", "by", "in front of", "next to"
        ])


class HansConjs(HasAnyPhrase):

    def __init__(self):
        super(HansConjs, self).__init__(phrases=[
            "while", "after", "before", "when", "although", "because", "since"
        ])


class HansPastParticiples(HasAnyPhrase):

    def __init__(self):
        super(HansPastParticiples, self).__init__(phrases=[
            "studied", "paid", "helped", "investigated", "presented"
        ])


class HansUnderstoodArgumentVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansUnderstoodArgumentVerbs, self).__init__(phrases=[
            "paid", "explored", "won", "wrote", "left", "read", "ate"
        ])


class HansNonEntQuotVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansNonEntQuotVerbs, self).__init__(phrases=[
            "hoped", "claimed", "thought", "believed", "said", "assumed"
        ])


class HansQuestionEmbeddingVerbs(HasAnyPhrase):

    def __init__(self):
        super(HansQuestionEmbeddingVerbs, self).__init__(phrases=[
            "wondered", "understood", "knew", "asked", "explained", "realized"
        ])


class HansCalledObjects(HasAnyPhrase):

    def __init__(self):
        super(HansCalledObjects, self).__init__(phrases=[
            "coward", "liar", "hero", "fool"
        ])


class HansToldObjects(HasAnyPhrase):

    def __init__(self):
        super(HansToldObjects, self).__init__(phrases=[
            "story", "lie", "truth", "secret"
        ])


class HansFoodWords(HasAnyPhrase):

    def __init__(self):
        super(HansFoodWords, self).__init__(phrases=[
            "fruit", "salad", "broccoli", "sandwich", "rice", "corn", "ice cream"
        ])


class HansLocationNounsA(HasAnyPhrase):

    def __init__(self):
        super(HansLocationNounsA, self).__init__(phrases=[
            "neighborhood", "region", "country", "town", "valley", "forest", "garden", "museum", "desert",
            "island", "town"
        ])


class HansLocationNounsB(HasAnyPhrase):

    def __init__(self):
        super(HansLocationNounsB, self).__init__(phrases=[
            "museum", "school", "library", "office", "laboratory"
        ])


class HansWonObjects(HasAnyPhrase):

    def __init__(self):
        super(HansWonObjects, self).__init__(phrases=[
            "race", "contest", "war", "prize", "competition", "election", "battle", "award", "tournament"
        ])


class HansReadWroteObjects(HasAnyPhrase):

    def __init__(self):
        super(HansReadWroteObjects, self).__init__(phrases=[
            "book", "column", "report", "poem", "letter", "novel", "story", "play", "speech"
        ])


class HansAdjectives(HasAnyPhrase):

    def __init__(self):
        super(HansAdjectives, self).__init__(phrases=[
            "important", "popular", "famous", "young", "happy", "helpful", "serious", "angry"
        ])


class HansAdjectivesCompNonEnt(HasAnyPhrase):

    def __init__(self):
        super(HansAdjectivesCompNonEnt, self).__init__(phrases=[
            "afraid", "sure", "certain"
        ])


class HansAdjectivesCompEnt(HasAnyPhrase):

    def __init__(self):
        super(HansAdjectivesCompEnt, self).__init__(phrases=[
            "sorry", "aware", "glad"
        ])


class HansAdverbs(HasAnyPhrase):

    def __init__(self):
        super(HansAdverbs, self).__init__(phrases=[
            "quickly", "slowly", "happily", "easily", "quietly", "thoughtfully"
        ])


class HansConstAdv(HasAnyPhrase):

    def __init__(self):
        super(HansConstAdv, self).__init__(phrases=[
            "after", "before", "because", "although", "though", "since", "while"
        ])


class HansConstQuotEntailed(HasAnyPhrase):

    def __init__(self):
        super(HansConstQuotEntailed, self).__init__(phrases=[
            "forgot", "learned", "remembered", "knew"
        ])


class HansRelations(HasAnyPhrase):

    def __init__(self):
        super(HansRelations, self).__init__(phrases=[
            "who", "that"
        ])


class HansQuestions(HasAnyPhrase):

    def __init__(self):
        super(HansQuestions, self).__init__(phrases=[
            "why", "how"
        ])


class HansNonEntComplementNouns(HasAnyPhrase):

    def __init__(self):
        super(HansNonEntComplementNouns, self).__init__(phrases=[
            "feeling", "evidence", "idea", "belief"
        ])


class HansEntComplementNouns(HasAnyPhrase):

    def __init__(self):
        super(HansEntComplementNouns, self).__init__(phrases=[
            "fact", "reason", "news", "time"
        ])


class HansAdvsNonEntailed(HasAnyPhrase):

    def __init__(self):
        super(HansAdvsNonEntailed, self).__init__(phrases=[
            "supposedly", "probably", "maybe", "hopefully"
        ])


class HansAdvsEntailed(HasAnyPhrase):

    def __init__(self):
        super(HansAdvsEntailed, self).__init__(phrases=[
            "certainly", "definitely", "clearly", "obviously", "suddenly"
        ])
