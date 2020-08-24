from robustness_gym import *

import textattack.attack_recipes as attack_recipes
import cytoolz as tz
from textattack.attack_recipes import AttackRecipe
from textattack.models.wrappers import ModelWrapper


class TextAttack(Attack):

    def __init__(
            self,
            attack: AttackRecipe,
    ):
        super(TextAttack, self).__init__(
            num_slices=1,
            identifiers=[attack.__class__.__name__],
        )

        self.attack = attack

    def apply(self, batch, keys):
        # Group the batch into inputs and output
        batch_inputs = tz.keyfilter(lambda k: k in keys[:-1], batch)
        batch_inputs = [dict(zip(batch_inputs, t)) for t in zip(*batch_inputs.values())]

        batch_output = batch[keys[-1]]

        # Create a fake dataset for textattack
        fake_dataset = list(zip(batch_inputs, batch_output))

        # Attack the dataset
        outputs = self.attack.attack_dataset(fake_dataset)

        # TODO(karan): finish this up

    @classmethod
    def from_recipe(cls,
                    recipe: str,
                    model: ModelWrapper):
        return cls(attack=getattr(attack_recipes, recipe).build(model=model))
