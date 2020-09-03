from collections import OrderedDict
from typing import List, Dict, Tuple

import cytoolz as tz
import numpy as np
import textattack.attack_recipes as attack_recipes
from textattack.attack_recipes import AttackRecipe
from textattack.models.wrappers import ModelWrapper

from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.attack import Attack


class TextAttack(Attack):

    def __init__(
            self,
            attack: AttackRecipe,
    ):
        super(TextAttack, self).__init__(
            identifiers=[Identifier(
                self.__class__.__name__,
                attack=attack,
            )],
        )

        self.attack = attack

    def apply(self,
              skeleton_batches: List[Dict[str, List]],
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> Tuple[List[Dict[str, List]], np.ndarray]:

        # Group the batch into inputs and output
        batch_inputs = tz.keyfilter(lambda k: k in keys[:-1], batch)
        batch_inputs = [OrderedDict(zip(batch_inputs, t)) for t in zip(*batch_inputs.values())]

        batch_output = [int(e) for e in batch[keys[-1]]]

        # Create a fake dataset for textattack
        fake_dataset = list(zip(batch_inputs, batch_output))

        # Attack the dataset
        outputs = list(self.attack.attack_dataset(fake_dataset))

        for i, output in enumerate(outputs):
            # Check if the goal succeeded
            if output.perturbed_result.goal_status == 0:
                # If success, fill out the skeleton batch
                for key, val in output.perturbed_result.attacked_text._text_input.items():
                    # TODO(karan): support num_attacked_texts > 1
                    skeleton_batches[0][key][i] = val

                # Fill the perturbed output
                skeleton_batches[0][keys[-1]][i] = output.perturbed_result.output
            else:
                # Unable to attack the example: set its slice membership to zero
                slice_membership[i, 0] = 0

        return skeleton_batches, slice_membership

    @classmethod
    def from_recipe(cls,
                    recipe: str,
                    model: ModelWrapper):
        return cls(attack=getattr(attack_recipes, recipe).build(model=model))
