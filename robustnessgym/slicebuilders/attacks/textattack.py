from collections import OrderedDict
from typing import Dict, List, Tuple

import cytoolz as tz
import numpy as np

try:
    import textattack.attack_recipes as attack_recipes
    from textattack.attack_recipes import AttackRecipe
    from textattack.models.wrappers import HuggingFaceModelWrapper, ModelWrapper
except ImportError:
    _textattack_available = False
else:
    _textattack_available = True

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.model import Model
from robustnessgym.slicebuilders.attack import Attack


class TextAttack(Attack):
    """Class for TextAttack."""

    def __init__(
        self,
        attack: AttackRecipe,
    ):
        if not _textattack_available:
            raise ImportError("Textattack not found. Please `pip install textattack`.")

        super(TextAttack, self).__init__(
            identifiers=[
                Identifier(
                    self.__class__.__name__,
                    attack=attack,
                )
            ],
        )

        self.attack = attack

    @classmethod
    def recipes(cls):
        recipes = []
        for possible_recipe_name in dir(attack_recipes):
            possible_recipe = getattr(attack_recipes, possible_recipe_name)
            if hasattr(possible_recipe, "mro"):
                for _cls in possible_recipe.mro():
                    if _cls == AttackRecipe and possible_recipe != AttackRecipe:
                        recipes.append(possible_recipe_name)
        return recipes

    def apply(
        self,
        skeleton_batches: List[Dict[str, List]],
        slice_membership: np.ndarray,
        batch: Dict[str, List],
        columns: List[str],
        *args,
        **kwargs
    ) -> Tuple[List[Dict[str, List]], np.ndarray]:

        # Group the batch into inputs and output
        batch_inputs = tz.keyfilter(lambda k: k in columns[:-1], batch)
        batch_inputs = [
            OrderedDict(zip(batch_inputs, t)) for t in zip(*batch_inputs.values())
        ]

        batch_output = [int(e) for e in batch[columns[-1]]]

        # Create a fake dataset for textattack
        fake_dataset = list(zip(batch_inputs, batch_output))

        # Attack the dataset
        outputs = list(self.attack.attack_dataset(fake_dataset))

        for i, output in enumerate(outputs):
            # Check if the goal succeeded
            if output.perturbed_result.goal_status == 0:
                # If success, fill out the skeleton batch
                for (
                    key,
                    val,
                ) in output.perturbed_result.attacked_text._text_input.items():
                    # TODO(karan): support num_attacked_texts > 1
                    skeleton_batches[0][key][i] = val

                # # Fill the perturbed output: *this was incorrect, removing this
                # statement*
                # # TODO(karan): delete this snippet
                # skeleton_batches[0][columns[-1]][i] = output.perturbed_result.output
            else:
                # Unable to attack the example: set its slice membership to zero
                slice_membership[i, 0] = 0

        return skeleton_batches, slice_membership

    @classmethod
    def from_recipe(cls, recipe: str, model: ModelWrapper):
        return cls(attack=getattr(attack_recipes, recipe).build(model=model))

    @classmethod
    def wrap_huggingface_model(cls, model: Model) -> ModelWrapper:
        return HuggingFaceModelWrapper(model=model.model, tokenizer=model.tokenizer)
