from collections import OrderedDict
from typing import List, Tuple

import cytoolz as tz
import numpy as np
from meerkat.tools.lazy_loader import LazyLoader

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.model import Model
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.slicebuilders.attack import Attack

attack_recipes = LazyLoader(
    "textattack.attack_recipes",
    error="Install TextAttack with `pip install textattack`.",
)
wrappers = LazyLoader("textattack.models.wrappers")


class TextAttack(Attack):
    """Class for TextAttack."""

    def __init__(
        self,
        attack: "attack_recipes.AttackRecipe",
    ):
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
                    if (
                        _cls == attack_recipes.AttackRecipe
                        and possible_recipe != attack_recipes.AttackRecipe
                    ):
                        recipes.append(possible_recipe_name)
        return recipes

    def apply(
        self,
        batch: DataPanel,
        columns: List[str],
        skeleton_batches: List[DataPanel],
        slice_membership: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[List[DataPanel], np.ndarray]:

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
    def from_recipe(cls, recipe: str, model: "wrappers.ModelWrapper"):
        return cls(attack=getattr(attack_recipes, recipe).build(model=model))

    @classmethod
    def wrap_huggingface_model(cls, model: Model) -> "wrappers.ModelWrapper":
        return wrappers.HuggingFaceModelWrapper(
            model=model.model,
            tokenizer=model.tokenizer,
        )
