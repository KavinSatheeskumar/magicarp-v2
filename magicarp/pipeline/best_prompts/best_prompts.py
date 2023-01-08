from typing import Tuple, Callable, Iterable, Any
from torchtyping import TensorType

from datasets import load_from_disk
import os
import joblib
from torch import Tensor

from magicarp.pipeline import Pipeline, register_datapipeline
from magicarp.data import TextElement

@register_datapipeline
class BestPromptsPipeline(Pipeline):
	"""
    Pipeline for the data set from the paper 
    "Best Prompts for Text-to-Image Models and How to Find Them"
    """
	def __init__(self, prompt_path : str, preference_path : str):
		super().__init__()
		self.prompts = {}

		with open(prompts_path) as csvfile:
			spamreader = csv.reader(csvfile)

			skip_first = True

			for row in spamreader:
				if skip_first:
					skip_first = False
					continue

				self.prompts.append({"prompt": row[0], "examples": []})

		with open(preference_path) as csvfile:
			spamreader = csv.reader(csvfile)

			skip_first = True

			for row in spamreader:
				if skip_first:
					skip_first = False
					continue

				self.prompts[int(row[0])]["examples"].append(if row[4] == "left" (row[1], row[2]) else (row[2], row[1]))

		self.prep : Callable[[Iterable[str], Iterable[str]], TextElement] = None

	def query(self, pair: Tuple[str, str], idx: int):
		pass

	def __getitem__(self, index: int) -> Tuple[str, str]:
		idx = idx % 16
		remainder = (index - query) / 16

		index = 0

		while remainder >= len(self.prompts[index]["examples"]):
			index += 1
			remainder -= len(self.prompts[index]["examples"])

		return self.query(self.prompts[index]["examples"][remainder], idx)

	def __len__(self) -> int:
		sum([len(elem["examples"]) * 16 for elem in self.prompts])
        

