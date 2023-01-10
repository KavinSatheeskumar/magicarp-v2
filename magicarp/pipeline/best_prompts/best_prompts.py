from typing import Tuple, Callable, Iterable, Any
from torchtyping import TensorType

from datasets import load_from_disk
import os
import joblib
from torch import Tensor

import shutil
import requests

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
		good_image_url = f'https://storage.yandexcloud.net/diffusion/{pair[0]}_{idx%4}.png'
		bad_image_url = f'https://storage.yandexcloud.net/diffusion/{pair[0]}_{(idx - idx%4)/4}.png'

		good_img_req = requests.get(good_image_url, stream=True)
		bad_img_req = requests.get(bad_image_url, stream=True)

		with open(f'{pair[0]}_{idx%4}.png', 'wb') as out_file:
    		shutil.copyfileobj(good_img_req.raw, out_file)

    	with open(f'{pair[0]}_{(idx - idx%4)/4}.png', 'wb') as out_file:
    		shutil.copyfileobj(bad_img_req.raw, out_file)

    def get_img_paths(self, pair: Tuple[str, str], idx: int):
    	if (not os.path.exists(f'{pair[0]}_{idx%4}.png')) or (not f'{pair[0]}_{(idx - idx%4)/4}.png'):
    		self.query(pair, idx)
    	
    	return (f'{pair[0]}_{idx%4}.png', f'{pair[0]}_{(idx - idx%4)/4}.png')


	def __getitem__(self, index: int) -> Tuple[str, str]:
		idx = idx % 16
		remainder = (index - query) / 16

		index = 0

		while remainder >= len(self.prompts[index]["examples"]):
			index += 1
			remainder -= len(self.prompts[index]["examples"])

		return self.get_img_paths(self.prompts[index]["examples"][remainder], idx)

	def __len__(self) -> int:
		sum([len(elem["examples"]) * 16 for elem in self.prompts])
        

