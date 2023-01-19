from typing import Tuple, Callable, Iterable, Any

from datasets import load_from_disk
import os
import joblib
from PIL import Image
from torch import Tensor

import shutil
import requests
import csv

from magicarp.pipeline import Pipeline, register_datapipeline

@register_datapipeline
class BestPromptsPipeline(Pipeline):
	"""
    Pipeline for the data set from the paper 
    "Best Prompts for Text-to-Image Models and How to Find Them"
    """
	def __init__(self, prompt_path : str, preference_path : str):
		super().__init__()
		self.prompts = []

		with open(prompt_path, encoding='utf-8') as csvfile:
			spamreader = csv.reader(csvfile)

			skip_first = True

			for row in spamreader:
				if skip_first:
					skip_first = False
					continue

				self.prompts.append({"prompt": row[0], "examples": []})

		with open(preference_path, encoding='utf-8') as csvfile:
			spamreader = csv.reader(csvfile)

			skip_first = True

			for row in spamreader:
				if skip_first:
					skip_first = False
					continue

				self.prompts[int(row[0])]["examples"].append((row[1], row[2]) if row[4] == "left" else (row[2], row[1]))

		self.prep : Callable[[Iterable[str], Iterable[str]], TextElement] = None

	def create_preprocess_fn(self, call_feature_extractor : Callable):
		# call_feature_extractor(img : Iterable[PIL.Image.Image], txt : Iterable[str]) -> Tuple[Something1, Something2]
		def prep(
			batch_prompt : Iterable[str],
			batch_chosen : Iterable[str],
			batch_rejected : Iterable[str]
		) -> Tuple[TextElement, TextElement]:
			prompt_tok_out, chosen_pixel_vals = call_feature_extractor(batch_prompt, batch_chosen) 
			_, rejected_pixel_vals = call_feature_extractor(batch_prompt, batch_rejected)

			prompt_elem = TextElement(
				input_ids = prompt_tok_out.input_ids,
				attention_mask = prompt_tok_out.attention_mask
			)

			chosen_elem = ImageElement(
				pixel_values = chosen_pixel_vals
			)
			rejected_elem = ImageElement(
				pixel_values = rejected_pixel_vals
			)

			a_elem = DataElement.concatenate([prompt_elem, prompt_elem])
			b_elem = DataElement.concatenate([chosen_elem, rejected_elem])

			return a_elem, b_elem

		self.prep = prep

	def query(self, pair: Tuple[str, str], idx: int):
		idx_good = idx % 4
		idx_bad = int((idx - idx % 4)/4)
		print("wee query")
		good_image_url = f'https://storage.yandexcloud.net/diffusion/{pair[0]}_{idx_good}.png'
		bad_image_url = f'https://storage.yandexcloud.net/diffusion/{pair[1]}_{idx_bad }.png'

		good_img_req = requests.get(good_image_url, stream=True)
		bad_img_req = requests.get(bad_image_url, stream=True)

		with open(f'{pair[0]}_{idx_good}.png', 'wb') as out_file:
			shutil.copyfileobj(good_img_req.raw, out_file)

		with open(f'{pair[1]}_{int((idx - idx%4)/4)}.png', 'wb') as out_file:
			shutil.copyfileobj(bad_img_req.raw, out_file)

	def get_img_paths(self, pair: Tuple[str, str], idx: int):
		idx_good = idx % 4
		idx_bad = int((idx - idx % 4)/4)
		if (not os.path.exists(f'{pair[0]}_{idx_good}.png')) or (not os.path.exists(f'{pair[1]}_{idx_bad }.png')):
			self.query(pair, idx)
    	
		return (
			Image.open(f'{pair[0]}_{idx_good}.png'),
			Image.open(f'{pair[1]}_{idx_bad}.png')
			)


	def __getitem__(self, index: int) -> Tuple[str, str]:
		idx = index % 16
		remainder = int((index - idx) / 16)

		i = 0

		while remainder >= len(self.prompts[i]["examples"]):
			index += 1
			remainder -= len(self.prompts[i]["examples"])

		return self.get_img_paths(self.prompts[i]["examples"][remainder], idx)

	def __len__(self) -> int:
		return sum([len(elem["examples"]) * 16 for elem in self.prompts])
        

