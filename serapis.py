import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='jax')
warnings.filterwarnings("ignore", module='PIL')

from transformers import ResNetForImageClassification
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import logging
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import urllib.parse
import argparse
import requests
import asyncio
import torch
import sys
import os
import io

async def call_serpapi(target, api_key, ijn: 0):
	q = urllib.parse.quote(target["q"])
	url = "https://serpapi.com/search.json?api_key={}&tbm=isch&q={}&ijn={}".format(api_key, q, ijn)
	if "chips" in target:
		chips = None

		# First call to retrieve chips from the search
		response = requests.get(url)
		results = response.json()
		if "error" in results:
			print(results["error"])
			sys.exit(1)

		if "chips" in suggestion and suggestion["name"] == target["chips"]:
			if "suggested_searches" in results:
				for suggestion in results["suggested_searches"]:
					chips = urllib.parse.quote(suggestion["chips"])

		if chips != None:
			# Second call to make a chips search
			url = "https://serpapi.com/search.json?api_key={}&tbm=isch&q={}&chips={}&ijn={}".format(api_key, q, chips, ijn)
			response = requests.get(url)
	else:
		# Call without chips
		response = requests.get(url)
	return response.json()

async def search(targets, api_key):
	images_results = []
	for target in targets:
		searches = []
		# First page results
		searches.append(call_serpapi(target, api_key, ijn=0))
		if "page" in target and target["page"] > 1:
			# If more than one page is requested
			[searches.append(call_serpapi(target, api_key, ijn=i)) for i in range(target["page"]) if i != 0]
		images = await asyncio.gather(*searches)
		images_results.append([target["q"], images])
	return images_results

async def get_single_image_url(label, url, df):
	try:
		response = requests.get(url, timeout=2)
		f = io.BytesIO(response.content)

		last_item = len(os.listdir("images"))
		image_path = "images/{}.png".format(last_item)

		im = Image.open(f)
		im.save(image_path, format='PNG', quality=95)

		df.loc[0] = [label, image_path]
		print("Downloaded {}".format(url))
		return df
	except:
		return df

async def get_images(images_results, limit = None):
	all_dfs = []
	for images in images_results:
		calls = []
		if len(images) == 2:
			for page_result in images[1]:
				if "images_results" in page_result:
					for i in range(len(page_result["images_results"])):
						result = page_result["images_results"][i]
						if limit != None and i == limit:
							break

						if "original" in result:
							df = pd.DataFrame(columns=["label", "image_path"])
							url = result["original"]
							calls.append(get_single_image_url(images[0], url, df))
							print("Added Coroutine: {} - {}".format(images[0], url, df))
		dfs = await asyncio.gather(*calls)
		all_dfs = all_dfs + dfs
	df = pd.concat(all_dfs)
	print("---")
	return df


class CustomDataset(Dataset):
	def __init__(self, df):
		self.df = df
			
	def __len__(self):
		return len(self.df)

	def preprocess_image(self, image_path):
		# Load image using PIL
		image = Image.open(image_path)
		# Convert image to RGB if it is not already in that format
		if image.mode != 'RGB':
				image = image.convert('RGB')
		# Resize image to a fixed size
		image = image.resize((224, 224))
		# Convert image to a tensor
		image = transforms.ToTensor()(image)
		# Normalize image
		image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
		return image
	
	def __getitem__(self, index):
		row = self.df.iloc[index]
		label = row['label']
		image_path = row['image_path']
		# Load and pre-process image data here
		image_data = self.preprocess_image(image_path)
		return image_data, label

def train_and_save_model(df, target_labels):
	print("Training a new model. If you get a warning about shapes below, you can ignore it.")
	dataset = CustomDataset(df)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
	class_to_idx = {class_name: idx for idx, class_name in enumerate(target_labels)}
	idx_to_class = {idx: class_name for idx, class_name in enumerate(target_labels)}
	model = ResNetForImageClassification.from_pretrained(
		"microsoft/resnet-50",
		num_labels=len(target_labels),
		id2label=idx_to_class,
		label2id=class_to_idx,
		ignore_mismatched_sizes=True
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	total_examples = len(dataset)
	num_epochs = 2
	processed_examples = 0

	print("---")
	for epoch in range(num_epochs):
		for data in dataloader:
			images, labels = data
			model.train()
			model.zero_grad()
			
			logits = model(images).logits
			labels = [class_to_idx[label] for label in labels if label in class_to_idx]
			labels = torch.tensor(labels)
			
			loss = F.cross_entropy(logits, labels)
			loss.backward()
			optimizer.step()

			model.eval()
			logits = model(images, labels = labels).logits
			_, predicted_labels = logits.max(dim=1)
			accuracy = (predicted_labels == labels).float().mean()
			
			batch_size = images.size(0)
			processed_examples += batch_size
			progress = (processed_examples / (total_examples * num_epochs)) * 100
			print(f'Accuracy: {accuracy} | Progress: {progress:.2f}%')

	last_item = len(os.listdir("models"))
	model_path = "models/{}.pth".format(last_item)
	torch.save(model.state_dict(), model_path)
	df = pd.DataFrame(columns=["model_path", "target_labels"])
	df.loc[0] = [model_path, "--".join(target_labels)]
	old_df = pd.read_csv("models/old_models.csv")
	new_df = pd.concat([df, old_df], ignore_index=True)
	new_df.to_csv("models/old_models.csv", index=False)
	
	return model

def train_a_new_model(targets=None, use_catalogue=True, api_key=None, limit=None):
	target_labels = [dictionary["q"] for dictionary in targets]
	if use_catalogue:
		# Call the Catalogue CSV and get only the keys you targeted.
		df = pd.read_csv("images/catalogue.csv")
		all_labels = list(set(df["label"]))
		for label in all_labels:
			if label not in target_labels:
				df = df.drop(df[df["label"] == label].index)
	else:
		# Save the new images to Catalogue CSV but use only the ones you targeted.
		images_results = asyncio.run(search(targets, api_key))
		df = asyncio.run(get_images(images_results, limit = limit))
		old_df = pd.read_csv("images/catalogue.csv")
		new_df = pd.concat([df, old_df], ignore_index=True)
		new_df.to_csv("images/catalogue.csv", index=False)
	
	model = train_and_save_model(df, target_labels)
	return model, target_labels

def use_old_model(model_path=None):
	model_df = pd.read_csv("models/old_models.csv")
	target_labels = model_df.loc[model_df[model_df["model_path"] == model_path].index]["target_labels"].iloc[0]
	target_labels = target_labels.split("--")
	class_to_idx = {class_name: idx for idx, class_name in enumerate(target_labels)}
	idx_to_class = {idx: class_name for idx, class_name in enumerate(target_labels)}
	model = ResNetForImageClassification.from_pretrained(
		"microsoft/resnet-50",
		num_labels=len(target_labels),
		id2label=idx_to_class,
		label2id=class_to_idx,
		ignore_mismatched_sizes=True
		)
	state_dict = torch.load(model_path)
	model.load_state_dict(state_dict)
	return model, target_labels

def predict_image(image_path=None, model=None, target_labels=None):
	df = pd.DataFrame(columns=["label", "image_path"])
	dataloader = CustomDataset(df)
	image = dataloader.preprocess_image(image_path)
	model.eval()
	logits = model(image.unsqueeze(0)).logits
	_, predicted_labels = logits.max(dim=1)
	predicted_class_names = [target_labels[int(label)] for label in predicted_labels]
	return predicted_class_names[0]

def questions():
	api_key = ""
	model_path = ""
	train_new_model = False
	use_catalogue = True
	limit = None
	targets = []
	while True:
		print("What do you want to do?")
		print("1. Train a new model.")
		print("2. Use an old model.")
		choice = input("Enter your choice: ")
		if choice == "1" or choice == "2":
			break
		else:
			print("Please enter a valid choice.")
	print("---")

	if choice == "1":
		train_new_model = True
		while True:
			print("Would you like to use the old images you have stored, or scrape fresh images?")
			print("1. Use old images.")
			print("2. Scrape new images.")
			second_choice = input("Enter your choice: ")
			if second_choice == "1" or second_choice == "2":
				break
			else:
				print("Please enter a valid choice.")
		print("---")

		if second_choice == "1":
			use_catalogue = True
			old_df = pd.read_csv("images/catalogue.csv")
			all_labels = list(set(old_df["label"]))
			while True:
				print("Here are the labels of images you have stored.")
				print("{}".format(", ".join(all_labels)))
				print("Which labels do you want to use?")
				desired_labels = input("Enter the labels (case sensitive) separated by a comma: ")
				desired_labels = desired_labels.split(",")
				desired_labels = [label.strip() for label in desired_labels if label.strip != ""]
				if len(desired_labels) < 2:
					print("Please enter at least two labels.")
				elif not all(elem in all_labels for elem in desired_labels):
					print("Please enter valid labels that are already in the labels of images you have stored.")
				elif all(elem in all_labels for elem in desired_labels):
					break
			print("---")
		elif second_choice == "2":
			use_catalogue = False
			while True:
				print("Which labels do you want to use?")
				desired_labels = input("Enter the labels (case sensitive) separated by a comma: ")
				desired_labels = desired_labels.split(",")
				desired_labels = [label.strip() for label in desired_labels if label.strip != ""]
				if len(desired_labels) < 2:
					print("Please enter at least two labels.")
				else:
					break
			print("---")

			while True:
				print("How many images do you want to scrape at most for each label?")
				limit = input("Enter the limit (Enter nothing to pass): ")
				if limit == "":
					break
				elif limit.isdigit():
					limit = int(limit)
					break
				else:
					print("Please enter a valid integer.")
			print("---")

			while True:
				api_key = input("Enter your SerpApi API key: ")
				if api_key == "":
					print("Please enter a valid API key.")
				else:
					break
			print("---")
		targets = [{"q": label} for label in desired_labels]
	elif choice == "2":
		while True:
			print("Which model do you want to use?")
			model_df = pd.read_csv("models/old_models.csv")
			print(model_df)
			model_path = input("Enter the model path: ")
			if model_path not in list(model_df["model_path"]):
				print("Please enter a valid model path.")
			elif not os.path.isfile(model_path):
				print("The model exists in CSV, but there is no model at the path. Please enter a valid model path.")
			else:
				break
		print("---")
	
	while True:
		image_path = input("Enter the image path you want to predict: ")
		if not os.path.isfile(image_path):
			print("Please enter a valid image path.")
		else:
			break
	
	print("---")
	return targets, train_new_model, use_catalogue, api_key, model_path, limit, image_path

logging.set_verbosity_error()
parser = argparse.ArgumentParser()

# Mode
parser.add_argument('--train', action='store_true', help='Whether to train a new model')
parser.add_argument('--model-path', type=str, help='Pretrained Model path you want to use')
parser.add_argument('--dialogue', action='store_false', help='Whether to use dialogue to navigate through the program')

# Training
parser.add_argument('--use-old-images', action='store_true', help='Whether to use old images you have downloaded to train a new model')
parser.add_argument('--api-key', type=str, help='SerpApi API Key')
parser.add_argument('--limit', type=int, help='Number of images you want to scrape at most for each label')
parser.add_argument('--labels', type=str, nargs='+', help='Labels you want to use to train a new model')

# Prediction
parser.add_argument('--image-path', type=str, help='Path to the image you want to classify')

args = parser.parse_args()

# New Training with Image Scraping
if args.train and args.labels and args.image_path and not args.model_path:
	if args.limit:
		limit = args.limit
	else:
		limit = None

	if not args.api_key:
		print("You need to enter your SerpApi API key to scrape new images.")
		sys.exit(1)

	labels = [label.replace(",","").strip() for label in args.labels if label.replace(",","").strip() != ""]
	targets = [{"q": label} for label in labels]

	train_new_model = True
	use_catalogue = False
	api_key = args.api_key
	image_path = args.image_path

	model, target_labels = train_a_new_model(targets, use_catalogue, api_key, limit)
	print("---")
	print("The image contains {}".format(predict_image(image_path, model, target_labels)))

# New Training without Old Scraped Images
elif args.train and args.labels and args.use_old_images and args.image_path and not args.model_path:
	labels = [label.replace(",","").strip() for label in args.labels if label.replace(",","").strip() != ""]
	targets = [{"q": label} for label in labels]
	train_new_model = True
	use_catalogue = True
	api_key = None
	limit = None
	image_path = args.image_path

	old_df = pd.read_csv("images/catalogue.csv")
	all_labels = list(set(old_df["label"]))
	if len(labels) < 2:
		print("Please enter at least two labels.")
		sys.exit(1)
	elif not all(elem in all_labels for elem in labels):
		print("Please enter labels that are in the catalogue.")
		sys.exit(1)

	model, target_labels = train_a_new_model(targets, use_catalogue, api_key, limit)
	print("---")
	print("The image contains {}".format(predict_image(image_path, model, target_labels)))

# Old Model Prediction
elif args.model_path and args.image_path and not args.train:
	model_path = args.model_path
	image_path = args.image_path

	if not os.path.isfile(model_path):
		print("Please enter a valid model path.")
		sys.exit(1)

	model, target_labels = use_old_model(model_path)
	print("---")
	print("The image contains {}".format(predict_image(image_path, model, target_labels)))

# Actions from Dialogue
elif args.dialogue and not args.model_path and not args.image_path and not args.train:
	targets, train_new_model, use_catalogue, api_key, model_path, limit, image_path = questions()

	if train_new_model:
		model, target_labels = train_a_new_model(targets, use_catalogue, api_key, limit)
	else:
		model, target_labels = use_old_model(model_path)

	print("---")
	print("The image contains {}".format(predict_image(image_path, model, target_labels)))
else:
	print("Please enter the correct arguments.")
	sys.exit(1)

# Tips for Advanced Usage
# Below is an example of how to target specific images.
#
#targets = [
#  {
#    "q": "Elephant",
#    "page": 10,
#    "chips": "male"
#  }
#]
# `q` stands for the query you want to make to
# SerpApi's Google Images Scraper API.
#
# `page` stands for how many pages you want to
# scrape. Each page has 100 images. Not all images
# are usable for training. But it will download a
# lot of images enough for you to finetune ResNet50.
#  
# `chips` stands for the chips you want to add to
# the query. Chips are the labels you want to add
# to the query. For example, if you want to target
# male elephants only, you can use the chips below.
# The script will make a double call to create a
# chips search. The chips can be found on top of
# the page just below the search bar.
#
# You can tweak the code to insert manual targets
# to the program.
