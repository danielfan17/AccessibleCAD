import os
import math
import shutil
import pandas as pd

categories = ["bear", "bird", "butterfly", "cat", "cow", "dog", "fish", "horse", "sheep",
"air conditioner", "blender", "dishwasher", "microwave", "rice cooker", "toaster",
"washer", "dryer", "backpack", "purse", "ball", "basket", "battery", "board", "book",
"bottle", "bucket", "cage", "calculator", "candle", "canister", "cassette", "standing clock",
"table clock", "wall clock", "cap", "hat", "pants", "shirt", "shoes", "coaster", "coin",
"computer mouse", "controller", "cutting ,board", "door", "camera", "computer", "ipad", 
"monitor", "printer", "radio", "tv", "fan", "apple", "banana", "bread slice", "cake", "carrot",
"cereal box", "cookie", "cup cake", "donut", "box", "egg", "fries", "fruit bowl", "orange",
"pizza", "sandwich", "bed", "bookcase", "cabinet", "mattress", "bench", "chair", "couch",
"stool", "counter", "desk", "table", "glasses", "flashlight", "hammer", "scissors", "screw driver",
"hanger", "headphones", "jar", "jug", "keyboard", "bowl", "cup", "mug", "teacup", "wine glass",
"fork", "knife", "spoon", "pan", "plate", "pot", "ladder", "lamp", "light switch", "media discs",
"mouse pad", "guitar", "piano", "trumpet", "violin", "paper box", "person", "phone",
"pill bottle", "pillow", "plant", "bathtub", "faucet", "showerhead", "sink", "toilet", "power strip",
"railing", "ring", "rubik", "rug", "soda can", "speaker", "eraser", "notepad", "paper",
"paper clip", "ruler", "stapler", "tape dispenser", "tape measure", "thumbtack", "suitcase",
"tissue box", "shampoo", "soap bar", "soap bottle", "toilet paper", "toothbrush", "trash bin",
"usbstick", "umbrella", "vase", "car", "bus", "boat", "truck", "wallet", "watch", "gun", "sword", "pencil"]

counter = 0

# input reference
df = pd.read_csv("data/ShapeNetSem.csv")


# for each file in folder
for filename in os.listdir("data/ShapeNetSem"):

	# if file is not a folder
	if not os.path.isdir("data/ShapeNetSem/" + filename):

		# parse name
		file_id = filename.replace('.obj', '')

		# retrieve index value of matching filename to entry
		entry_index = df.index[df['fullId'].str.replace('wss.', '') == file_id].tolist()

		# if index found/ not empty
		if entry_index: 

			entry_index = entry_index[0]

		# retrieve categories of files based on index
		file_categories = df['category'][entry_index]

		# if entry has label
		if isinstance(file_categories, str):
		
			# split file categories into list
			#file_categories = file_categories.lower().split(",")

			# to lower case
			file_categories = file_categories.lower()

			# loop through each category
			for idx, category in enumerate(categories):

				# if category a substring of file_categories
				if category in file_categories or category == file_categories:

					# check if category folder does not exist
					if not os.path.exists("data/ShapeNetSem/" + category):

						# make folder
						os.makedirs("data/ShapeNetSem/" + category)

					# move file into category folder
					shutil.move("data/ShapeNetSem/" + filename, "data/ShapeNetSem/" + category + "/" + filename)

					# update progress
					counter += 1

					# print progress
					print(counter)

					# break through category-matching for loop
					break




