import os
# Tell KaggleHub to cache everything inside ./data
os.environ["KAGGLEHUB_CACHE"] = os.path.abspath("data")

import kagglehub
dataset_root = kagglehub.dataset_download("drgfreeman/rockpaperscissors")

# Inside that cache KaggleHub keeps the original dataset folder name,
# so the class folders are at:
train_dir = os.path.join(dataset_root, "rockpaperscissors")
print("Dataset ready at:", train_dir)
