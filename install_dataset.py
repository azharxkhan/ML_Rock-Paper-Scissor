import os
import subprocess

# Step 1: Ensure kaggle is installed
try:
    import kaggle
except ImportError:
    print("Kaggle CLI not found. Installing...")
    subprocess.run(["pip", "install", "kaggle"])

# Step 2: Ensure ~/.kaggle directory exists
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Step 3: Check if kaggle.json exists
if not os.path.exists(f"{kaggle_dir}/kaggle.json"):
    print("\nPlease download kaggle.json from https://www.kaggle.com/account")
    print(f"Then place it in {kaggle_dir}/kaggle.json and rerun this script.")
    exit(1)

# Step 4: Set proper permissions
os.chmod(f"{kaggle_dir}/kaggle.json", 0o600)

# Step 5: Download datasets
print("\nDownloading Rock Paper Scissors datasets...")
datasets = [
    "drgfreeman/rockpaperscissors",
    "gpiosenka/rock-paper-scissors-image-dataset"
]

for dataset in datasets:
    print(f"Downloading {dataset}...")
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset, "-p", "data/kaggle_dataset", "--unzip"
    ])

print("\nDataset download complete.")