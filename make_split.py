import random

# to make all_files.txt, run: ls my_dir/with/corrected/compressed/.root > all_files.txt
# or something like that

# Load file paths from all_files.txt
with open("all_files.txt", "r") as f:
    files = [line.strip() for line in f]

# Shuffle the list
random.shuffle(files)

# Split into train and validation
split_ratio = 0.8
split_idx = int(len(files) * split_ratio)

train_files = files[:split_idx]
val_files = files[split_idx:]

# Write train file list
with open("train_files.txt", "w") as f:
    for line in train_files:
        f.write(line + "\n")

# Write validation file list
with open("val_files.txt", "w") as f:
    for line in val_files:
        f.write(line + "\n")

# Might want to add a test set later