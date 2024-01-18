from pathlib import Path

ROOT_DIR = "/mnt/SSD1/rabin/datasets/phrasecut/masks/"

unique_pairs = set()
non_unique_count = 0
for file in Path(ROOT_DIR).glob("*.png"):
    task, rest = file.stem.split("-", 1)

    if rest in unique_pairs:
        non_unique_count += 1
        print(file)
    else:
        unique_pairs.add(rest)

print("Non-unique image phrase pairs number:", non_unique_count)
