
from pathlib import Path

data_dir = Path(r"c:\Users\Thana\dev\research_models\mel_spectrograms")
files = list(data_dir.glob("*.png"))
prefixes = set()
for f in files:
    prefix = f.name.split('_')[0]
    prefixes.add(prefix)

print("Found prefixes:", prefixes)
