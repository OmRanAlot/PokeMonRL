"""
Download initial pre-enviroment data from

https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays 

"""

import pandas as pd
import numpy as np
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="jakegrigsby/metamon-parsed-replays",
    repo_type="dataset",
    allow_patterns=["gen9ou.tar.gz"],
    revision="v6",
    local_dir=".\data"
)

print("\n")
print('='*50)
print(path)
print('='*50)

