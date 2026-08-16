"""
Extracts all the data from the .gz file into .lz4 files for access
"""


import json, lz4.frame, tarfile

filename = "gen9ou.tar.gz"

# Use 'r' for uncompressed, 'r:gz' for .tar.gz, 'r:bz2' for .tar.bz2
with tarfile.open(f"data/{filename}", "r:gz") as tar:
    tar.extractall(path="./data")

print("done")