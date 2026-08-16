"""
Split Data into shards for easier memory management when training

"""

import tarfile, os,io

def reshard_tar(src_tar_gz, out_dir, files_per_shard=2000):

    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(src_tar_gz, "r:gz") as src:
        shard_idx = 0
        count = 0
        out_tar = tarfile.open(f"{out_dir}/shard_{shard_idx:05d}.tar", "w")  # no gzip on the shards
        for member in src:

            if not member.isfile():
                continue

            f = src.extractfile(member)
            data = f.read()
            obj = io.BytesIO(data)

            out_tar.addfile(member, fileobj=obj)  # placeholder, see note below
            count += 1
            if count >= files_per_shard:
                out_tar.close()
                shard_idx += 1
                count = 0
                out_tar = tarfile.open(f"{out_dir}/shard_{shard_idx:05d}.tar", "w")
                print(f"Creating new shard {shard_idx}!")    
        
        if count > 0:
            out_tar.close()

if __name__ == "__main__":
    from pathlib import Path  
    BASE_DIR = Path(__file__).resolve().parent.parent

    filename = "gen9ou.tar.gz"
    # C:\Users\omran\code\PokemonRL\data\data\gen9ou.tar.gz
    DATA_DIR = BASE_DIR / "data"/"data" / filename
    print(DATA_DIR)

    OUT_DIR= BASE_DIR / "data"/"data" / "shard"


    reshard_tar(DATA_DIR, OUT_DIR, 20000)

