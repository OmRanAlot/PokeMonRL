import os
import glob
# from data.parseData import item_idx
import torch
from torch.utils.data import DataLoader, Dataset

# Categorical ids nn.Embedding / CrossEntropy need as int64 (long).
# preprocess.py stores them as uint16/int8 to keep .pt shards small.
_POKEMON_IDX_KEYS = (
    "species_idx",
    "item_idx",
    "ability_idx",
    "status_idx",
    "tera_idx",
    "type_idxs",
    "move_idxs",
)
_FIELD_IDX_KEYS = (
    "format_idx",
    "weather_idx",
    "terrain_idx",
    "player_prev_move_idx",
    "opp_prev_move_idx",
)


def pack_pokemon_group(feats: dict) -> dict:
    """Rebuild PokemonEncoder kwargs from compact preprocess tensors.
    preprocess splits numeric_feats into hp_pct/boosts/base_stats/move_pp
    with mixed dtypes. The encoder still takes one float32 vector of length 18.
    """
    packed = {key: feats[key].long() for key in _POKEMON_IDX_KEYS}
    # unsqueeze hp_pct so it cats on the last dim: [N] -> [N, 1], [N, 5] -> [N, 5, 1]
    packed["numeric_feats"] = torch.cat(
        [
            feats["hp_pct"].to(torch.float32).unsqueeze(-1),
            feats["boosts"].to(torch.float32),
            feats["base_stats"].to(torch.float32),
            feats["move_pp"].to(torch.float32),
        ],
        dim=-1,
    )
    return packed


def pack_field_feats(field_feats: dict) -> dict:
    """Cast field ids to long and hazards/scalars to float32 for FieldEncoder cat()."""
    packed = {key: field_feats[key].long() for key in _FIELD_IDX_KEYS}
    # stored as bool / float16; embeddings are float32 so cat() needs matching dtype
    packed["hazards"] = field_feats["hazards"].to(torch.float32)
    packed["scalars"] = field_feats["scalars"].to(torch.float32)
    return packed


def pack_compact_batch(batch: dict) -> dict:
    """Widen one mini-batch from .pt storage dtypes to encoder/loss dtypes.

    Call this AFTER slicing, not on the full 20k-row chunk, so CPU/GPU RAM
    only hold long/float32 for mini_batch_size rows at a time.
    """
    return {
        "field_feats": pack_field_feats(batch["field_feats"]),
        "mine_active": pack_pokemon_group(batch["mine_active"]),
        "mine_bench": pack_pokemon_group(batch["mine_bench"]),
        "opp_active": pack_pokemon_group(batch["opp_active"]),
        "opp_revealed": pack_pokemon_group(batch["opp_revealed"]),
        "action_mask": batch["action_mask"],  # bool is what masked_fill(~mask) wants
        "action_label": batch["action_label"].long(),  # CrossEntropyLoss requires long
    }

#Loads the data itself and passes it to the NN
class FastDataLoader(DataLoader):
    def __init__(self, data_dir="/content/processed_data/", split="train", split_ratio=0.9):

        all_files = sorted(glob.glob(str(data_dir + "*.pt")))

        print("file 0: ", all_files[0])

        split_idx = int(len(all_files) * split_ratio)

        self.train_files = all_files[:split_idx]
        self.eval_files = all_files[split_idx:]

        if split=="train":
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu", weights_only=True)


# Opens the files and gets the data itself.
# Returns compact preprocess tensors as saved; pack_compact_batch widens them later.
class FastDataSet(Dataset):
    def __init__(self, files):
        self.files=files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu", weights_only=True)


