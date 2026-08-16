import json
import random
import glob
import torch
import webdataset as wds
from pathlib import Path
import os, sys
import lz4.frame

class Vocab:
    """
    Wraps the flat DefaultObservationSpace-v1.json token->idx map. 
    Every category 
    (species, moves, items, abilities, statuses, types, weather/field, format tags)
    shares ONE index space 
    """

    def __init__(self, path="C:\\Users\\omran\\code\\PokemonRL\\data\\DefaultObservationSpace-v1.json"):
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
        print(root_dir)
        with open(path, "r") as f:
            self.token_to_idx = json.load(f)
        self.BLANK = self.token_to_idx["<blank>"]  # fallback for anything not in vocab

    def get(self, token):
        if token is None:
            return self.BLANK
        return self.token_to_idx.get(token, self.BLANK)

    def format_token(self, fmt):
        return self.get(f"<{fmt}>")

class DataLoader():
    """
    Outputs the processed data so I can pass it into the neural network
    """
    def __init__(self, split=0.9, subset_fraction=None):
        """
        Initalize training and validation data
        Set up constants
        """
        #load vocab
        self.vocab = Vocab()

        # Shards live at <repo>/data/data/shard (this file is bot/data/loadData.py).
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.shard_dir = repo_root / "data" / "data" / "shard"

        all_shards = sorted(glob.glob(str(self.shard_dir / "*.tar")))
        if not all_shards:
            raise FileNotFoundError(f"No shard .tar files found in {self.shard_dir}")
        random.seed(42)
        random.shuffle(all_shards) #randomize the shards for test/train split

        n = len(all_shards)

        #temp
        # split = 0.002
        self.train_shards = all_shards[:int(split*n)]
        self.val_shards = all_shards[int(split*n):]

        if subset_fraction is not None:
            n_train = max(1, int(len(self.train_shards) * subset_fraction))
            n_val = max(1, int(len(self.val_shards) * subset_fraction))
            self.train_shards = self.train_shards[:n_train]
            self.val_shards = self.val_shards[:n_val]

        # constants 

        self.MOVE_SLOTS = 4
        self.BENCH_SLOTS = 5
        # 0-3 moves, 4-8 switches, 9-12 tera-moves (all alphabetical)
        self.ACTION_DIM = self.MOVE_SLOTS + self.BENCH_SLOTS + self.MOVE_SLOTS
        self.HAZARD_TOKENS = ["spikes", "stealthrock", "toxicspikes", "stickyweb"]
        self.NUMERIC_DIM = 1 + 7 + 6 + self.MOVE_SLOTS  # hp_pct + boosts(7) + base_stats(6) + per-move pp_fraction

    def battle_to_example(self, sample):
        """
        Turn an individual battle file into examples
        """
        # print("SAMPKE")
        # print("\n"*2)
        # print(sample)

        compressed = sample["json.lz4"] # raw compressed information

        battle = json.loads(lz4.frame.decompress(compressed))
        states, actions = battle["states"], battle["actions"]
        for s, a in zip(states, actions):
            example = self.state_to_tensors(s)
            example["action_label"] = torch.tensor(
                self._clamp_action(a, example["action_mask"]),
                dtype=torch.long,
            )
            yield example

    def unknown_mon_dict(self) -> dict:
        """
        For unknown opponent pokemon
        """
        blank = self.vocab.BLANK
        return {
            "species_idx": torch.tensor(blank, dtype=torch.long),
            "item_idx": torch.tensor(blank, dtype=torch.long),
            "ability_idx": torch.tensor(blank, dtype=torch.long),
            "status_idx": torch.tensor(blank, dtype=torch.long),
            "tera_idx": torch.tensor(blank, dtype=torch.long),
            "type_idxs": torch.tensor([blank, blank], dtype=torch.long),
            "move_idxs": torch.tensor([blank] * self.MOVE_SLOTS, dtype=torch.long),
            "numeric_feats": torch.zeros(self.NUMERIC_DIM, dtype=torch.float32),
        }
    
    def mon_to_index_dict(self, mon_json) -> dict:
        """
        Build a pokemon tensor from the vocab
        Just the indicies from the vocab(no processing or embeddings)
        """
        if mon_json is None:
            return self.unknown_mon_dict()

        types = mon_json["types"].split()
        types = (types + [None, None])[:2]

        # action slots 0-3 / 9-12 are alphabetical by move name
        moves = sorted(mon_json["moves"], key=lambda m: m["name"])[:self.MOVE_SLOTS]
        move_idxs = self._pad_or_trim(
            [self.vocab.get(m["name"]) for m in moves],
            self.MOVE_SLOTS,
            self.vocab.BLANK,
        )
        move_pp_frac = self._pad_or_trim(
            [m["current_pp"] / m["max_pp"] if m["max_pp"] > 0 else 0.0 for m in moves],
            self.MOVE_SLOTS,
            0.0,
        )


        boosts = [mon_json[k] for k in (
                "atk_boost", "spa_boost", "def_boost", "spd_boost",
                "spe_boost", "accuracy_boost", "evasion_boost",
        )]
        base_stats = [mon_json[k] for k in (
            "base_atk", "base_spa", "base_def", "base_spd", "base_spe", "base_hp",
        )]

        numeric_feats = torch.tensor(
            [mon_json["hp_pct"]] + boosts + base_stats + move_pp_frac,
            dtype=torch.float32,
        )

        return {
            "species_idx": torch.tensor(self.vocab.get(mon_json["base_species"]), dtype=torch.long),
            "item_idx": torch.tensor(self.vocab.get(mon_json["item"]), dtype=torch.long),
            "ability_idx": torch.tensor(self.vocab.get(mon_json["ability"]), dtype=torch.long),
            "status_idx": torch.tensor(self.vocab.get(mon_json["status"]), dtype=torch.long),
            "tera_idx": torch.tensor(self.vocab.get(mon_json["tera_type"]), dtype=torch.long),
            "type_idxs": torch.tensor([self.vocab.get(t) for t in types], dtype=torch.long),
            "move_idxs": torch.tensor(move_idxs, dtype=torch.long),
            "numeric_feats": numeric_feats,
        }

    def species_known_only_dict(self, species_name):
        # only the species is known
        d = self.unknown_mon_dict()
        d["species_idx"] = torch.tensor(self.vocab.get(species_name), dtype=torch.long)
        return d

    def pad_bench(self, mon_list, num_slots):
        # action slots 4-8 are alphabetical by species
        ordered = sorted(mon_list, key=lambda m: m["base_species"])[:num_slots]
        mons = [self.mon_to_index_dict(m) for m in ordered]
        while len(mons) < num_slots:
            mons.append(self.unknown_mon_dict())
        return {k: torch.stack([m[k] for m in mons]) for k in mons[0]}

    def build_field_facts(self, state):
        """
        Build a the battle field tensor
        """
        player_hazards = torch.tensor(
            [1.0 if h in state["player_conditions"] else 0.0 for h in self.HAZARD_TOKENS], dtype=torch.float32
        )
        opp_hazards = torch.tensor(
            [1.0 if h in state["opponent_conditions"] else 0.0 for h in self.HAZARD_TOKENS], dtype=torch.float32
        )
        scalar_feats = torch.tensor([
            state["opponents_remaining"] / 6.0,
            float(state["forced_switch"]),
            float(state["can_tera"]),
        ], dtype=torch.float32)

        return {
            "format_idx": torch.tensor(self.vocab.format_token(state["format"]), dtype=torch.long),
            "weather_idx": torch.tensor(self.vocab.get(state["weather"]), dtype=torch.long),
            "terrain_idx": torch.tensor(self.vocab.get(state["battle_field"]), dtype=torch.long),
            "player_prev_move_idx": torch.tensor(self.vocab.get(state["player_prev_move"]["name"]), dtype=torch.long),
            "opp_prev_move_idx": torch.tensor(self.vocab.get(state["opponent_prev_move"]["name"]), dtype=torch.long),
            "hazards": torch.cat([player_hazards, opp_hazards]),
            "scalars": scalar_feats,
        }

    @staticmethod
    def _pad_or_trim(seq, n, fill):
        seq = list(seq)[:n]
        if len(seq) < n:
            seq = seq + [fill] * (n - len(seq))
        return seq

    def _clamp_action(self, action, action_mask):
        """Keep 0-12 if legal; map missing / oob / masked-off labels to -1."""
        if action is None or action < 0:
            return -1
        if action >= self.ACTION_DIM or not bool(action_mask[action]):
            return -1
        return int(action)

    def build_action_mask(self, state):
        """
        Length 13: 0-3 moves, 4-8 switches, 9-12 tera-moves.
        Moves and switches are alphabetical so they match dataset labels.
        """
        forced = bool(state["forced_switch"])
        can_tera = bool(state["can_tera"])
        moves = sorted(state["player_active_pokemon"]["moves"], key=lambda m: m["name"])
        has_pp = [m["current_pp"] > 0 for m in moves]

        move_legal = self._pad_or_trim(
            [not forced and pp for pp in has_pp],
            self.MOVE_SLOTS,
            False,
        )
        switches = sorted(state["available_switches"], key=lambda m: m["base_species"])
        switch_legal = self._pad_or_trim(
            [True] * len(switches),
            self.BENCH_SLOTS,
            False,
        )
        tera_legal = self._pad_or_trim(
            [can_tera and not forced and pp for pp in has_pp],
            self.MOVE_SLOTS,
            False,
        )

        return torch.tensor(move_legal + switch_legal + tera_legal, dtype=torch.bool)

    def state_to_tensors(self, state):
        """
        Package everything into one function 
        """
        mine_active = self.mon_to_index_dict(state["player_active_pokemon"])
        mine_bench = self.pad_bench(state["available_switches"], self.BENCH_SLOTS)

        opp_active = self.mon_to_index_dict(state["opponent_active_pokemon"])
        opp_active_species = state["opponent_active_pokemon"]["base_species"]
        opp_bench_species = [s for s in state["opponent_teampreview"] if s != opp_active_species]

        opp_revealed_list = [self.species_known_only_dict(s) for s in opp_bench_species[:self.BENCH_SLOTS]]
        while len(opp_revealed_list) < self.BENCH_SLOTS:
            opp_revealed_list.append(self.unknown_mon_dict())
        opp_revealed = {k: torch.stack([m[k] for m in opp_revealed_list]) for k in opp_revealed_list[0]}

        return {
            "field_feats": self.build_field_facts(state),
            "mine_active": mine_active,
            "mine_bench": mine_bench,
            "opp_active": opp_active,
            "opp_revealed": opp_revealed,
            "action_mask": self.build_action_mask(state),
        }

    @staticmethod
    def collate_battle_examples(examples):
        """
        Examples are nested dicts so the
        default collate_fn can't stack them 
        do it field-by-field with py torch.
        """

        def stack_group(key):

            # print(examples[0])
            return {
                field : torch.stack([ex[key][field] for ex in examples]) for field in examples[0][key]
            }

        return {
            "field_feats": stack_group("field_feats"),
            "mine_active": stack_group("mine_active"),
            "mine_bench": stack_group("mine_bench"),
            "opp_active": stack_group("opp_active"),
            "opp_revealed": stack_group("opp_revealed"),
            "action_mask": torch.stack([ex["action_mask"] for ex in examples]),
            "action_label": torch.stack([ex["action_label"] for ex in examples]),
        }
    
    def _flatten(self, samples):
        """
        every shard has a lot of (state,action) pairs and examples
        compose() needs to genrator, regular map() doesnt do one to many
        """
        for s in samples:
            yield from self.battle_to_example(s)

    def _build_dataset(self, shards, shuffle):
        shard_urls = [self._to_shard_url(s) for s in shards]
        ds = wds.WebDataset(shard_urls, shardshuffle=shuffle)
        ds = ds.compose(self._flatten)

        if shuffle:
            ds=ds.shuffle(20000)
        
        return ds
    
    def get_train_loader(self, batch_size=256, workers=6):
        dataset = self._build_dataset(self.train_shards, shuffle=True)
        return wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=self.collate_battle_examples,
        )
    
    def get_val_loader(self, batch_size=256, workers=6):
        dataset = self._build_dataset(self.val_shards, shuffle=False)
        return wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=self.collate_battle_examples,
        )
    
    def _to_shard_url(self, path_str):
        shard_path = Path(path_str)
        if not shard_path.is_absolute():
            shard_path = self.shard_dir / shard_path
        return "file:" + shard_path.resolve().as_posix()





"""
testing purposes only!
"""
if __name__=="__main__":
    dataloader = DataLoader()
    train_loader = dataloader.get_train_loader(batch_size=256, workers=6)
    val_loader = dataloader.get_val_loader(batch_size=256, workers=2)

    for batch in train_loader:
        print(batch["action_label"].shape)   # (256,)
        print(batch["mine_bench"]["species_idx"].shape)  # (256, 5)
        break
    
