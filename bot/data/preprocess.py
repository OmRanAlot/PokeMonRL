import json
import random
import glob
import torch
import webdataset as wds
from pathlib import Path
import os, sys
import lz4.frame
import os
from tqdm import tqdm 
from multiprocessing import Pool, cpu_count
"""
Preprocess all of the data FIRST!
Don't use tensors and convert at the end
Do any sorts, 
"""

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
        # print(root_dir)
        with open(path, "r") as f:
            self.token_to_idx = json.load(f)
        self.BLANK = self.token_to_idx["<blank>"]  # fallback for anything not in vocab

    def get(self, token):
        if token is None:
            return self.BLANK
        return self.token_to_idx.get(token, self.BLANK)

    def format_token(self, fmt):
        return self.get(f"<{fmt}>")

class FastDataPreprocessor:
    def __init__(self, vocab):
        self.vocab = vocab
        self.MOVE_SLOTS = 4
        self.BENCH_SLOTS = 5
        self.ACTION_DIM = 13
        self.HAZARD_TOKENS = ["spikes", "stealthrock", "toxicspikes", "stickyweb"]
        self.NUMERIC_DIM = 1 + 7 + 6 + self.MOVE_SLOTS

    @staticmethod
    def _pad_or_trim(seq, n, fill):
        seq = list(seq)[:n]
        if len(seq) < n:
            seq = seq + [fill] * (n - len(seq))
        return seq

    
    def unknown_mon_dict(self):
        blank = self.vocab.BLANK
        return {
            "species_idx": blank,
            "item_idx": blank,
            "ability_idx": blank,
            "status_idx": blank,
            "tera_idx": blank,
            "type_idxs": [blank, blank],
            "move_idxs": [blank] * self.MOVE_SLOTS,
            "numeric_feats": [0.0] * self.NUMERIC_DIM,
        }
    

    def mon_to_index_dict(self, mon_json):
        if mon_json is None:
            return self.unknown_mon_dict()

        types = mon_json["types"].split()
        types = (types + [None, None])[:2]
        moves = sorted(mon_json["moves"], key=lambda m: m["name"])[:self.MOVE_SLOTS]
        
        move_idxs = self._pad_or_trim([self.vocab.get(m["name"]) for m in moves], self.MOVE_SLOTS, self.vocab.BLANK)
        move_pp_frac = self._pad_or_trim(
            [m["current_pp"] / m["max_pp"] if m["max_pp"] > 0 else 0.0 for m in moves], self.MOVE_SLOTS, 0.0
        )
        boosts = [mon_json[k] for k in ("atk_boost", "spa_boost", "def_boost", "spd_boost", "spe_boost", "accuracy_boost", "evasion_boost")]
        base_stats = [mon_json[k] for k in ("base_atk", "base_spa", "base_def", "base_spd", "base_spe", "base_hp")]

        return {
            "species_idx": self.vocab.get(mon_json["base_species"]),
            "item_idx": self.vocab.get(mon_json["item"]),
            "ability_idx": self.vocab.get(mon_json["ability"]),
            "status_idx": self.vocab.get(mon_json["status"]),
            "tera_idx": self.vocab.get(mon_json["tera_type"]),
            "type_idxs": [self.vocab.get(t) for t in types],
            "move_idxs": move_idxs,
            "numeric_feats": [mon_json["hp_pct"]] + boosts + base_stats + move_pp_frac,
        }

    def species_known_only_dict(self, species_name):
        d = self.unknown_mon_dict()
        d["species_idx"] = self.vocab.get(species_name)
        return d

    def build_field_facts(self, state):
        player_hazards = [1.0 if h in state["player_conditions"] else 0.0 for h in self.HAZARD_TOKENS]
        opp_hazards = [1.0 if h in state["opponent_conditions"] else 0.0 for h in self.HAZARD_TOKENS]
        return {
            "format_idx": self.vocab.format_token(state["format"]),
            "weather_idx": self.vocab.get(state["weather"]),
            "terrain_idx": self.vocab.get(state["battle_field"]),
            "player_prev_move_idx": self.vocab.get(state["player_prev_move"]["name"]),
            "opp_prev_move_idx": self.vocab.get(state["opponent_prev_move"]["name"]),
            "hazards": player_hazards + opp_hazards,
            "scalars": [state["opponents_remaining"] / 6.0, float(state["forced_switch"]), float(state["can_tera"])],
        }

    def build_action_mask(self, state):
        forced = bool(state["forced_switch"])
        can_tera = bool(state["can_tera"])
        moves = sorted(state["player_active_pokemon"]["moves"], key=lambda m: m["name"])
        has_pp = [m["current_pp"] > 0 for m in moves]
        move_legal = self._pad_or_trim([not forced and pp for pp in has_pp], self.MOVE_SLOTS, False)
        switches = sorted(state["available_switches"], key=lambda m: m["base_species"])
        switch_legal = self._pad_or_trim([True] * len(switches), self.BENCH_SLOTS, False)
        tera_legal = self._pad_or_trim([can_tera and not forced and pp for pp in has_pp], self.MOVE_SLOTS, False)
        return move_legal + switch_legal + tera_legal

    def process_state(self, state, action):
        mine_active = self.mon_to_index_dict(state["player_active_pokemon"])
        ordered_bench = sorted(state["available_switches"], key=lambda m: m["base_species"])[:self.BENCH_SLOTS]
        mine_bench = [self.mon_to_index_dict(m) for m in ordered_bench]
        while len(mine_bench) < self.BENCH_SLOTS:
            mine_bench.append(self.unknown_mon_dict())

        opp_active = self.mon_to_index_dict(state["opponent_active_pokemon"])
        opp_active_species = state["opponent_active_pokemon"]["base_species"]
        opp_bench_species = [s for s in state["opponent_teampreview"] if s != opp_active_species]
        opp_revealed = [self.species_known_only_dict(s) for s in opp_bench_species[:self.BENCH_SLOTS]]
        while len(opp_revealed) < self.BENCH_SLOTS:
            opp_revealed.append(self.unknown_mon_dict())
            
        action_mask = self.build_action_mask(state)
        action_label = action if (action is not None and 0 <= action < self.ACTION_DIM and action_mask[action]) else -1

        # NOT TENSORS
        return {
            "field_feats": self.build_field_facts(state),
            "mine_active": mine_active,
            "mine_bench": mine_bench, # List of dicts
            "opp_active": opp_active,
            "opp_revealed": opp_revealed, # List of dicts
            "action_mask": action_mask,
            "action_label": action_label,
        }


def collate_primitives_to_tensors(examples):
    """
    Converts a large list of Python primitive dictionaries into one batched PyTorch dictionary.
    Covert all at once to save on computation
    """
    def stack_group(key, is_list=False):
        if is_list:
            # For mine_bench and opp_revealed
            return {
                field: torch.tensor([[ex[key][i][field] for i in range(len(ex[key]))] for ex in examples], dtype=torch.float32 if "numeric" in field else torch.long)
                for field in examples[0][key][0]
            }
        return {
            field: torch.tensor([ex[key][field] for ex in examples], dtype=torch.float32 if "numeric" in field or field in ["hazards", "scalars"] else torch.long)
            for field in examples[0][key]
        }

    return {
        "field_feats": stack_group("field_feats"),
        "mine_active": stack_group("mine_active"),
        "mine_bench": stack_group("mine_bench", is_list=True),
        "opp_active": stack_group("opp_active"),
        "opp_revealed": stack_group("opp_revealed", is_list=True),
        "action_mask": torch.tensor([ex["action_mask"] for ex in examples], dtype=torch.bool),
        "action_label": torch.tensor([ex["action_label"] for ex in examples], dtype=torch.long),
    }

def _to_shard_url(path):
    """WebDataset's gopen treats 'C:' as a URL scheme and strips backslashes."""
    return "file:" + Path(path).resolve().as_posix()



# sequentual on one thread
def process_and_save_shards(shard_dir, output_dir, chunk_size=20000):
    os.makedirs(output_dir, exist_ok=True)
    vocab = Vocab()
    processor = FastDataPreprocessor(vocab)
    
    all_shards = sorted(glob.glob(os.path.join(shard_dir, "*.tar")))
    print(f"Found {len(all_shards)} shards. Starting processing...")
    
    buffer = []
    chunk_index = 0
    
    for shard_path in tqdm(all_shards):
        dataset = wds.WebDataset(_to_shard_url(shard_path), shardshuffle=False)
        for sample in dataset:
            compressed = sample["json.lz4"]
            battle = json.loads(lz4.frame.decompress(compressed))
            states, actions = battle["states"], battle["actions"]
            # print("yo")
            for s, a in zip(states, actions):
                buffer.append(processor.process_state(s, a))
                
                # When buffer hits chunk size, convert to tensors and save
                if len(buffer) >= chunk_size:
                    batch_tensors = collate_primitives_to_tensors(buffer)
                    torch.save(batch_tensors, os.path.join(output_dir, f"batch_{chunk_index:05d}.pt"))
                    chunk_index += 1
                    buffer = [] # Clear buffer
                    
    # Save any remaining examples
    if buffer:
        batch_tensors = collate_primitives_to_tensors(buffer)
        torch.save(batch_tensors, os.path.join(output_dir, f"batch_{chunk_index:05d}.pt"))
        print(f"Saved final chunk {chunk_index}.")

# only one shard
def process_single_shard(args):
    shard_path, output_dir, chunk_size = args
    # Each CPU core needs its own isolated instance of the Vocab and Processor
    vocab = Vocab() 
    processor = FastDataPreprocessor(vocab)

    chunk_index =0
    buffer = []
    dataset = wds.WebDataset(_to_shard_url(shard_path), shardshuffle=False)
    
    #use shards OG name so processes dont overwrite each other
    shard_name = Path(shard_path).stem

    for sample in dataset:
        compressed = sample["json.lz4"]
        battle = json.loads(lz4.frame.decompress(compressed))
        states, actions = battle["states"], battle["actions"]
        
        for s, a in zip(states, actions):
            buffer.append(processor.process_state(s, a))
               
            if len(buffer) >= chunk_size:
                batch_tensors = collate_primitives_to_tensors(buffer)
                torch.save(batch_tensors, os.path.join(output_dir, f"{shard_name}_batch_{chunk_index:03d}.pt"))
                chunk_index += 1
                buffer = [] 

    if buffer:
        batch_tensors = collate_primitives_to_tensors(buffer)
        torch.save(batch_tensors, os.path.join(output_dir, f"{shard_name}_batch_{chunk_index:05d}.pt"))
        print(f"Saved final chunk {chunk_index}.")
                

def process_and_save_shards_parrallel(shard_dir, output_dir, chunk_size=20000):
    all_shards = sorted(glob.glob(os.path.join(shard_dir, "*.tar")))
    print(f"Found {len(all_shards)} shards. Starting processing...")

    cores = cpu_count()

    print(f"Processing with {cores} cores")

    args_list = [(shard, output_dir, chunk_size) for shard in all_shards]

    with Pool(processes=cores) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_shard, args_list), total=len(all_shards)):
            pass

    print("Finished converting all shards!")


"""
RUN THE PIPELINE
"""
if __name__=="__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    shard_dir = repo_root / "data" / "data" / "shard"
    output_dir = repo_root / "data" / "data" / "processed_data"

    process_and_save_shards_parrallel(shard_dir, output_dir)
    
