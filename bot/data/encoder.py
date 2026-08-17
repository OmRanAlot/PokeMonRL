import pandas as pd
import matplotlib as plt
import numpy as np
import json, lz4.frame
import sys
import torch
import torch.nn as nn
import torch.optim as optim

class SharedEmbedding(nn.Module):
    #this is just for my understanding and this class is not technically needed!
    def __init__(self, vocab_size=2540, embed_dim=64):
        super().__init__()

        #embeddings are basically ways to represent non-numerical data as numbers
        #they are vectors of whatever embed_dim is

        #they start as random values
        #but through backprop, then we can adjust them and figure out should be closer to each other 
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
    #returns the idx
    def forward(self, idx):
        return self.embeddings(idx)
        
class PokemonEncoder(nn.Module):
    def __init__(self, embed, numeric_dim):
        super().__init__()
        self.embed = embed

        self.output_dimensions = 11 * embed.embed_dim + numeric_dim

    def forward(self, species_idx, item_idx, ability_idx, status_idx,
                tera_idx, type_idxs, move_idxs, numeric_feats):
        
        #access the vectors for each index from the embeddings
        species_vector = self.embed(species_idx)
        item_vector = self.embed(item_idx)
        ability_vector = self.embed(ability_idx)
        status_vector = self.embed(status_idx)
        tera_vector = self.embed(tera_idx)
        
        #flatten the 3D tensors into 2D since types are 2 long and moves are 4
        type_vector = self.embed(type_idxs)    
        type_vector = type_vector.reshape(type_vector.shape[0],-1)

        move_vector = self.embed(move_idxs)
        move_vector = move_vector.reshape(move_vector.shape[0],-1)

        out = torch.cat([species_vector,item_vector,ability_vector,status_vector,tera_vector, type_vector, move_vector, numeric_feats],
                        dim = -1)

        return out

class FieldEncoder(nn.Module):
    def __init__(self, embed, numeric_dim=11):
        super().__init__()
        self.embed = embed

        # format, weather, terrain, player_prev_move, opp_prev_move
        # numeric: hazards(8) + scalars(3)
        self.output_dimensions = 5 * embed.embed_dim + numeric_dim

    def forward(self, format_idx, weather_idx, terrain_idx,
                player_prev_move_idx, opp_prev_move_idx, hazards, scalars):
        
        format_vector = self.embed(format_idx)
        weather_vector = self.embed(weather_idx)
        terrain_vector = self.embed(terrain_idx)
        player_prev_move_vector = self.embed(player_prev_move_idx)
        opp_prev_move_vector = self.embed(opp_prev_move_idx)

        out = torch.cat([
            format_vector, weather_vector, terrain_vector,
            player_prev_move_vector, opp_prev_move_vector,
            hazards, scalars,
        ], dim=-1)

        return out

class BattleStateEncoder(nn.Module):
    def __init__(self, field_encoder, pokemon_encoder, num_bench=5, num_opp_revealed=5):
        super().__init__()
        self.field_encoder = field_encoder
        self.pokemon_encoder = pokemon_encoder
        self.num_bench = num_bench
        self.num_opp_revealed = num_opp_revealed

        pkmn_dim = pokemon_encoder.output_dimensions
        self.output_dimensions = (
            field_encoder.output_dimensions
            + pkmn_dim                        # my active
            + pkmn_dim * num_bench            # my bench
            + pkmn_dim                        # opp active
            + pkmn_dim * num_opp_revealed     # opp revealed
        )

    def _encode_group(self, feats, num_slots):
        batch_size = feats["species_idx"].shape[0]
        flat_feats = {k: v.reshape(batch_size * num_slots, *v.shape[2:]) for k, v in feats.items()}
        out = self.pokemon_encoder(**flat_feats)                 # (batch*num_slots, pkmn_dim)
        return out.reshape(batch_size, num_slots * out.shape[-1])  # (batch, num_slots*pkmn_dim)

    def forward(self, field_feats, mine_active, mine_bench, opp_active, opp_revealed):
        field_enc = self.field_encoder(**field_feats)

        active_enc = self.pokemon_encoder(**mine_active)
        opp_active_enc = self.pokemon_encoder(**opp_active)

        bench_enc = self._encode_group(mine_bench, self.num_bench)
        opp_revealed_enc = self._encode_group(opp_revealed, self.num_opp_revealed)

        return torch.cat(
            [field_enc, active_enc, bench_enc, opp_active_enc, opp_revealed_enc],
            dim=-1,
        )

class BCPolicy(nn.Module):
    def __init__(self, state_encoder, action_dim=13, hidden_dim=256):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=0.1) # dropout with probability of 0.1
        self.head = nn.Sequential(
            nn.Linear(state_encoder.output_dimensions, hidden_dim), # y = w^T * x + b
            nn.ReLU(), # max(x,0)
            self.dropout, # dropout layer
            nn.Linear(hidden_dim, action_dim),
        )

    def to_dict(self):
        embed = self.state_encoder.pokemon_encoder.embed
        poke = self.state_encoder.pokemon_encoder
        field = self.state_encoder.field_encoder
        return {
            "state_dict": self.state_dict(),
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "vocab_size": embed.vocab_size,
            "embed_dim": embed.embed_dim,
            "pkmn_numeric_dim": poke.output_dimensions - 11 * embed.embed_dim,
            "field_numeric_dim": field.output_dimensions - 5 * embed.embed_dim,
            "num_bench": self.state_encoder.num_bench,
            "num_opp_revealed": self.state_encoder.num_opp_revealed,
        }

    def forward(self, field_feats, mine_active, mine_bench, opp_active, opp_revealed, action_mask):
        # forward pass into the Neural Network
        state = self.state_encoder(field_feats, mine_active, mine_bench, opp_active, opp_revealed)
        logits = self.head(state)
        logits = logits.masked_fill(~action_mask, float("-inf")) #for any action that is illegal(switch to fainted mon, etc.)

        return logits
    
    