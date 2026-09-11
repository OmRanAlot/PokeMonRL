"""
Used for testing. Actual training did not occur via running this file on my computer
but on a Google Colab and connected to a Google Drive with all of the data

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from multiprocessing import freeze_support
import numpy as np
import json, lz4.frame
import sys
import os
from pathlib import Path  
import itertools 

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(root_dir)

from data.loadData import Vocab
from data.encoder import SharedEmbedding, FieldEncoder, PokemonEncoder, BattleStateEncoder, BCPolicy
from data.optimizeLoadData import FastDataLoader, FastDataSet, pack_compact_batch


def evaluate(model, evaluate, loss_fn, device, mini_batch_size):
    model.eval()
    total_loss = 0.0
    correct =0
    total = 0

    # batch_size=None: each .pt chunk is already a stacked batch; don't add a leading dim
    val_loader = DataLoader(
        evaluate, 
        batch_size=None,
        shuffle=True,        # Shuffles which chunk is loaded
        num_workers=3,       
        pin_memory=True      
    )

    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            # Keep the compact chunk on CPU; only widen + move each mini-batch
            size = batch["action_label"].shape[0]
            for start_idx in range(0, size, mini_batch_size):
                end_idx = start_idx + mini_batch_size
                mini_batch = pack_compact_batch(slice_batch(batch, start_idx, end_idx))
                mini_batch = move_to_device(mini_batch, device)

                y = mini_batch.pop("action_label")
                logits = model(**mini_batch)

                loss = loss_fn(logits, y)
                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim= -1)
                correct += (preds == y).sum().item()
                total += y.size(0)
    
    avg_loss = total_loss / n_batches
    accuracy = correct / total
    return avg_loss, accuracy, n_batches

def slice_batch(batch, start_idx, end_idx):
    """Recursively slices nested dictionaries of tensors."""
    if isinstance(batch, torch.Tensor):
        return batch[start_idx:end_idx]
    elif isinstance(batch, dict):
        return {k: slice_batch(v, start_idx, end_idx) for k, v in batch.items()}
    return batch

def main():
    """
    === SET UP ===
    """
    print("setting up!")
    device = "cpu"

    shared_embed = SharedEmbedding(vocab_size=2541, embed_dim=32)  # built ONCE
    pokemon_encoder = PokemonEncoder(shared_embed, numeric_dim=18)
    field_encoder = FieldEncoder(shared_embed, numeric_dim=11)      # same instance passed in
    state_encoder = BattleStateEncoder(field_encoder, pokemon_encoder, num_bench=5, num_opp_revealed=5)

    model = BCPolicy(state_encoder, action_dim=13, hidden_dim=256)

    #learning rate
    alpha = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=alpha, weight_decay=1e-2, eps=1e-9)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)


    """
    === LOAD DATA ===
    """
    print("loading data!")
    dl = FastDataLoader(data_dir="C:\\Users\\omran\\code\\PokemonRL\\data\\data\\processed_data\\")
    train = FastDataSet(dl.train_files)
    evaluate = FastDataSet(dl.eval_files)

    train_loader = DataLoader(
        train, 
        batch_size=None, #since PyTorch batch_size=1 adds a 1 dimension to my tensor :P        
        shuffle=True,        # Shuffles which chunk is loaded
        num_workers=6,       
        pin_memory=True      
    )

    print("done loading data")
    num_epochs = 3
    best_val_acc = 0.0

    model = model.to(device)

    mini_batch_size = 1024 # mini batch size to ensure weights are update frequently enough 

    print("started training!")

    # print(train_loader[0])
    for i in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_train_batches = 0

        for j, batch in enumerate(train_loader):
            # Chunk stays compact (uint16/int8/float16) on CPU.
            # Cast to long/float32 only after slicing so RAM/VRAM don't 4-8x the full 20k rows.
            size = batch["action_label"].shape[0] #get the size of the batch

            for start_idx in range(0, size, mini_batch_size): # run a for loop on the mini batches            
                end_idx = start_idx + mini_batch_size
                mini_batch = pack_compact_batch(slice_batch(batch, start_idx, end_idx))
                mini_batch = move_to_device(mini_batch, device)

                y = mini_batch.pop("action_label")
                action_mask = mini_batch.pop("action_mask")
                logits = model(**mini_batch, action_mask=action_mask) # forward phase into network

                loss = loss_fn(logits, y) # detect how far the forward phase is off by

                optimizer.zero_grad() # gradients accumalate over the batches so clear them
                loss.backward() # backpropagation
                optimizer.step() # apply update to ADAM optimizer
                print(f"Finshed Backprop {j} of epoch {i}")
                running_loss += loss.item()
                n_train_batches += 1
            
        
        """
        === EVALUATE ===
        """
        avg_train_loss = running_loss / n_train_batches
        val_loss, val_acc, n_val_batches = evaluate(model, evaluate, loss_fn, device, mini_batch_size)

        print(
            f"epoch {i}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} train_batches={n_train_batches} val_batches={n_val_batches}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.to_dict(), "best_bc_policy.pt")

def move_to_device(batch, device):

    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True) # occurs asynchrousnly

    #recursively go back for each dictionary value
    elif isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}

    return batch # something else

if __name__ == "__main__":
       freeze_support()
       main()