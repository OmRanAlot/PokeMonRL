"""
Used for testing. Actual training did not occur via running this file on my computer
but on a Google Colab and connected to a Google Drive with all of the data

"""

import torch
import torch.nn as nn
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

from data.loadData import DataLoader, Vocab
from data.encoder import SharedEmbedding, FieldEncoder, PokemonEncoder, BattleStateEncoder, BCPolicy

def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct =0
    total = 0

    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else
                         {kk: vv.to(device) for kk, vv in v.items()}
                     for k, v in batch.items()}

            y = batch.pop("action_label")
            logits = model(**batch)

            loss = loss_fn(logits, y)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / n_batches
    accuracy = correct / total
    return avg_loss, accuracy, n_batches

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
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)


    """
    === LOAD DATA ===
    """
    print("loadin data!")
    dataloader = DataLoader(split=0.9, subset_fraction=0.00625)
    train_loader = dataloader.get_train_loader(batch_size=256, workers=1)
    val_loader = dataloader.get_val_loader(batch_size=256, workers=1)
    print("done loading data")
    num_epochs = 3
    best_val_acc = 0.0

    print("started training!")

    # max_batches = int(0.01*(2_500_000/256)) # use only 1%

    for i in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_train_batches = 0

        for j, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else
                 {kk: vv.to(device) for kk, vv in v.items()}
             for k, v in batch.items()}
             
            y = batch.pop("action_label")
            logits = model(**batch) # forward phase into network

            # bad = (y == -1).nonzero(as_tuple=True)[0]
            # print(f"{len(bad)} / {y.numel()} targets are -1")
            # print(y[bad][:5])  # debugging

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
        val_loss, val_acc, n_val_batches = evaluate(model, val_loader, loss_fn, device)

        print(
            f"epoch {i}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} train_batches={n_train_batches} val_batches={n_val_batches}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.to_dict(), "best_bc_policy.pt")


if __name__ == "__main__":
       freeze_support()
       main()