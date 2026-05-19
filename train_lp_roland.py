import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import List
import numpy as np
from evaluate_roland import evaluate_mrr, get_negative_samples, evaluate_fixed_split

class StatefulGNNWrapper(nn.Module):
    def __init__(self, base_model, num_nodes: int, device: torch.device, use_node_emb: bool = True, node_emb_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.num_nodes = num_nodes
        self.device = device
        
        self.use_node_emb = use_node_emb
        if self.use_node_emb:
            self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
            
        self.is_lstm = hasattr(self.base_model, 'init_hidden')
        self.reset_states()
        
    def reset_states(self):
        if self.is_lstm:
            self.H_list, self.C_list = self.base_model.init_hidden(self.num_nodes, self.device)
            self.history_list = None
        else:
            self.H_list, self.C_list = None, None
            self.history_list = self.base_model.init_history(self.num_nodes, self.device)
        
    def detach_states(self):
        if self.H_list is not None:
            self.H_list = [h.detach() for h in self.H_list]
        if self.C_list is not None:
            self.C_list = [c.detach() for c in self.C_list]
        if hasattr(self, 'history_list') and self.history_list is not None:
            self.history_list = [h.detach() for h in self.history_list]
            
    def clear_states(self):
        self.H_list, self.C_list, self.history_list = None, None, None

    def clone_states(self):
        if self.is_lstm:
            h = [h_i.clone() for h_i in self.H_list] if self.H_list else None
            c = [c_i.clone() for c_i in self.C_list] if self.C_list else None
            return h, c
        else:
            hist = [h_i.clone() for h_i in self.history_list] if self.history_list else None
            return hist, None
        
    def load_states(self, state1, state2):
        if self.is_lstm:
            self.H_list = state1
            self.C_list = state2
        else:
            self.history_list = state1

    def forward(self, x, edge_index, edge_attr=None, update_state=True):
        if (self.is_lstm and self.H_list is None) or (not self.is_lstm and getattr(self, 'history_list', None) is None):
            self.reset_states()
            
        if self.use_node_emb:
            node_ids = torch.arange(self.num_nodes, device=self.device)
            x = self.node_emb(node_ids)
            
        if self.is_lstm:
            out, new_H, new_C = self.base_model.forward_step(x, edge_index, self.H_list, self.C_list, edge_attr=edge_attr)
            if update_state:
                self.H_list = new_H
                self.C_list = new_C
        else:
            out, new_history = self.base_model.forward_step(x, edge_index, self.history_list, edge_attr=edge_attr)
            if update_state:
                self.history_list = new_history
            
        return out

def train_live_update(snapshots: List[Data], model: StatefulGNNWrapper, optimizer: torch.optim.Optimizer, epochs_per_snapshot: int = 1):
    print("Starting Live-Update Training & Evaluation...")
    device = model.device
    cumulative_mrr = []
    
    # In live update, we evaluate on t using states from t-1,
    # then we train on t updating states from t-1 -> t.
    # To train for multiple epochs on t, we need to save the t-1 states.
    for t in range(len(snapshots)):
        snap = snapshots[t].to(device)
        if snap.edge_index.size(1) == 0:
            continue
            
        # Evaluate
        if t > 0:
            mrr_t = evaluate_mrr(model, snap) # inside this, model(..., update_state=False) is called... Wait, evaluate_mrr calls model(..., update_state=False)? We need to change evaluate_mrr!
            cumulative_mrr.append(mrr_t)
            print(f"Snapshot Time {t:03d} | Test MRR: {mrr_t:.4f}")
            
        # Train
        # Save pre-training states (t-1)
        prev_h, prev_c = model.clone_states()
        
        model.train()
        for epoch in range(epochs_per_snapshot):
            # Load t-1 states for fresh forward pass
            model.load_states(prev_h, prev_c)
            # clone again so we don't accidentally modify inplace if graph is recurrent
            model.load_states(*(model.clone_states())) 
            
            optimizer.zero_grad()
            node_emb = model(snap.x, snap.edge_index, snap.edge_attr, update_state=True)
            
            pos_scores = (node_emb[snap.edge_index[0]] * node_emb[snap.edge_index[1]]).sum(dim=-1)
            neg_edge_index = get_negative_samples(snap.edge_index, snap.num_nodes, num_neg=1)
            neg_scores = (node_emb[neg_edge_index[0]] * node_emb[neg_edge_index[1]]).sum(dim=-1)
            
            loss = torch.nn.functional.margin_ranking_loss(
                pos_scores, neg_scores, torch.ones_like(pos_scores), margin=1.0
            )
            
            loss.backward()
            model.detach_states()
            optimizer.step()
            
        # The final states for 't' are now inside model (from the last epoch).
        # We will use these for 't+1' evaluation.
        
        # Free memory of the processed snapshot
        snapshots[t] = None
        del snap
        if str(model.device).startswith('cuda'):
            torch.cuda.empty_cache()

    avg_mrr = np.mean(cumulative_mrr) if cumulative_mrr else 0.0
    print(f"Live-Update Complete. Average MRR over time: {avg_mrr:.4f}")
    return avg_mrr

def train_fixed_split(snapshots: List[Data], model: StatefulGNNWrapper, optimizer: torch.optim.Optimizer, num_epochs: int = 50):
    print("Starting Fixed-Split Training & Evaluation...")
    num_snaps = len(snapshots)
    train_end = int(num_snaps * 0.9)
    train_snaps = snapshots[:train_end]
    device = model.device
    
    for epoch in range(num_epochs):
        model.train()
        model.reset_states()
        total_loss = 0
        valid_snaps = 0
        
        for snap in train_snaps:
            snap = snap.to(device)
            if snap.edge_index.size(1) == 0:
                continue
                
            optimizer.zero_grad()
            node_emb = model(snap.x, snap.edge_index, snap.edge_attr, update_state=True)
            
            pos_scores = (node_emb[snap.edge_index[0]] * node_emb[snap.edge_index[1]]).sum(dim=-1)
            neg_edge_index = get_negative_samples(snap.edge_index, snap.num_nodes, num_neg=1)
            neg_scores = (node_emb[neg_edge_index[0]] * node_emb[neg_edge_index[1]]).sum(dim=-1)
            
            loss = torch.nn.functional.margin_ranking_loss(
                pos_scores, neg_scores, torch.ones_like(pos_scores), margin=1.0
            )
            
            loss.backward()
            model.detach_states()
            optimizer.step()
            total_loss += loss.item()
            valid_snaps += 1
            
        if (epoch + 1) % 10 == 0 and valid_snaps > 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss / valid_snaps:.4f}")
            
    avg_mrr = evaluate_fixed_split(model, snapshots, train_end)
    print(f"Fixed-Split Evaluation | Final 10% MRR: {avg_mrr:.4f}")
    return avg_mrr
