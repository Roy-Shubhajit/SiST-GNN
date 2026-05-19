"""
Test script to verify multi-head attention output dimension handling
"""

import torch
from temporal_gnn import TemporalGNNModel, TemporalTransformerGNNModel

def test_gat_multihead():
    """Test TemporalGNNModel with GAT multi-head attention"""
    print("=" * 70)
    print("Testing TemporalGNNModel with GAT (4 heads)")
    print("=" * 70)
    
    # Parameters
    N = 10
    T = 5
    input_dim = 16
    hidden_dim = 32
    output_dim = 4
    E = 20
    device = torch.device("cpu")
    
    # Synthetic data
    X_seq = [torch.randn(N, input_dim) for _ in range(T)]
    edge_seq = [torch.randint(0, N, (2, E)) for _ in range(T)]
    
    # Model with GAT and multi-head attention
    model = TemporalGNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        gnn_type="GATConv",
        heads=4,  # 4 attention heads
        dropout=0.1,
    )
    
    print(model)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Forward pass
    out, outputs = model(X_seq, edge_seq)
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected shape: torch.Size([{N}, {output_dim}])")
    assert out.shape == torch.Size([N, output_dim]), f"Unexpected output shape: {out.shape}"
    print()
    
    # Single-step pass
    H, C = model.init_hidden(N, device)
    step_out, H, C = model.forward_step(X_seq[0], edge_seq[0], H, C)
    print(f"✓ Single-step forward pass successful!")
    print(f"  Single-step output shape: {step_out.shape}")
    assert step_out.shape == torch.Size([N, output_dim]), f"Unexpected single-step output shape: {step_out.shape}"
    print()


def test_transformer_gatv2_multihead():
    """Test TemporalTransformerGNNModel with GATv2 multi-head attention"""
    print("=" * 70)
    print("Testing TemporalTransformerGNNModel with GATv2 (4 heads)")
    print("=" * 70)
    
    # Parameters
    N = 10
    T = 5
    input_dim = 16
    hidden_dim = 32
    output_dim = 4
    E = 20
    device = torch.device("cpu")
    
    # Synthetic data
    X_seq = [torch.randn(N, input_dim) for _ in range(T)]
    edge_seq = [torch.randint(0, N, (2, E)) for _ in range(T)]
    
    # Model with GATv2 and multi-head attention
    model = TemporalTransformerGNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        gnn_type="GATv2Conv",
        transformer_heads=4,
        transformer_layers=1,
        heads=4,  # GATv2 heads parameter (for GNN, not Transformer)
        dropout=0.1,
        max_history=8,
    )
    
    print(model)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Forward pass
    out, outputs = model(X_seq, edge_seq)
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected shape: torch.Size([{N}, {output_dim}])")
    assert out.shape == torch.Size([N, output_dim]), f"Unexpected output shape: {out.shape}"
    print()
    
    # Single-step pass
    history_list = model.init_history(N, device)
    step_out, history_list = model.forward_step(X_seq[0], edge_seq[0], history_list)
    print(f"✓ Single-step forward pass successful!")
    print(f"  Single-step output shape: {step_out.shape}")
    assert step_out.shape == torch.Size([N, output_dim]), f"Unexpected single-step output shape: {step_out.shape}"
    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    test_gat_multihead()
    test_transformer_gatv2_multihead()
    
    print("=" * 70)
    print("✓ All tests passed! Multi-head attention is working correctly.")
    print("=" * 70)
