"""
Test edge attribute handling with GAT layers (without edge_dim initialization)
"""

import torch
from temporal_gnn import TemporalGNNModel, TemporalTransformerGNNModel

def test_gat_with_edge_attr():
    """Test TemporalGNNModel with GAT and edge attributes"""
    print("=" * 70)
    print("Testing TemporalGNNModel with GAT + edge attributes")
    print("=" * 70)
    
    # Parameters
    N = 10
    T = 5
    input_dim = 16
    hidden_dim = 32
    output_dim = 4
    E = 20
    device = torch.device("cpu")
    
    # Synthetic data with edge attributes
    X_seq = [torch.randn(N, input_dim) for _ in range(T)]
    edge_seq = [torch.randint(0, N, (2, E)) for _ in range(T)]
    edge_attr_seq = [torch.randn(E) for _ in range(T)]  # Edge attributes
    
    # Model with GAT and multi-head attention (no edge_dim specified)
    model = TemporalGNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        gnn_type="GATConv",
        heads=4,
        dropout=0.1,
    )
    
    print(f"Model created: {model.layers[0].gnn}")
    print()
    
    # Forward pass with edge attributes
    # This should gracefully handle the edge_attr even though GATConv
    # wasn't initialized with edge_dim
    try:
        H, C = model.init_hidden(N, device)
        step_out, H, C = model.forward_step(
            X_seq[0], edge_seq[0], H, C, edge_attr=edge_attr_seq[0]
        )
        print(f"✓ Forward pass with edge attributes successful!")
        print(f"  Output shape: {step_out.shape}")
        assert step_out.shape == torch.Size([N, output_dim])
        print(f"✓ Output shape is correct!")
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        raise
    
    print()


def test_gatv2_with_edge_attr():
    """Test TemporalTransformerGNNModel with GATv2 and edge attributes"""
    print("=" * 70)
    print("Testing TemporalTransformerGNNModel with GATv2 + edge attributes")
    print("=" * 70)
    
    # Parameters
    N = 10
    T = 5
    input_dim = 16
    hidden_dim = 32
    output_dim = 4
    E = 20
    device = torch.device("cpu")
    
    # Synthetic data with edge attributes
    X_seq = [torch.randn(N, input_dim) for _ in range(T)]
    edge_seq = [torch.randint(0, N, (2, E)) for _ in range(T)]
    edge_attr_seq = [torch.randn(E) for _ in range(T)]
    
    # Model with GATv2 and multi-head attention (no edge_dim specified)
    model = TemporalTransformerGNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        gnn_type="GATv2Conv",
        transformer_heads=4,
        transformer_layers=1,
        heads=4,
        dropout=0.1,
        max_history=8,
    )
    
    print(f"Model created: {model.layers[0].gnn}")
    print()
    
    # Forward pass with edge attributes
    try:
        history_list = model.init_history(N, device)
        step_out, history_list = model.forward_step(
            X_seq[0], edge_seq[0], history_list, edge_attr=edge_attr_seq[0]
        )
        print(f"✓ Forward pass with edge attributes successful!")
        print(f"  Output shape: {step_out.shape}")
        assert step_out.shape == torch.Size([N, output_dim])
        print(f"✓ Output shape is correct!")
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        raise
    
    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    test_gat_with_edge_attr()
    test_gatv2_with_edge_attr()
    
    print("=" * 70)
    print("✓ All edge attribute tests passed!")
    print("=" * 70)
