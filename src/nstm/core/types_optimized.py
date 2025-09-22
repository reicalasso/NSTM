# src/nstm/core/types_optimized.py

from typing import NamedTuple
import torch

# Basic type definitions
State = torch.Tensor
Token = torch.Tensor
States = torch.Tensor
Tokens = torch.Tensor
StateImportanceScores = torch.Tensor
TokenToStateRoutingWeights = torch.Tensor

class OptimizedNSTMConfig(NamedTuple):
    """Enhanced configuration parameters for the optimized NSTM model."""
    # Basic dimensions
    state_dim: int
    token_dim: int
    
    # Architecture parameters
    gate_type: str = 'gru'  # 'gru' or 'lstm'
    num_attention_heads: int = 4
    routing_heads: int = 4
    
    # State management
    max_states: int = 64
    initial_states: int = 16
    prune_threshold: float = 0.3
    adaptive_threshold: bool = True
    importance_ema_decay: float = 0.9
    
    # Regularization
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    gradient_clip_norm: float = 1.0
    
    # Routing parameters
    use_gumbel_routing: bool = False
    routing_entropy_weight: float = 0.01
    
    # Training parameters
    learning_rate: float = 1e-3
    warmup_steps: int = 1000
    weight_decay: float = 1e-4
    
    # Optimization flags
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = False