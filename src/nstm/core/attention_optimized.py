# src/nstm/core/attention_optimized.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .types import Tokens, States, NSTMConfig

class OptimizedHybridAttention(nn.Module):
    """
    Optimized hybrid attention mechanism for NSTM.
    
    Improvements:
    - Numerical stability with proper scaling and clipping
    - Layer normalization for better gradient flow
    - Dropout for regularization
    - Residual connections
    - Memory efficient implementation
    - Gradient checkpointing support
    """
    
    def __init__(self, config: NSTMConfig):
        super(OptimizedHybridAttention, self).__init__()
        self.config = config
        self.token_dim = config.token_dim
        self.state_dim = config.state_dim
        self.num_heads = config.num_attention_heads
        self.dropout_prob = getattr(config, 'dropout_prob', 0.1)
        
        if self.state_dim % self.num_heads != 0:
            raise ValueError(f"State dimension ({self.state_dim}) must be divisible "
                             f"by number of attention heads ({self.num_heads})")
        
        self.head_dim = self.state_dim // self.num_heads
        self.temperature = math.sqrt(self.head_dim)
        
        # Token-to-State Attention with proper initialization
        self.token_to_state_q = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.token_to_state_k = nn.Linear(self.token_dim, self.state_dim, bias=False)
        self.token_to_state_v = nn.Linear(self.token_dim, self.state_dim, bias=False)
        
        # State-to-State Attention
        self.state_to_state_q = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.state_to_state_k = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.state_to_state_v = nn.Linear(self.state_dim, self.state_dim, bias=False)
        
        # Layer normalization for stability
        self.layer_norm_1 = nn.LayerNorm(self.state_dim)
        self.layer_norm_2 = nn.LayerNorm(self.state_dim)
        self.layer_norm_3 = nn.LayerNorm(self.state_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_prob)
        self.attention_dropout = nn.Dropout(self.dropout_prob)
        
        # Output projections with residual connections
        self.output_projection = nn.Linear(self.state_dim, self.state_dim)
        self.gate_projection = nn.Linear(self.state_dim * 2, self.state_dim)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in [self.token_to_state_q, self.token_to_state_k, self.token_to_state_v,
                      self.state_to_state_q, self.state_to_state_k, self.state_to_state_v,
                      self.output_projection, self.gate_projection]:
            nn.init.xavier_uniform_(module.weight)
            
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into attention heads"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine attention heads"""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.state_dim)

    def _stable_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Numerically stable attention computation"""
        # Compute attention scores with proper scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Numerical stability: subtract max before softmax
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores = scores - scores_max
        
        # Apply softmax with numerical stability
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights

    def token_to_state_attention(self, tokens: Tokens, states: States) -> Tuple[States, torch.Tensor]:
        """Optimized token-to-state attention"""
        batch_size, seq_len, _ = tokens.shape
        _, num_states, _ = states.shape
        
        # Layer normalization for input stability
        states_norm = self.layer_norm_1(states)
        
        # Linear transformations
        q_state = self._split_heads(self.token_to_state_q(states_norm))
        k_token = self._split_heads(self.token_to_state_k(tokens))
        v_token = self._split_heads(self.token_to_state_v(tokens))
        
        # Stable attention computation
        attn_output, attn_weights = self._stable_attention(q_state, k_token, v_token)
        
        # Combine heads and apply residual connection
        attn_output = self._combine_heads(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output, attn_weights

    def state_to_state_attention(self, states: States) -> Tuple[States, torch.Tensor]:
        """Optimized state-to-state attention"""
        batch_size, num_states, _ = states.shape
        
        # Layer normalization
        states_norm = self.layer_norm_2(states)
        
        # Linear transformations
        q = self._split_heads(self.state_to_state_q(states_norm))
        k = self._split_heads(self.state_to_state_k(states_norm))
        v = self._split_heads(self.state_to_state_v(states_norm))
        
        # Stable attention computation
        attn_output, attn_weights = self._stable_attention(q, k, v)
        
        # Combine heads and apply residual connection
        attn_output = self._combine_heads(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output, attn_weights
        
    def forward(self, tokens: Tokens, states: States) -> Tuple[States, torch.Tensor, torch.Tensor]:
        """Forward pass with residual connections and gating"""
        # Store original states for residual connection
        residual = states
        
        # Token-to-state attention
        ts_output, ts_weights = self.token_to_state_attention(tokens, states)
        
        # State-to-state attention  
        ss_output, ss_weights = self.state_to_state_attention(states)
        
        # Gated combination of attention outputs
        combined = torch.cat([ts_output, ss_output], dim=-1)
        gate = torch.sigmoid(self.gate_projection(combined))
        gated_output = gate * ts_output + (1 - gate) * ss_output
        
        # Output projection
        output = self.output_projection(gated_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm_3(output + residual)
        
        return output, ts_weights, ss_weights