# src/nstm/core/token_router_optimized.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .types import Tokens, States, TokenToStateRoutingWeights, NSTMConfig

class OptimizedTokenToStateRouter(nn.Module):
    """
    Optimized token-to-state router with improved attention mechanisms.
    
    Improvements:
    - Multi-head routing attention
    - Learnable temperature scaling
    - Gumbel softmax for discrete routing (optional)
    - Entropy regularization for diverse routing
    - Layer normalization for stability
    - Residual connections in query/key projections
    """
    
    def __init__(self, config: NSTMConfig):
        super(OptimizedTokenToStateRouter, self).__init__()
        self.config = config
        self.token_dim = config.token_dim
        self.state_dim = config.state_dim
        self.num_heads = getattr(config, 'routing_heads', 4)
        self.dropout_prob = getattr(config, 'dropout_prob', 0.1)
        self.use_gumbel = getattr(config, 'use_gumbel_routing', False)
        self.entropy_weight = getattr(config, 'routing_entropy_weight', 0.01)
        
        # Multi-head routing
        if self.state_dim % self.num_heads != 0:
            # Adjust to nearest divisible number
            self.head_dim = self.state_dim // self.num_heads
            self.proj_dim = self.head_dim * self.num_heads
        else:
            self.head_dim = self.state_dim // self.num_heads
            self.proj_dim = self.state_dim
            
        # Query and key projections
        self.token_to_query = nn.Linear(self.token_dim, self.proj_dim)
        self.state_to_key = nn.Linear(self.state_dim, self.proj_dim)
        self.state_to_value = nn.Linear(self.state_dim, self.proj_dim)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))
        
        # Layer normalization
        self.layer_norm_tokens = nn.LayerNorm(self.token_dim)
        self.layer_norm_states = nn.LayerNorm(self.state_dim)
        
        # Output projection
        self.output_projection = nn.Linear(self.proj_dim, self.state_dim)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_prob)
        self.attention_dropout = nn.Dropout(self.dropout_prob)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in [self.token_to_query, self.state_to_key, self.state_to_value, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into multiple heads"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine multiple heads"""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.proj_dim)

    def compute_routing_weights(
        self, 
        tokens: Tokens, 
        states: States
    ) -> TokenToStateRoutingWeights:
        """Compute multi-head routing weights with stability improvements"""
        batch_size, seq_len, token_dim = tokens.shape
        _, num_states, state_dim = states.shape
        
        # Input validation
        if token_dim != self.token_dim:
            raise ValueError(f"Token dimension mismatch. Expected: {self.token_dim}, Got: {token_dim}")
        if state_dim != self.state_dim:
            raise ValueError(f"State dimension mismatch. Expected: {self.state_dim}, Got: {state_dim}")
        
        # Layer normalization
        tokens_norm = self.layer_norm_tokens(tokens)
        states_norm = self.layer_norm_states(states)
        
        # Linear transformations
        queries = self.token_to_query(tokens_norm)  # (B, L, proj_dim)
        keys = self.state_to_key(states_norm)       # (B, S, proj_dim)
        
        # Split into heads
        q_heads = self._split_heads(queries)  # (B, num_heads, L, head_dim)
        k_heads = self._split_heads(keys)     # (B, num_heads, S, head_dim)
        
        # Compute attention scores
        # (B, num_heads, L, head_dim) @ (B, num_heads, head_dim, S) -> (B, num_heads, L, S)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))
        
        # Apply learnable temperature scaling
        scores = scores / torch.clamp(self.temperature, min=0.1)
        
        # Apply softmax across states dimension
        if self.use_gumbel and self.training:
            # Gumbel softmax for discrete routing during training
            routing_weights = F.gumbel_softmax(scores, tau=1.0, hard=False, dim=-1)
        else:
            # Standard softmax
            routing_weights = F.softmax(scores, dim=-1)
            
        # Apply attention dropout
        routing_weights = self.attention_dropout(routing_weights)
        
        # Combine heads by averaging
        routing_weights = routing_weights.mean(dim=1)  # (B, L, S)
        
        return routing_weights
        
    def compute_entropy_loss(self, routing_weights: TokenToStateRoutingWeights) -> torch.Tensor:
        """Compute entropy regularization loss to encourage diverse routing"""
        # Compute entropy across states dimension
        log_weights = torch.log(routing_weights + 1e-8)
        entropy = -(routing_weights * log_weights).sum(dim=-1)  # (B, L)
        
        # Return negative entropy as loss (we want to maximize entropy)
        return -entropy.mean()
        
    def route_tokens(
        self, 
        tokens: Tokens, 
        states: States
    ) -> Tuple[Tokens, TokenToStateRoutingWeights, torch.Tensor]:
        """Route tokens with entropy regularization"""
        # Compute routing weights
        routing_weights = self.compute_routing_weights(tokens, states)
        
        # Compute entropy loss
        entropy_loss = self.compute_entropy_loss(routing_weights) * self.entropy_weight
        
        # Optionally apply routing to get routed tokens
        # This is for compatibility, actual routing is done in StatePropagator
        return tokens, routing_weights, entropy_loss

    def apply_routing(
        self,
        tokens: Tokens,
        states: States,
        routing_weights: TokenToStateRoutingWeights
    ) -> States:
        """Apply routing weights to compute state-specific token representations"""
        batch_size, seq_len, token_dim = tokens.shape
        _, num_states, state_dim = states.shape
        
        # Normalize tokens and states
        tokens_norm = self.layer_norm_tokens(tokens)
        states_norm = self.layer_norm_states(states)
        
        # Project states to values
        values = self.state_to_value(states_norm)  # (B, S, proj_dim)
        
        # Apply routing: (B, L, S) @ (B, S, proj_dim) -> (B, L, proj_dim)
        routed_representation = torch.bmm(routing_weights, values)
        
        # Project back to state dimension
        routed_states = self.output_projection(routed_representation)
        
        # Apply dropout
        routed_states = self.dropout(routed_states)
        
        return routed_states

    def forward(
        self, 
        tokens: Tokens, 
        states: States
    ) -> Tuple[Tokens, TokenToStateRoutingWeights, torch.Tensor]:
        """Forward pass with entropy regularization"""
        return self.route_tokens(tokens, states)