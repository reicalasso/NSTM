# src/nstm/core/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .types import Tokens, States, NSTMConfig

class HybridAttention(nn.Module):
    """
    Hybrid attention mechanism for NSTM.
    
    This class combines two different attention mechanisms:
    1. Token-to-State Attention: Determines how tokens affect states.
    2. State-to-State Attention: Enables information exchange between states.
    
    This allows the model to consider both information from input tokens and
    interactions between internal states.
    """
    
    def __init__(self, config: NSTMConfig):
        """
        Args:
            config (NSTMConfig): NSTM configuration object.
        """
        super(HybridAttention, self).__init__()
        self.config = config
        self.token_dim = config.token_dim
        self.state_dim = config.state_dim
        self.num_heads = config.num_attention_heads
        
        if self.state_dim % self.num_heads != 0:
            raise ValueError(f"State dimension ({self.state_dim}) must be divisible "
                             f"by number of attention heads ({self.num_heads})")
        
        self.head_dim = self.state_dim // self.num_heads
        
        # Linear layers for Token-to-State Attention
        # Q: tokens, K,V: states
        self.token_to_state_q = nn.Linear(self.token_dim, self.state_dim)
        self.token_to_state_k = nn.Linear(self.state_dim, self.state_dim)
        self.token_to_state_v = nn.Linear(self.state_dim, self.state_dim)
        
        # Linear layers for State-to-State Attention
        # Q,K,V: states
        self.state_to_state_q = nn.Linear(self.state_dim, self.state_dim)
        self.state_to_state_k = nn.Linear(self.state_dim, self.state_dim)
        self.state_to_state_v = nn.Linear(self.state_dim, self.state_dim)
        
        # Output layer
        self.output_projection = nn.Linear(self.state_dim, self.state_dim)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits tensor into heads for multi-head attention.
        
        Args:
            x (Tensor): Input tensor. Shape: (batch, seq_len, state_dim)
            
        Returns:
            Tensor: Tensor split into heads. 
                    Shape: (batch, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        # (B, L, D) -> (B, L, H, D/H) -> (B, H, L, D/H)
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines multi-head attention output.
        
        Args:
            x (Tensor): Tensor split into heads. 
                        Shape: (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Tensor: Combined tensor. Shape: (batch, seq_len, state_dim)
        """
        batch_size, _, seq_len, _ = x.shape
        # (B, H, L, D/H) -> (B, L, H, D/H) -> (B, L, D)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.state_dim)

    def token_to_state_attention(
        self, 
        tokens: Tokens, 
        states: States
    ) -> Tuple[States, torch.Tensor]:
        """
        Applies attention mechanism between tokens and states.
        
        This determines which states should be affected by which tokens.
        
        Args:
            tokens (Tokens): Input tokens. Shape: (batch_size, seq_len, token_dim)
            states (States): Current state vectors. Shape: (batch_size, num_states, state_dim)
            
        Returns:
            Tuple[States, Tensor]: 
                - Attention output (states enriched with token information). 
                  Shape: (batch_size, num_states, state_dim)
                - Attention weights. Shape: (batch_size, num_heads, seq_len, num_states)
        """
        batch_size, seq_len, _ = tokens.shape
        _, num_states, _ = states.shape
        
        # Linear transformations and splitting into heads
        q = self._split_heads(self.token_to_state_q(tokens))  # (B, H, L, D/H)
        k = self._split_heads(self.token_to_state_k(states))  # (B, H, S, D/H)
        v = self._split_heads(self.token_to_state_v(states))  # (B, H, S, D/H)
        
        # Scaled dot-product attention
        # (B, H, L, D/H) @ (B, H, D/H, S) -> (B, H, L, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, S)
        
        # Weighted sum
        # (B, H, L, S) @ (B, H, S, D/H) -> (B, H, L, D/H)
        attn_output = torch.matmul(attn_weights, v)
        # Combine heads
        attn_output = self._combine_heads(attn_output)  # (B, L, D)
        
        # There's an issue here: output is (B, L, D) but we need to return (B, S, D).
        # This is based on the idea that tokens affect states.
        # One solution: We can aggregate the effect of tokens on states.
        # For example, take the average over tokens and associate it with states.
        # However, this is a bit confusing. A better approach:
        # Token-to-state attention determines which tokens each state should attend to.
        # So, the output is still (B, S, D), but carries filtered token information.
        # 
        # Alternative approach: Token-to-state updates states with token information.
        # This also makes sense. Then, tokens query states and states update themselves
        # in light of tokens.
        # 
        # In this implementation, each state takes the weighted average of related tokens.
        # But this doesn't match the tensor dimensions above.
        # 
        # The cleanest: Token-to-state tells how tokens affect states.
        # So, tokens are "values", states are "queries". 
        # In this case, output is (B, S, D).
        # 
        # However, in the code above, q's length is L (token length), 
        # k and v's length is S (number of states).
        # This is different from normal attention where q and k have the same length.
        # 
        # To resolve confusion, let's think of token-to-state as:
        # Each state finds the most related tokens to itself.
        # So, states are query (Q), tokens are key (K) and value (V).
        # In this case:
        # Q: states, K: tokens, V: tokens
        # 
        # Let's implement this approach.
        
        # Correction: Token-to-State as states querying tokens.
        # Q: states, K: tokens, V: tokens
        q_state = self._split_heads(self.token_to_state_q(states))  # (B, H, S, D/H)
        k_token = self._split_heads(self.token_to_state_k(tokens))  # (B, H, L, D/H)
        v_token = self._split_heads(self.token_to_state_v(tokens))  # (B, H, L, D/H)
        
        # (B, H, S, D/H) @ (B, H, D/H, L) -> (B, H, S, L)
        scores = torch.matmul(q_state, k_token.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights_ts = F.softmax(scores, dim=-1)  # (B, H, S, L)
        
        # Weighted sum
        # (B, H, S, L) @ (B, H, L, D/H) -> (B, H, S, D/H)
        attn_output_ts = torch.matmul(attn_weights_ts, v_token)
        # Combine heads
        attn_output_ts = self._combine_heads(attn_output_ts)  # (B, S, D)
        
        return attn_output_ts, attn_weights_ts

    def state_to_state_attention(self, states: States) -> Tuple[States, torch.Tensor]:
        """
        Applies attention mechanism between states.
        
        This enables information exchange and interaction between states.
        
        Args:
            states (States): Current state vectors. Shape: (batch_size, num_states, state_dim)
            
        Returns:
            Tuple[States, Tensor]: 
                - Attention output (states enriched with inter-state interactions). 
                  Shape: (batch_size, num_states, state_dim)
                - Attention weights. Shape: (batch_size, num_heads, num_states, num_states)
        """
        batch_size, num_states, _ = states.shape
        
        # Linear transformations and splitting into heads
        q = self._split_heads(self.state_to_state_q(states))  # (B, H, S, D/H)
        k = self._split_heads(self.state_to_state_k(states))  # (B, H, S, D/H)
        v = self._split_heads(self.state_to_state_v(states))  # (B, H, S, D/H)
        
        # Scaled dot-product attention
        # (B, H, S, D/H) @ (B, H, D/H, S) -> (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights_ss = F.softmax(scores, dim=-1)  # (B, H, S, S)
        
        # Weighted sum
        # (B, H, S, S) @ (B, H, S, D/H) -> (B, H, S, D/H)
        attn_output_ss = torch.matmul(attn_weights_ss, v)
        # Combine heads
        attn_output_ss = self._combine_heads(attn_output_ss)  # (B, S, D)
        
        return attn_output_ss, attn_weights_ss
        
    def forward(
        self, 
        tokens: Tokens, 
        states: States
    ) -> Tuple[States, torch.Tensor, torch.Tensor]:
        """
        Applies the hybrid attention mechanism.
        
        First applies token-to-state, then state-to-state attention.
        Combines the outputs.
        
        Args:
            tokens (Tokens): Input tokens. Shape: (batch_size, seq_len, token_dim)
            states (States): Current state vectors. Shape: (batch_size, num_states, state_dim)
            
        Returns:
            Tuple[States, Tensor, Tensor]: 
                - Hybrid attention output. Shape: (batch_size, num_states, state_dim)
                - Token-to-state attention weights. 
                  Shape: (batch_size, num_heads, num_states, seq_len)
                - State-to-state attention weights. 
                  Shape: (batch_size, num_heads, num_states, num_states)
        """
        # 1. Token-to-State Attention
        ts_output, ts_weights = self.token_to_state_attention(tokens, states)
        
        # 2. State-to-State Attention
        ss_output, ss_weights = self.state_to_state_attention(states)
        
        # 3. Combine outputs (here we add them, other methods possible)
        combined_output = ts_output + ss_output  # (B, S, D)
        
        # 4. Output projection
        final_output = self.output_projection(combined_output)  # (B, S, D)
        
        return final_output, ts_weights, ss_weights