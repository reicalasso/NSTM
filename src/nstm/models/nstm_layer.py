# src/nstm/models/nstm_layer.py

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..core.types import Tokens, States, NSTMConfig
from ..core.state_manager import StateManager
from ..core.state_propagator import StatePropagator
from ..core.token_router import TokenToStateRouter
from ..core.attention import HybridAttention

class NSMLayer(nn.Module):
    """
    Core layer of NSTM.
    
    This class composes all core components (StateManager, StatePropagator,
    TokenToStateRouter, HybridAttention) and coordinates their operation.
    
    This layer accepts a sequence of tokens, updates the current states, and
    returns the new state vectors.
    """
    
    def __init__(self, config: NSTMConfig):
        """
        Args:
            config (NSTMConfig): NSTM configuration object.
        """
        super(NSMLayer, self).__init__()
        self.config = config
        
        # Instantiate core components
        self.state_manager = StateManager(config)
        self.state_propagator = StatePropagator(config)
        self.token_router = TokenToStateRouter(config)
        self.hybrid_attention = HybridAttention(config)
        
    def forward(
        self, 
        tokens: Tokens, 
        states: Optional[States] = None
    ) -> Tuple[States, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the NSM layer.
        
        Workflow:
        1. Obtain (or create) current states
        2. Route tokens to states
        3. Apply the hybrid attention mechanism
        4. Update states
        
        Args:
            tokens (Tokens): Input tokens. Shape: (batch_size, seq_len, token_dim)
            states (Optional[States]): Previous states.
                                     Shape: (batch_size, num_states, state_dim)
                                     If None, states are obtained from StateManager.
            
        Returns:
            Tuple[States, Tensor, Tensor]: 
                - updated_states: Updated states. 
                                Shape: (batch_size, num_states, state_dim)
                - ts_attn_weights: Token-to-state attention weights. 
                                 Shape: (batch_size, num_heads, num_states, seq_len)
                - ss_attn_weights: State-to-state attention weights. 
                                 Shape: (batch_size, num_heads, num_states, num_states)
        """
        batch_size, seq_len, token_dim = tokens.shape
        
        # 1. Obtain current states
        if states is None:
            # Get states from StateManager
            states = self.state_manager(batch_size)  # (B, S, D)
        else:
            # Check state batch dimension
            if states.size(0) == 1 and batch_size > 1:
                # Broadcast
                states = states.expand(batch_size, -1, -1)  # (B, S, D)
            elif states.size(0) != batch_size:
                raise ValueError(f"State batch size mismatch. "
                                 f"States: {states.size(0)}, Tokens: {batch_size}")
        
        # 2. Route tokens to states
        # This step determines which states each token is associated with
        _, routing_weights = self.token_router(tokens, states)
        # routing_weights: (B, L, S)
        
        # 3. Apply hybrid attention mechanism
        # This step computes interactions both between tokens and states and among states themselves.
        attended_states, ts_weights, ss_weights = self.hybrid_attention(tokens, states)
        # attended_states: (B, S, D)
        # ts_weights: (B, H, S, L)
        # ss_weights: (B, H, S, S)
        
        # 4. Update states
        # StatePropagator computes new states using attended_states and original token information.
        # Note: StatePropagator's design can be a bit confusing.
        # Simplified approach:
        # attended_states are already enriched by attention.
        # Token information is associated via routing_weights.
        # New states can be a transformation of attended_states.
        # 
        # Alternatively, we can use StatePropagator with its original design:
        # prev_states = states
        # input_tokens = tokens
        # updated_states = self.state_propagator(prev_states, input_tokens, routing_weights)
        # 
        # Which is more appropriate?
        # 
        # StatePropagator takes prev_states and input_tokens.
        # routing_weights indicate how input_tokens are routed to prev_states.
        # This aligns better with the original design.
        updated_states = self.state_propagator(states, tokens, routing_weights)
        
        return updated_states, ts_weights, ss_weights

    def get_states(self, batch_size: int = 1) -> States:
        """
        Return current states.
        
        Args:
            batch_size (int, optional): Desired batch size. Default 1.
            
        Returns:
            States: Current states.
        """
        return self.state_manager.get_states(batch_size)
        
    def update_state_importance(self, new_scores: torch.Tensor) -> None:
        """
        Update importance scores for states.
        
        Args:
            new_scores (Tensor): New importance scores. 
                               Shape: (batch_size, num_states)
        """
        self.state_manager.update_importance_scores(new_scores)
        
    def prune_states(self) -> int:
        """
        Prune states with low importance scores.
        
        Returns:
            int: Number of pruned states.
        """
        return self.state_manager.prune_low_importance_states()
        
    def allocate_states(self, num_to_allocate: int) -> int:
        """
        Allocate new states.
        
        Args:
            num_to_allocate (int): Number of states to allocate.
            
        Returns:
            int: Actual number of allocated states.
        """
        return self.state_manager.allocate_states(num_to_allocate)