# src/nstm/core/token_router.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .types import Tokens, States, TokenToStateRoutingWeights, NSTMConfig

class TokenToStateRouter(nn.Module):
    """
    Class that determines which states input tokens should be associated with.
    
    This class computes a routing weight matrix between tokens and states. These weights
    determine which token should affect which state. This makes information flow more
    efficient and selective.
    """
    
    def __init__(self, config: NSTMConfig):
        """
        Args:
            config (NSTMConfig): NSTM configuration object.
        """
        super(TokenToStateRouter, self).__init__()
        self.config = config
        self.token_dim = config.token_dim
        self.state_dim = config.state_dim
        
        # Linear layer to transform tokens into query vectors
        self.token_to_query = nn.Linear(self.token_dim, self.state_dim)
        
        # Linear layer to transform states into key vectors
        # Since each state is already state_dim long, it can be used directly.
        # However, a learnable transformation can be added.
        # For example, to learn "key" versions of states:
        # self.state_to_key = nn.Linear(self.state_dim, self.state_dim)
        # For now, we use state vectors directly.
        
    def compute_routing_weights(
        self, 
        tokens: Tokens, 
        states: States
    ) -> TokenToStateRoutingWeights:
        """
        Computes routing weights between tokens and states.
        
        Method: Learnable dot-product attention.
        1. Transform tokens into query vectors.
        2. Use states as key vectors.
        3. Compute similarity via dot product.
        4. Normalize weights with softmax.
        
        Args:
            tokens (Tokens): Input tokens. Shape: (batch_size, seq_len, token_dim)
            states (States): Current state vectors. Shape: (batch_size, num_states, state_dim)
            
        Returns:
            TokenToStateRoutingWeights: Routing weights. 
                                      Shape: (batch_size, seq_len, num_states)
        """
        batch_size, seq_len, token_dim = tokens.shape
        _, num_states, state_dim = states.shape
        
        if token_dim != self.token_dim:
            raise ValueError(f"Token dimension mismatch. "
                             f"Expected: {self.token_dim}, Got: {token_dim}")
        if state_dim != self.state_dim:
            raise ValueError(f"State dimension mismatch. "
                             f"Expected: {self.state_dim}, Got: {state_dim}")
                             
        # 1. Transform tokens into query vectors
        queries = self.token_to_query(tokens)  # (B, L, D)
        
        # 2. Use states as key vectors (or self.state_to_key(states))
        keys = states  # (B, S, D)
        
        # 3. Dot product attention (scaled)
        # (B, L, D) @ (B, D, S) -> (B, L, S)
        # scaling factor
        scale = self.state_dim ** 0.5  
        routing_logits = torch.bmm(queries, keys.transpose(-2, -1)) / scale
        
        # 4. Normalize weights with softmax
        routing_weights = F.softmax(routing_logits, dim=-1)  # (B, L, S)
        
        return routing_weights
        
    def route_tokens(
        self, 
        tokens: Tokens, 
        states: States
    ) -> Tuple[Tokens, TokenToStateRoutingWeights]:
        """
        Routes tokens to states based on computed routing weights.
        
        This function determines how tokens should be weighted by states, but does not
        directly route the tokens themselves. The routing weights are used by other
        components like `StatePropagator`.
        
        Args:
            tokens (Tokens): Input tokens. Shape: (batch_size, seq_len, token_dim)
            states (States): Current state vectors. Shape: (batch_size, num_states, state_dim)
            
        Returns:
            Tuple[Tokens, TokenToStateRoutingWeights]: 
                - Routed tokens (in this implementation, original tokens are returned)
                - Routing weights
        """
        # Compute routing weights
        routing_weights = self.compute_routing_weights(tokens, states)
        
        # In this implementation, tokens themselves are not routed, 
        # only weights are returned.
        # Actual routing (e.g., weighted average) 
        # is done by `StatePropagator`.
        
        return tokens, routing_weights

    def forward(
        self, 
        tokens: Tokens, 
        states: States
    ) -> Tuple[Tokens, TokenToStateRoutingWeights]:
        """
        Standard forward method for routing operation.
        
        Args:
            tokens (Tokens): Input tokens.
            states (States): Current states.
            
        Returns:
            Tuple[Tokens, TokenToStateRoutingWeights]: Routed tokens 
                                                     and routing weights.
        """
        return self.route_tokens(tokens, states)