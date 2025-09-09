# src/nstm/core/state_propagator.py

import torch
import torch.nn as nn
from typing import Optional
from .types import State, States, Token, NSTMConfig

class StatePropagator(nn.Module):
    """
    Class that manages how states are updated and inter-state communication in NSTM.
    
    This class performs state updates using gate mechanisms like LSTM or GRU. It can also
    integrate attention mechanisms for inter-state communication.
    """
    
    def __init__(self, config: NSTMConfig):
        """
        Args:
            config (NSTMConfig): NSTM configuration object.
        """
        super(StatePropagator, self).__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.gate_type = config.gate_type.lower()
        
        if self.gate_type == 'gru':
            # Linear layers for GRU-like gate mechanism
            # [h_prev, input] -> [update_gate, reset_gate, proposal]
            # Output up to state_dim for each gate
            gate_input_dim = self.state_dim * 2  # [prev_state, input]
            self.gate_layer = nn.Linear(gate_input_dim, self.state_dim * 3)
            
        elif self.gate_type == 'lstm':
            # LSTM-like gate mechanism
            # [h_prev, input] -> [forget_gate, input_gate, output_gate, proposal]
            gate_input_dim = self.state_dim * 2
            self.gate_layer = nn.Linear(gate_input_dim, self.state_dim * 4)
            
            # Tracking a cell state for LSTM may be necessary.
            # However, in this simplified version, the output state (hidden state) 
            # can act like an "internal" cell state.
            # A separate cell state can be maintained for a more complex implementation.
        else:
            raise ValueError(f"Unsupported gate type: {self.gate_type}. "
                             f"Supported types: 'gru', 'lstm'")
    
    def forward_single_state(self, prev_state: State, input_token: Token) -> State:
        """
        Computes how a single state is updated.
        
        Args:
            prev_state (State): Previous state vector. Shape: (batch_size, state_dim)
            input_token (Token): New input token. Shape: (batch_size, state_dim)
                                (Note: Here, it is assumed that the token is already 
                                 the correct size. In a real application, an embedding 
                                 layer is needed.)
            
        Returns:
            State: Updated state vector. Shape: (batch_size, state_dim)
        """
        # 1. Concatenate inputs
        concat_input = torch.cat([prev_state, input_token], dim=-1)  # (batch, state_dim*2)
        
        if self.gate_type == 'gru':
            # 2. Compute all gates
            gates = self.gate_layer(concat_input)  # (batch, state_dim * 3)
            update_gate, reset_gate, proposal_input = gates.chunk(3, dim=-1)
            
            # 3. Apply sigmoid and tanh activations
            update_gate = torch.sigmoid(update_gate)
            reset_gate = torch.sigmoid(reset_gate)
            proposal_input = torch.tanh(
                self.gate_layer(
                    torch.cat([reset_gate * prev_state, input_token], dim=-1)
                ).chunk(3, dim=-1)[2]
            )
            
            # 4. Compute new state
            new_state = (1 - update_gate) * prev_state + update_gate * proposal_input
            return new_state
            
        elif self.gate_type == 'lstm':
            # 2. Compute all gates
            gates = self.gate_layer(concat_input)  # (batch, state_dim * 4)
            forget_gate, input_gate, output_gate, proposal_input = gates.chunk(4, dim=-1)
            
            # 3. Apply sigmoid and tanh activations
            forget_gate = torch.sigmoid(forget_gate)
            input_gate = torch.sigmoid(input_gate)
            output_gate = torch.sigmoid(output_gate)
            proposal_input = torch.tanh(proposal_input)
            
            # 4. Compute cell state and new hidden state
            # Simplification: Using previous state as cell state.
            # In real LSTM, a separate cell state is maintained.
            cell_state_candidate = forget_gate * prev_state + input_gate * proposal_input
            new_state = output_gate * torch.tanh(cell_state_candidate)
            return new_state
    
    def forward_multiple_states(
        self, 
        prev_states: States, 
        input_tokens: Tokens,
        routing_weights: Optional[torch.Tensor] = None
    ) -> States:
        """
        Computes state updates in the case of multiple states and tokens. If tokens are routed 
        to states (routing_weights), each state uses relevant tokens. Otherwise, all tokens 
        are considered by all states.
        
        Args:
            prev_states (States): Previous state vectors. 
                                  Shape: (batch_size, num_states, state_dim)
            input_tokens (Tokens): New input tokens. 
                                   Shape: (batch_size, seq_len, token_dim)
            routing_weights (Optional[Tensor]): Token-to-state routing weights.
                                               Shape: (batch_size, seq_len, num_states)
                                               If None, all tokens are used for all states.
            
        Returns:
            States: Updated state vectors. 
                    Shape: (batch_size, num_states, state_dim)
        """
        batch_size, num_states, state_dim = prev_states.shape
        _, seq_len, token_dim = input_tokens.shape
        
        if token_dim != state_dim:
            raise ValueError(f"Input token dimension ({token_dim}) must match "
                             f"state dimension ({state_dim}) for propagation.")
        
        # Compute or use routing weights
        if routing_weights is not None:
            # Check dimensions of weights
            if routing_weights.shape != (batch_size, seq_len, num_states):
                raise ValueError(f"Routing weights shape mismatch. "
                                 f"Expected: {(batch_size, seq_len, num_states)}, "
                                 f"Got: {routing_weights.shape}")
            # Normalize weights (like a probability distribution)
            routing_weights = torch.softmax(routing_weights, dim=-1)  # (B, L, S)
            
            # Route tokens to states using weighted average
            # (B, L, S) @ (B, L, D) -> (B, S, D)
            routed_inputs = torch.bmm(routing_weights.transpose(-1, -2), input_tokens)
        else:
            # If no routing, take average of all tokens
            # This means all tokens affect all states
            # (B, L, D) -> (B, 1, D) -> (B, S, D)
            avg_input = input_tokens.mean(dim=1, keepdim=True)  # (B, 1, D)
            routed_inputs = avg_input.expand(-1, num_states, -1)  # (B, S, D)
            
        # Update each state individually
        updated_states = []
        for i in range(num_states):
            # Input for i-th state
            state_input = routed_inputs[:, i, :]  # (B, D)
            # i-th previous state
            prev_state = prev_states[:, i, :]     # (B, D)
            # Update state
            new_state = self.forward_single_state(prev_state, state_input)  # (B, D)
            updated_states.append(new_state)
            
        # Convert list to tensor
        # [(B, D), ...] * S -> (B, S, D)
        updated_states_tensor = torch.stack(updated_states, dim=1)
        return updated_states_tensor

    def forward(
        self, 
        prev_states: States, 
        input_tokens: Tokens,
        routing_weights: Optional[torch.Tensor] = None
    ) -> States:
        """
        Updates states. This is the standard forward method for nn.Module.
        Can work with a single state or multiple states.
        
        Args:
            prev_states (States): Previous states.
            input_tokens (Tokens): New tokens.
            routing_weights (Optional[Tensor]): Routing weights.
            
        Returns:
            States: Updated states.
        """
        # Decide based on dimensions: single or multiple processing
        batch_size, num_states, state_dim = prev_states.shape
        
        if num_states == 1:
            # Single state assumption: input_tokens should be (B, 1, D)
            if input_tokens.shape[1] != 1:
                # Alternative: take average of input_tokens
                input_token = input_tokens.mean(dim=1, keepdim=True)  # (B, 1, D)
            else:
                input_token = input_tokens  # (B, 1, D)
                
            prev_state = prev_states.squeeze(1)  # (B, D)
            input_token_squeezed = input_token.squeeze(1)  # (B, D)
            new_state = self.forward_single_state(prev_state, input_token_squeezed)  # (B, D)
            return new_state.unsqueeze(1)  # (B, 1, D)
        else:
            # Multiple states
            return self.forward_multiple_states(prev_states, input_tokens, routing_weights)