# src/nstm/core/state_manager.py

import torch
import torch.nn as nn
from typing import Optional
from .types import State, States, StateImportanceScores, NSTMConfig

class StateManager(nn.Module):
    """
    Class for managing states in NSTM.
    
    This class handles the creation, storage, updating based on importance scores,
    and dynamic allocation/pruning of states.
    """
    
    def __init__(self, config: NSTMConfig):
        """
        Args:
            config (NSTMConfig): NSTM configuration object.
        """
        super(StateManager, self).__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.max_states = config.max_states
        self.initial_states = min(config.initial_states, config.max_states)
        self.prune_threshold = config.prune_threshold
        
        # Buffers to store state vectors and importance scores
        # This makes the model stateful.
        self.register_buffer('states', torch.randn(1, self.initial_states, self.state_dim))
        self.register_buffer('importance_scores', torch.ones(1, self.initial_states))
        
        # There may not be a learnable parameter for state creation,
        # but it can be added here for more complex initialization methods in the future.
        
    def create_states(self, num_states: int, batch_size: int = 1) -> States:
        """
        Creates new random state vectors.
        
        Args:
            num_states (int): Number of states to create.
            batch_size (int, optional): Batch size. Default 1.
            
        Returns:
            States: Newly created states.
                    Shape: (batch_size, num_states, state_dim)
        """
        # More sophisticated initialization strategies (xavier, kaiming) can be used here.
        new_states = torch.randn(batch_size, num_states, self.state_dim)
        return new_states
    
    def get_states(self, batch_size: int = 1) -> States:
        """
        Returns the current state vectors.
        If the batch size differs from the current states' batch size,
        states are expanded to the new batch size.
        
        Args:
            batch_size (int, optional): Requested batch size. Default 1.
            
        Returns:
            States: Current state vectors.
                    Shape: (batch_size, num_current_states, state_dim)
        """
        current_states = self.states
        current_batch_size = current_states.size(0)
        
        if current_batch_size == batch_size:
            return current_states
        elif current_batch_size == 1:
            # Broadcast to requested batch size
            return current_states.expand(batch_size, -1, -1)
        else:
            # This case should ideally not happen if used correctly
            # but we handle it for robustness.
            raise ValueError(f"StateManager batch size mismatch. "
                             f"Current: {current_batch_size}, Requested: {batch_size}")
    
    def get_importance_scores(self, batch_size: int = 1) -> StateImportanceScores:
        """
        Returns the importance scores of current states.
        
        Args:
            batch_size (int, optional): Requested batch size. Default 1.
            
        Returns:
            StateImportanceScores: Current importance scores.
                                   Shape: (batch_size, num_current_states)
        """
        current_scores = self.importance_scores
        current_batch_size = current_scores.size(0)
        
        if current_batch_size == batch_size:
            return current_scores
        elif current_batch_size == 1:
            # Broadcast to requested batch size
            return current_scores.expand(batch_size, -1)
        else:
            raise ValueError(f"StateManager batch size mismatch for scores. "
                             f"Current: {current_batch_size}, Requested: {batch_size}")
    
    def update_importance_scores(self, new_scores: StateImportanceScores) -> None:
        """
        Updates the importance scores of states.
        
        Args:
            new_scores (StateImportanceScores): New importance scores.
                                                Shape: (batch_size, num_states)
        """
        # More complex update rules (e.g., moving average)
        # can be applied here, but for now, direct assignment.
        batch_size, num_scores = new_scores.shape
        
        # Check if the dimensions of scores are appropriate
        if num_scores != self.states.size(1):
            raise ValueError(f"Score dimension mismatch. "
                             f"Expected: {self.states.size(1)}, Got: {num_scores}")
        
        # Save importance scores to buffer
        # Buffers are saved in checkpoints if `persistent=True`.
        self.importance_scores = new_scores.detach()
        
    def prune_low_importance_states(self) -> int:
        """
        Deletes states with low importance scores.
        
        Returns:
            int: Number of pruned states.
        """
        # Get average scores (average over batch dimension)
        mean_scores = self.importance_scores.mean(dim=0) # (num_states,)
        
        # Determine states with scores above threshold
        keep_mask = mean_scores >= self.prune_threshold  # (num_states,)
        num_kept = keep_mask.sum().item()
        
        # If all states are to be pruned, keep at least one
        if num_kept == 0:
            # Keep the state with the highest score
            max_score_idx = mean_scores.argmax()
            keep_mask = torch.zeros_like(mean_scores, dtype=torch.bool)
            keep_mask[max_score_idx] = True
            num_kept = 1
            
        # Filter states and scores
        self.states = self.states[:, keep_mask, :]  # (1, num_kept, state_dim)
        self.importance_scores = self.importance_scores[:, keep_mask]  # (1, num_kept)
        
        # Calculate the number of pruned states
        num_pruned = self.initial_states - num_kept
        self.initial_states = num_kept
        
        return num_pruned
    
    def allocate_states(self, num_to_allocate: int) -> int:
        """
        Adds new states.
        
        Args:
            num_to_allocate (int): Number of states to allocate.
            
        Returns:
            int: Number of states actually allocated (due to max_states limit).
        """
        # Current number of states
        current_num_states = self.states.size(1)
        
        # Set to not exceed max states
        num_to_add = min(num_to_allocate, self.max_states - current_num_states)
        
        if num_to_add <= 0:
            return 0
            
        # Create new states
        new_states = self.create_states(num_to_add, batch_size=1)  # (1, num_to_add, state_dim)
        
        # Create new importance scores (initially average score)
        new_scores = torch.full((1, num_to_add), 0.5)  # (1, num_to_add)
        
        # Concatenate with existing states
        self.states = torch.cat([self.states, new_states], dim=1)  # (1, new_num_states, state_dim)
        self.importance_scores = torch.cat([self.importance_scores, new_scores], dim=1)  # (1, new_num_states)
        
        self.initial_states = self.states.size(1)
        return num_to_add

    def forward(self, batch_size: int = 1) -> States:
        """
        Returns current states. Standard forward method for nn.Module.
        
        Args:
            batch_size (int, optional): Requested batch size. Default 1.
            
        Returns:
            States: Current state vectors.
        """
        return self.get_states(batch_size)