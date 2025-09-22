# src/nstm/core/state_manager_optimized.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .types import State, States, StateImportanceScores, NSTMConfig

class OptimizedStateManager(nn.Module):
    """
    Optimized state manager with improved numerical stability and efficiency.
    
    Improvements:
    - Better state initialization strategies
    - Exponential moving average for importance scores
    - Adaptive pruning thresholds
    - Memory efficient operations
    - Gradient-friendly importance computation
    """
    
    def __init__(self, config: NSTMConfig):
        super(OptimizedStateManager, self).__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.max_states = config.max_states
        self.initial_states = min(config.initial_states, config.max_states)
        self.prune_threshold = config.prune_threshold
        
        # EMA parameters for importance scores
        self.ema_decay = getattr(config, 'importance_ema_decay', 0.9)
        self.adaptive_threshold = getattr(config, 'adaptive_threshold', True)
        
        # Learnable state initialization
        self.state_init_linear = nn.Linear(1, self.state_dim)
        self.importance_predictor = nn.Linear(self.state_dim, 1)
        
        # Initialize states with better strategy
        self._initialize_states()
        
    def _initialize_states(self):
        """Initialize states using learnable parameters"""
        # Use Xavier initialization for better gradient flow
        init_states = torch.randn(1, self.initial_states, self.state_dim) * 0.1
        nn.init.xavier_uniform_(init_states)
        
        self.register_buffer('states', init_states)
        self.register_buffer('importance_scores', torch.ones(1, self.initial_states) * 0.5)
        self.register_buffer('importance_history', torch.ones(1, self.initial_states) * 0.5)
        
    def create_states(self, num_states: int, batch_size: int = 1) -> States:
        """Create new states with learnable initialization"""
        # Use learnable initialization instead of random
        dummy_input = torch.ones(batch_size, num_states, 1, device=self.states.device)
        new_states = self.state_init_linear(dummy_input)
        
        # Add small random noise for diversity
        noise = torch.randn_like(new_states) * 0.01
        new_states = new_states + noise
        
        return new_states
    
    def get_states(self, batch_size: int = 1) -> States:
        """Get current states with efficient broadcasting"""
        current_states = self.states
        current_batch_size = current_states.size(0)
        
        if current_batch_size != batch_size:
            if current_batch_size == 1:
                return current_states.expand(batch_size, -1, -1).contiguous()
            else:
                raise ValueError(f"StateManager batch size mismatch. "
                               f"Current: {current_batch_size}, Requested: {batch_size}")
        return current_states
    
    def get_importance_scores(self, batch_size: int = 1) -> StateImportanceScores:
        """Get importance scores with efficient broadcasting"""
        current_scores = self.importance_scores
        current_batch_size = current_scores.size(0)
        
        if current_batch_size != batch_size:
            if current_batch_size == 1:
                return current_scores.expand(batch_size, -1).contiguous()
            else:
                raise ValueError(f"StateManager batch size mismatch for scores. "
                               f"Current: {current_batch_size}, Requested: {batch_size}")
        return current_scores
    
    def compute_importance_scores(self, states: States) -> StateImportanceScores:
        """Compute importance scores using learned predictor"""
        # Use learned predictor for importance
        raw_scores = self.importance_predictor(states).squeeze(-1)  # (B, S)
        
        # Apply sigmoid to ensure scores are in [0, 1]
        importance_scores = torch.sigmoid(raw_scores)
        
        return importance_scores
        
    def update_importance_scores(self, new_scores: StateImportanceScores) -> None:
        """Update importance scores with exponential moving average"""
        batch_size, num_scores = new_scores.shape
        
        if num_scores != self.states.size(1):
            raise ValueError(f"Score dimension mismatch. "
                           f"Expected: {self.states.size(1)}, Got: {num_scores}")
        
        # Take average over batch dimension for storage
        avg_new_scores = new_scores.mean(dim=0, keepdim=True)
        
        # Apply exponential moving average
        self.importance_scores = (self.ema_decay * self.importance_scores + 
                                (1 - self.ema_decay) * avg_new_scores.detach())
        
        # Update history for adaptive threshold
        self.importance_history = (0.95 * self.importance_history + 
                                 0.05 * avg_new_scores.detach())
        
    def get_adaptive_threshold(self) -> float:
        """Compute adaptive pruning threshold based on score distribution"""
        if not self.adaptive_threshold:
            return self.prune_threshold
            
        # Use percentile-based threshold
        scores_flat = self.importance_history.view(-1)
        threshold = torch.quantile(scores_flat, self.prune_threshold).item()
        
        # Ensure minimum threshold
        min_threshold = 0.1
        return max(threshold, min_threshold)
        
    def prune_low_importance_states(self) -> int:
        """Prune states with adaptive threshold and safety checks"""
        # Get adaptive threshold
        threshold = self.get_adaptive_threshold()
        
        # Get mean scores across batch dimension
        mean_scores = self.importance_scores.mean(dim=0)
        
        # Determine states to keep
        keep_mask = mean_scores >= threshold
        num_kept = keep_mask.sum().item()
        
        # Safety check: keep at least 25% of states or minimum 2 states
        min_keep = max(2, int(0.25 * self.states.size(1)))
        if num_kept < min_keep:
            # Keep top states if too many would be pruned
            _, top_indices = torch.topk(mean_scores, min_keep)
            keep_mask = torch.zeros_like(mean_scores, dtype=torch.bool)
            keep_mask[top_indices] = True
            num_kept = min_keep
            
        # Apply pruning
        if num_kept < self.states.size(1):
            self.states = self.states[:, keep_mask, :]
            self.importance_scores = self.importance_scores[:, keep_mask]
            self.importance_history = self.importance_history[:, keep_mask]
            
        num_pruned = self.initial_states - num_kept
        self.initial_states = num_kept
        
        return max(0, num_pruned)
    
    def allocate_states(self, num_to_allocate: int) -> int:
        """Allocate new states with improved initialization"""
        current_num_states = self.states.size(1)
        num_to_add = min(num_to_allocate, self.max_states - current_num_states)
        
        if num_to_add <= 0:
            return 0
            
        # Create new states with learnable initialization
        new_states = self.create_states(num_to_add, batch_size=1)
        
        # Initialize importance scores as average of existing scores
        avg_importance = self.importance_scores.mean()
        new_scores = torch.full((1, num_to_add), avg_importance.item(), 
                              device=self.importance_scores.device)
        new_history = torch.full((1, num_to_add), avg_importance.item(),
                               device=self.importance_history.device)
        
        # Concatenate with existing states
        self.states = torch.cat([self.states, new_states], dim=1)
        self.importance_scores = torch.cat([self.importance_scores, new_scores], dim=1)
        self.importance_history = torch.cat([self.importance_history, new_history], dim=1)
        
        self.initial_states = self.states.size(1)
        return num_to_add

    def forward(self, batch_size: int = 1) -> States:
        """Forward pass returning current states"""
        return self.get_states(batch_size)