# src/nstm/models/nstm_layer_optimized.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

from ..core.types_optimized import Tokens, States, OptimizedNSTMConfig
from ..core.state_manager_optimized import OptimizedStateManager
from ..core.state_propagator_optimized import OptimizedStatePropagator
from ..core.token_router_optimized import OptimizedTokenToStateRouter
from ..core.attention_optimized import OptimizedHybridAttention

class OptimizedNSMLayer(nn.Module):
    """
    Optimized NSTM layer with comprehensive improvements.
    
    Improvements:
    - Better gradient flow with residual connections
    - Layer normalization throughout
    - Improved training stability
    - Memory efficient implementation
    - Comprehensive metrics tracking
    - Gradient checkpointing support
    """
    
    def __init__(self, config: OptimizedNSTMConfig):
        super(OptimizedNSMLayer, self).__init__()
        self.config = config
        
        # Core components
        self.state_manager = OptimizedStateManager(config)
        self.state_propagator = OptimizedStatePropagator(config)
        self.token_router = OptimizedTokenToStateRouter(config)
        self.hybrid_attention = OptimizedHybridAttention(config)
        
        # Input/output projections
        if config.token_dim != config.state_dim:
            self.token_projection = nn.Linear(config.token_dim, config.state_dim)
            self.output_projection = nn.Linear(config.state_dim, config.token_dim)
        else:
            self.token_projection = None
            self.output_projection = None
            
        # Layer normalization for inputs/outputs
        self.input_layer_norm = nn.LayerNorm(config.token_dim)
        self.output_layer_norm = nn.LayerNorm(config.state_dim)
        
        # Positional encoding for sequence information
        self.positional_encoding = PositionalEncoding(config.token_dim, max_length=512)
        
        # Gradient clipping
        self.gradient_clip_norm = config.gradient_clip_norm
        
        # Metrics tracking
        self.register_buffer('step_count', torch.tensor(0))
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize metrics tracking"""
        self.metrics = {
            'routing_entropy': [],
            'attention_weights_entropy': [],
            'state_importance_mean': [],
            'gradient_norm': [],
            'states_pruned': [],
            'states_allocated': []
        }
        
    def _project_tokens(self, tokens: Tokens) -> Tokens:
        """Project tokens to state dimension if needed"""
        if self.token_projection is not None:
            return self.token_projection(tokens)
        return tokens
        
    def _project_states(self, states: States) -> States:
        """Project states back to token dimension if needed"""
        if self.output_projection is not None:
            return self.output_projection(states)
        return states
        
    def _compute_positional_encoding(self, tokens: Tokens) -> Tokens:
        """Add positional encoding to tokens"""
        return self.positional_encoding(tokens)
        
    def _update_metrics(self, 
                       routing_weights: torch.Tensor,
                       ts_weights: torch.Tensor,
                       ss_weights: torch.Tensor,
                       states: States,
                       entropy_loss: torch.Tensor):
        """Update internal metrics for monitoring"""
        with torch.no_grad():
            # Routing entropy
            routing_entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1).mean()
            self.metrics['routing_entropy'].append(routing_entropy.item())
            
            # Attention weights entropy
            ts_entropy = -(ts_weights * torch.log(ts_weights + 1e-8)).sum(dim=-1).mean()
            ss_entropy = -(ss_weights * torch.log(ss_weights + 1e-8)).sum(dim=-1).mean()
            avg_entropy = (ts_entropy + ss_entropy) / 2
            self.metrics['attention_weights_entropy'].append(avg_entropy.item())
            
            # State importance
            importance_scores = self.state_manager.get_importance_scores(states.size(0))
            self.metrics['state_importance_mean'].append(importance_scores.mean().item())
            
            # Keep only recent metrics (sliding window)
            max_history = 100
            for key in self.metrics:
                if len(self.metrics[key]) > max_history:
                    self.metrics[key] = self.metrics[key][-max_history:]

    def adaptive_state_management(self, states: States, importance_threshold: float = 0.8) -> Tuple[int, int]:
        """Adaptive state allocation and pruning based on importance and capacity"""
        current_num_states = states.size(1)
        
        # Get current importance scores
        importance_scores = self.state_manager.compute_importance_scores(states)
        avg_importance = importance_scores.mean()
        
        states_pruned = 0
        states_allocated = 0
        
        # Prune if average importance is low and we have many states
        if avg_importance < self.config.prune_threshold and current_num_states > 4:
            states_pruned = self.state_manager.prune_low_importance_states()
            
        # Allocate if average importance is high and we have capacity
        elif avg_importance > importance_threshold and current_num_states < self.config.max_states:
            num_to_allocate = min(2, self.config.max_states - current_num_states)
            states_allocated = self.state_manager.allocate_states(num_to_allocate)
            
        # Update metrics
        self.metrics['states_pruned'].append(states_pruned)
        self.metrics['states_allocated'].append(states_allocated)
        
        return states_pruned, states_allocated

    def forward(
        self, 
        tokens: Tokens, 
        states: Optional[States] = None,
        return_intermediates: bool = False
    ) -> Tuple[States, torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Optimized forward pass with comprehensive improvements.
        
        Args:
            tokens: Input tokens (batch_size, seq_len, token_dim)
            states: Previous states (optional)
            return_intermediates: Whether to return intermediate values
            
        Returns:
            Tuple of (updated_states, ts_weights, ss_weights, intermediates)
        """
        batch_size, seq_len, token_dim = tokens.shape
        
        # Input preprocessing
        tokens_norm = self.input_layer_norm(tokens)
        tokens_with_pos = self._compute_positional_encoding(tokens_norm)
        
        # Get or initialize states
        if states is None:
            states = self.state_manager(batch_size)
        else:
            if states.size(0) == 1 and batch_size > 1:
                states = states.expand(batch_size, -1, -1).contiguous()
            elif states.size(0) != batch_size:
                raise ValueError(f"State batch size mismatch: {states.size(0)} vs {batch_size}")
        
        # Store intermediate values if requested
        intermediates = {} if return_intermediates else None
        
        # 1. Token routing
        if self.config.use_gradient_checkpointing:
            routed_tokens, routing_weights, entropy_loss = torch.utils.checkpoint.checkpoint(
                self.token_router, tokens_with_pos, states, use_reentrant=False
            )
        else:
            routed_tokens, routing_weights, entropy_loss = self.token_router(tokens_with_pos, states)
            
        if return_intermediates:
            intermediates['routing_weights'] = routing_weights
            intermediates['entropy_loss'] = entropy_loss
        
        # 2. Hybrid attention
        if self.config.use_gradient_checkpointing:
            attended_states, ts_weights, ss_weights = torch.utils.checkpoint.checkpoint(
                self.hybrid_attention, tokens_with_pos, states, use_reentrant=False
            )
        else:
            attended_states, ts_weights, ss_weights = self.hybrid_attention(tokens_with_pos, states)
            
        if return_intermediates:
            intermediates['ts_weights'] = ts_weights
            intermediates['ss_weights'] = ss_weights
            intermediates['attended_states'] = attended_states
        
        # 3. State propagation
        projected_tokens = self._project_tokens(tokens_with_pos)
        if self.config.use_gradient_checkpointing:
            updated_states = torch.utils.checkpoint.checkpoint(
                self.state_propagator, attended_states, projected_tokens, routing_weights, use_reentrant=False
            )
        else:
            updated_states = self.state_propagator(attended_states, projected_tokens, routing_weights)
        
        # 4. Output layer normalization
        updated_states = self.output_layer_norm(updated_states)
        
        # 5. Update importance scores
        importance_scores = self.state_manager.compute_importance_scores(updated_states)
        self.state_manager.update_importance_scores(importance_scores)
        
        # 6. Adaptive state management
        if self.training:
            states_pruned, states_allocated = self.adaptive_state_management(updated_states)
            if states_pruned > 0 or states_allocated > 0:
                # Re-get states after management
                updated_states = self.state_manager(batch_size)
        
        # 7. Update metrics
        if self.training:
            self._update_metrics(routing_weights, ts_weights, ss_weights, updated_states, entropy_loss)
            
        # 8. Apply gradient clipping to parameters
        if self.training and self.gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
            self.metrics['gradient_norm'].append(grad_norm.item())
        
        # Update step count
        self.step_count += 1
        
        return updated_states, ts_weights, ss_weights, intermediates

    def get_states(self, batch_size: int = 1) -> States:
        """Get current states"""
        return self.state_manager.get_states(batch_size)
        
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return {k: v[-10:] if v else [] for k, v in self.metrics.items()}
        
    def reset_metrics(self):
        """Reset metrics tracking"""
        self._initialize_metrics()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'current_states': self.state_manager.states.numel(),
            'memory_mb': total_params * 4 / 1024 / 1024  # Assuming float32
        }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence information"""
    
    def __init__(self, d_model: int, max_length: int = 512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)