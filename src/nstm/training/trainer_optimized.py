# src/nstm/training/trainer_optimized.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class OptimizedTrainer:
    """
    Advanced trainer with comprehensive features for NSTM optimization.
    
    Features:
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Comprehensive metrics tracking
    - Gradient monitoring
    - Memory usage tracking
    - Advanced visualization
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 checkpoint_dir: str = './checkpoints'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('early_stopping_min_delta', 1e-4)
        )
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'gradient_norm': [],
            'model_metrics': [],
            'memory_usage': [],
            'epoch_time': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('max_epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == 'warmup_cosine':
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=self.config.get('warmup_steps', 1000),
                total_steps=len(self.train_loader) * self.config.get('max_epochs', 100)
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 10),
                gamma=self.config.get('lr_gamma', 0.1)
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        for batch_idx, (input_seq, target_seq) in enumerate(self.train_loader):
            # Move to device
            input_seq = input_seq.to(next(self.model.parameters()).device)
            target_seq = target_seq.to(next(self.model.parameters()).device)
            
            # Forward pass
            batch_size, input_len = input_seq.shape
            _, target_len = target_seq.shape
            
            # Get embeddings (assuming embedding layer is separate)
            embedded_input = self.model.embedding(input_seq) if hasattr(self.model, 'embedding') else input_seq
            
            # Conditioning phase
            conditioning_input = embedded_input[:, :-1, :]  # Remove end token
            final_states, ts_weights, ss_weights, intermediates = self.model(
                conditioning_input, return_intermediates=True
            )
            
            # Generation phase
            selected_states = final_states[:, :target_len, :]
            logits = self.model.output_layer(selected_states) if hasattr(self.model, 'output_layer') else selected_states
            
            # Compute loss
            loss = self.criterion(logits.transpose(1, 2), target_seq)
            
            # Add entropy regularization if available
            if intermediates and 'entropy_loss' in intermediates:
                loss = loss + intermediates['entropy_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('gradient_clip_norm', 1.0)
            )
            
            self.optimizer.step()
            
            # Update scheduler if it's step-based
            if isinstance(self.scheduler, WarmupCosineScheduler):
                self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == target_seq).float().mean().item()
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            self.global_step += 1
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 50) == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Accuracy: {accuracy:.4f}, "
                      f"Grad Norm: {grad_norm:.4f}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        epoch_time = time.time() - epoch_start_time
        
        # Get model-specific metrics
        model_metrics = self.model.get_metrics() if hasattr(self.model, 'get_metrics') else {}
        memory_usage = self.model.get_memory_usage() if hasattr(self.model, 'get_memory_usage') else {}
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'gradient_norm': grad_norm.item(),
            'model_metrics': model_metrics,
            'memory_usage': memory_usage,
            'epoch_time': epoch_time
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_seq, target_seq in self.val_loader:
                # Move to device
                input_seq = input_seq.to(next(self.model.parameters()).device)
                target_seq = target_seq.to(next(self.model.parameters()).device)
                
                # Forward pass
                batch_size, input_len = input_seq.shape
                _, target_len = target_seq.shape
                
                # Get embeddings
                embedded_input = self.model.embedding(input_seq) if hasattr(self.model, 'embedding') else input_seq
                
                # Conditioning phase
                conditioning_input = embedded_input[:, :-1, :]
                final_states, ts_weights, ss_weights, _ = self.model(conditioning_input)
                
                # Generation phase
                selected_states = final_states[:, :target_len, :]
                logits = self.model.output_layer(selected_states) if hasattr(self.model, 'output_layer') else selected_states
                
                # Compute loss
                loss = self.criterion(logits.transpose(1, 2), target_seq)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == target_seq).float().mean().item()
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update scheduler if it's epoch-based
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            elif not isinstance(self.scheduler, WarmupCosineScheduler):
                self.scheduler.step()
            
            # Update metrics
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['train_accuracy'].append(train_metrics['accuracy'])
            self.metrics['val_accuracy'].append(val_metrics['accuracy'])
            self.metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.metrics['gradient_norm'].append(train_metrics['gradient_norm'])
            self.metrics['model_metrics'].append(train_metrics['model_metrics'])
            self.metrics['memory_usage'].append(train_metrics['memory_usage'])
            self.metrics['epoch_time'].append(train_metrics['epoch_time'])
            
            # Print epoch summary
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                  f"Time: {train_metrics['epoch_time']:.2f}s")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint('best_model.pt')
                print("New best model saved!")
            
            # Check early stopping
            if self.early_stopping(val_metrics['loss']):
                print("Early stopping triggered!")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights.")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'config': self.config
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.checkpoint_dir / filename)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics = checkpoint['metrics']
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(self.metrics['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.metrics['train_accuracy'], label='Train')
        axes[0, 1].plot(self.metrics['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[0, 2].plot(self.metrics['learning_rate'])
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('LR')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True)
        
        # Gradient norm
        axes[1, 0].plot(self.metrics['gradient_norm'])
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Grad Norm')
        axes[1, 0].grid(True)
        
        # Memory usage
        if self.metrics['memory_usage']:
            memory_mb = [m.get('memory_mb', 0) for m in self.metrics['memory_usage']]
            axes[1, 1].plot(memory_mb)
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].grid(True)
        
        # Epoch time
        axes[1, 2].plot(self.metrics['epoch_time'])
        axes[1, 2].set_title('Epoch Time')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (s)')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class WarmupCosineScheduler:
    """Warmup cosine annealing scheduler"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def state_dict(self):
        return {'current_step': self.current_step}
        
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']