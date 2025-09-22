# src/nstm/data/dataset_optimized.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List
import random

class OptimizedCopyTaskDataset(Dataset):
    """
    Enhanced Copy Task dataset with improvements:
    - Variable sequence lengths
    - Curriculum learning support
    - Data augmentation
    - Noise injection
    - Multiple difficulty levels
    """
    
    def __init__(self, 
                 min_sequence_length: int = 5,
                 max_sequence_length: int = 20,
                 num_samples: int = 1000,
                 vocab_size: int = 8,
                 difficulty_level: str = 'easy',
                 add_noise: bool = False,
                 noise_prob: float = 0.1,
                 curriculum_learning: bool = True):
        """
        Args:
            min_sequence_length: Minimum sequence length
            max_sequence_length: Maximum sequence length
            num_samples: Number of samples in dataset
            vocab_size: Vocabulary size (excluding special tokens)
            difficulty_level: 'easy', 'medium', 'hard'
            add_noise: Whether to add noise to sequences
            noise_prob: Probability of noise injection
            curriculum_learning: Whether to use curriculum learning
        """
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.difficulty_level = difficulty_level
        self.add_noise = add_noise
        self.noise_prob = noise_prob
        self.curriculum_learning = curriculum_learning
        
        # Special tokens
        self.end_token = vocab_size      # End-of-sequence token
        self.pad_token = vocab_size + 1  # Padding token
        self.noise_token = vocab_size + 2  # Noise token
        self.total_vocab_size = vocab_size + 3
        
        # Generate samples based on difficulty
        self.samples = self._generate_samples()
        
        # Current curriculum step
        self.curriculum_step = 0
        
    def _generate_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate dataset samples"""
        samples = []
        
        for _ in range(self.num_samples):
            # Variable sequence length based on difficulty
            if self.difficulty_level == 'easy':
                seq_len = random.randint(self.min_sequence_length, 
                                       min(self.max_sequence_length, 10))
            elif self.difficulty_level == 'medium':
                seq_len = random.randint(max(self.min_sequence_length, 8),
                                       min(self.max_sequence_length, 15))
            else:  # hard
                seq_len = random.randint(max(self.min_sequence_length, 12),
                                       self.max_sequence_length)
            
            # Generate sequence
            sequence = self._generate_sequence(seq_len)
            
            # Create input and target
            input_seq, target_seq = self._create_input_target(sequence)
            
            samples.append((input_seq, target_seq))
            
        return samples
    
    def _generate_sequence(self, length: int) -> torch.Tensor:
        """Generate a single sequence"""
        if self.difficulty_level == 'easy':
            # Simple random sequence
            sequence = torch.randint(0, self.vocab_size, (length,))
        elif self.difficulty_level == 'medium':
            # Sequences with some patterns
            sequence = self._generate_patterned_sequence(length)
        else:  # hard
            # Complex sequences with dependencies
            sequence = self._generate_complex_sequence(length)
            
        # Add noise if specified
        if self.add_noise and random.random() < self.noise_prob:
            sequence = self._add_noise(sequence)
            
        return sequence
    
    def _generate_patterned_sequence(self, length: int) -> torch.Tensor:
        """Generate sequence with patterns (for medium difficulty)"""
        sequence = []
        
        # Add some repetitive patterns
        pattern_length = random.randint(2, 4)
        pattern = torch.randint(0, self.vocab_size, (pattern_length,))
        
        pos = 0
        while pos < length:
            if pos + pattern_length <= length and random.random() < 0.3:
                # Add pattern
                sequence.extend(pattern.tolist())
                pos += pattern_length
            else:
                # Add random token
                sequence.append(random.randint(0, self.vocab_size - 1))
                pos += 1
                
        return torch.tensor(sequence[:length])
    
    def _generate_complex_sequence(self, length: int) -> torch.Tensor:
        """Generate complex sequence with dependencies (for hard difficulty)"""
        sequence = []
        
        # Add long-range dependencies
        for i in range(length):
            if i > 5 and random.random() < 0.2:
                # Create dependency on earlier token
                dep_pos = random.randint(0, i - 3)
                sequence.append(sequence[dep_pos])
            else:
                sequence.append(random.randint(0, self.vocab_size - 1))
                
        return torch.tensor(sequence)
    
    def _add_noise(self, sequence: torch.Tensor) -> torch.Tensor:
        """Add noise to sequence"""
        sequence = sequence.clone()
        
        # Random token substitution
        num_substitutions = random.randint(1, max(1, len(sequence) // 4))
        positions = random.sample(range(len(sequence)), num_substitutions)
        
        for pos in positions:
            if random.random() < 0.5:
                # Replace with random token
                sequence[pos] = random.randint(0, self.vocab_size - 1)
            else:
                # Replace with noise token
                sequence[pos] = self.noise_token
                
        return sequence
    
    def _create_input_target(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create input and target sequences"""
        # Input: [sequence, end_token]
        input_seq = torch.cat([sequence, torch.tensor([self.end_token])])
        
        # Target: [sequence]
        target_seq = sequence.clone()
        
        return input_seq, target_seq
    
    def update_curriculum(self, step: int):
        """Update curriculum learning step"""
        self.curriculum_step = step
        
        if self.curriculum_learning:
            # Gradually increase difficulty
            if step < 1000:
                self.difficulty_level = 'easy'
                self.max_sequence_length = min(10, self.max_sequence_length)
            elif step < 3000:
                self.difficulty_level = 'medium'
                self.max_sequence_length = min(15, self.max_sequence_length)
            else:
                self.difficulty_level = 'hard'
                # Use full max_sequence_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class CopyTaskCollator:
    """Custom collator for copy task with padding"""
    
    def __init__(self, pad_token: int):
        self.pad_token = pad_token
        
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate batch with padding"""
        input_seqs, target_seqs = zip(*batch)
        
        # Pad input sequences
        max_input_len = max(seq.size(0) for seq in input_seqs)
        padded_inputs = []
        
        for seq in input_seqs:
            padded = torch.full((max_input_len,), self.pad_token, dtype=seq.dtype)
            padded[:seq.size(0)] = seq
            padded_inputs.append(padded)
            
        # Pad target sequences
        max_target_len = max(seq.size(0) for seq in target_seqs)
        padded_targets = []
        
        for seq in target_seqs:
            padded = torch.full((max_target_len,), self.pad_token, dtype=seq.dtype)
            padded[:seq.size(0)] = seq
            padded_targets.append(padded)
            
        return torch.stack(padded_inputs), torch.stack(padded_targets)


def create_optimized_dataloaders(
    config: dict,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create optimized data loaders for copy task"""
    
    # Calculate sample sizes
    total_samples = config.get('total_samples', 10000)
    train_samples = int(total_samples * train_split)
    val_samples = int(total_samples * val_split)
    test_samples = total_samples - train_samples - val_samples
    
    # Create datasets
    train_dataset = OptimizedCopyTaskDataset(
        min_sequence_length=config.get('min_seq_len', 5),
        max_sequence_length=config.get('max_seq_len', 20),
        num_samples=train_samples,
        vocab_size=config.get('vocab_size', 8),
        difficulty_level='easy',
        add_noise=config.get('add_noise', True),
        noise_prob=config.get('noise_prob', 0.1),
        curriculum_learning=config.get('curriculum_learning', True)
    )
    
    val_dataset = OptimizedCopyTaskDataset(
        min_sequence_length=config.get('min_seq_len', 5),
        max_sequence_length=config.get('max_seq_len', 20),
        num_samples=val_samples,
        vocab_size=config.get('vocab_size', 8),
        difficulty_level='medium',
        add_noise=False,  # No noise in validation
        curriculum_learning=False
    )
    
    test_dataset = OptimizedCopyTaskDataset(
        min_sequence_length=config.get('min_seq_len', 5),
        max_sequence_length=config.get('max_seq_len', 20),
        num_samples=test_samples,
        vocab_size=config.get('vocab_size', 8),
        difficulty_level='hard',
        add_noise=False,  # No noise in test
        curriculum_learning=False
    )
    
    # Create collator
    collator = CopyTaskCollator(pad_token=train_dataset.pad_token)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        collate_fn=collator,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=collator,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=collator,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )
    
    return train_loader, val_loader, test_loader


class SequenceGenerationEvaluator:
    """Evaluator for sequence generation tasks"""
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        
    def evaluate_exact_match(self, dataloader: DataLoader) -> dict:
        """Evaluate exact sequence match accuracy"""
        self.model.eval()
        total_sequences = 0
        exact_matches = 0
        token_accuracies = []
        
        with torch.no_grad():
            for input_seq, target_seq in dataloader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                batch_size, input_len = input_seq.shape
                _, target_len = target_seq.shape
                
                # Forward pass
                embedded_input = self.model.embedding(input_seq) if hasattr(self.model, 'embedding') else input_seq
                conditioning_input = embedded_input[:, :-1, :]
                final_states, _, _, _ = self.model(conditioning_input)
                
                # Generate predictions
                selected_states = final_states[:, :target_len, :]
                logits = self.model.output_layer(selected_states) if hasattr(self.model, 'output_layer') else selected_states
                predictions = torch.argmax(logits, dim=-1)
                
                # Calculate metrics
                for i in range(batch_size):
                    pred_seq = predictions[i]
                    true_seq = target_seq[i]
                    
                    # Remove padding
                    if hasattr(dataloader.dataset, 'pad_token'):
                        pad_token = dataloader.dataset.pad_token
                        true_len = (true_seq != pad_token).sum().item()
                        pred_seq = pred_seq[:true_len]
                        true_seq = true_seq[:true_len]
                    
                    # Exact match
                    if torch.equal(pred_seq, true_seq):
                        exact_matches += 1
                    
                    # Token accuracy
                    token_acc = (pred_seq == true_seq).float().mean().item()
                    token_accuracies.append(token_acc)
                    
                    total_sequences += 1
        
        return {
            'exact_match_accuracy': exact_matches / total_sequences,
            'token_accuracy': np.mean(token_accuracies),
            'token_accuracy_std': np.std(token_accuracies),
            'total_sequences': total_sequences
        }
    
    def evaluate_by_length(self, dataloader: DataLoader) -> dict:
        """Evaluate accuracy by sequence length"""
        self.model.eval()
        length_stats = {}
        
        with torch.no_grad():
            for input_seq, target_seq in dataloader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                batch_size, input_len = input_seq.shape
                _, target_len = target_seq.shape
                
                # Forward pass
                embedded_input = self.model.embedding(input_seq) if hasattr(self.model, 'embedding') else input_seq
                conditioning_input = embedded_input[:, :-1, :]
                final_states, _, _, _ = self.model(conditioning_input)
                
                # Generate predictions
                selected_states = final_states[:, :target_len, :]
                logits = self.model.output_layer(selected_states) if hasattr(self.model, 'output_layer') else selected_states
                predictions = torch.argmax(logits, dim=-1)
                
                # Group by length
                for i in range(batch_size):
                    pred_seq = predictions[i]
                    true_seq = target_seq[i]
                    
                    # Get actual length
                    if hasattr(dataloader.dataset, 'pad_token'):
                        pad_token = dataloader.dataset.pad_token
                        true_len = (true_seq != pad_token).sum().item()
                    else:
                        true_len = len(true_seq)
                    
                    if true_len not in length_stats:
                        length_stats[true_len] = {'correct': 0, 'total': 0}
                    
                    # Check exact match
                    if torch.equal(pred_seq[:true_len], true_seq[:true_len]):
                        length_stats[true_len]['correct'] += 1
                    length_stats[true_len]['total'] += 1
        
        # Calculate accuracies
        length_accuracies = {}
        for length, stats in length_stats.items():
            length_accuracies[length] = stats['correct'] / stats['total']
            
        return length_accuracies