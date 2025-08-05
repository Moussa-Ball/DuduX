#!/usr/bin/env python3
"""
DUDUX-GPT Dataset Manager
========================

🗃️ Professional dataset handling for DUDUX binary neural architecture
📊 Support for conversation formats with GPU optimization
🔄 Efficient data loading and preprocessing
🎯 Compatible with DuduxGPT model training

Authors: DUDUX Research Team
Version: 4.0.0 Professional
Created: August 5, 2025
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union, Iterator
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@dataclass
class DatasetConfig:
    """Configuration for DUDUX dataset processing"""
    file_path: str
    max_input_length: int = 512
    max_output_length: int = 512
    min_length: int = 5
    shuffle: bool = True
    seed: int = 42
    validation_split: float = 0.1
    test_split: float = 0.0  # No test split for training
    encoding: str = 'utf-8'
    batch_size: int = 8
    num_workers: int = 4


class DuduxDataset(Dataset):
    """
    Professional dataset for DUDUX-GPT training
    Handles conversation pairs with efficient loading and preprocessing
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer,
        split: str = 'train',
        verbose: bool = True
    ):
        """
        Initialize DUDUX dataset

        Args:
            config: Dataset configuration
            tokenizer: DUDUX tokenizer instance
            split: Data split ('train', 'val', 'test')
            verbose: Print loading info
        """
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.verbose = verbose

        # Data storage
        self.conversations = []
        self.train_data = []
        self.val_data = []
        self.test_data = []

        # Load and process data
        self.load_data()
        self.split_data()

        # Set active data based on split
        if split == 'train':
            self.data = self.train_data
        elif split == 'val':
            self.data = self.val_data
        elif split == 'test':
            self.data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")

        if verbose:
            print(f"📊 DUDUX Dataset ({split}):")
            print(f"   📁 File: {config.file_path}")
            print(
                f"   📏 Max Length: {config.max_input_length}/{config.max_output_length}")
            print(f"   🔢 Examples: {len(self.data):,}")
            print(f"   🎮 Device: {tokenizer.device}")

    def load_data(self) -> None:
        """Load conversation data from file"""
        if self.verbose:
            print(f"📂 Loading conversations from: {self.config.file_path}")

        if not os.path.exists(self.config.file_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.config.file_path}")

        # Detect file format and load accordingly
        file_path = Path(self.config.file_path)
        extension = file_path.suffix.lower()

        if extension == '.json':
            self._load_json()
        elif extension == '.txt':
            self._load_conversation_format()
        else:
            # Default to conversation format
            self._load_conversation_format()

        if self.verbose:
            print(f"   ✅ Loaded {len(self.conversations):,} conversations")

    def _load_conversation_format(self) -> None:
        """Load INPUT:/OUTPUT: format conversations"""
        with open(self.config.file_path, 'r', encoding=self.config.encoding) as f:
            content = f.read()

        lines = content.split('\n')
        current_input = None

        for line in lines:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            if line.startswith("INPUT:"):
                current_input = line[6:].strip()
            elif line.startswith("OUTPUT:") and current_input is not None:
                current_output = line[7:].strip()

                # Clean and validate
                input_clean = self.clean_text(current_input)
                output_clean = self.clean_text(current_output)

                if self.validate_conversation(input_clean, output_clean):
                    self.conversations.append((input_clean, output_clean))

                current_input = None

    def _load_json(self) -> None:
        """Load JSON format conversations"""
        with open(self.config.file_path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if self._parse_json_item(item):
                    pass  # Item already added in _parse_json_item
        elif isinstance(data, dict):
            if 'conversations' in data:
                for item in data['conversations']:
                    self._parse_json_item(item)

    def _parse_json_item(self, item: Dict) -> bool:
        """Parse individual JSON conversation item"""
        # Try different field combinations
        input_fields = ['input', 'question', 'prompt', 'user', 'human']
        output_fields = ['output', 'answer', 'response', 'assistant', 'ai']

        input_text = None
        output_text = None

        for field in input_fields:
            if field in item:
                input_text = str(item[field])
                break

        for field in output_fields:
            if field in item:
                output_text = str(item[field])
                break

        if input_text and output_text:
            input_clean = self.clean_text(input_text)
            output_clean = self.clean_text(output_text)

            if self.validate_conversation(input_clean, output_clean):
                self.conversations.append((input_clean, output_clean))
                return True

        return False

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove control characters
        text = re.sub(
            r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)

        return text

    def validate_conversation(self, input_text: str, output_text: str) -> bool:
        """Validate conversation pair"""
        if len(input_text.strip()) < self.config.min_length:
            return False
        if len(output_text.strip()) < self.config.min_length:
            return False
        if len(input_text) > self.config.max_input_length:
            return False
        if len(output_text) > self.config.max_output_length:
            return False
        return True

    def split_data(self) -> None:
        """Split conversations into train/validation/test sets"""
        if self.config.shuffle:
            random.seed(self.config.seed)
            random.shuffle(self.conversations)

        total_size = len(self.conversations)
        val_size = int(total_size * self.config.validation_split)
        test_size = int(total_size * self.config.test_split)
        train_size = total_size - val_size - test_size

        self.train_data = self.conversations[:train_size]
        self.val_data = self.conversations[train_size:train_size + val_size]
        self.test_data = self.conversations[train_size + val_size:]

        if self.verbose:
            print(f"📊 Data split:")
            print(f"   🏋️ Training: {len(self.train_data):,} examples")
            print(f"   ✅ Validation: {len(self.val_data):,} examples")
            if test_size > 0:
                print(f"   🧪 Test: {len(self.test_data):,} examples")

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item for training

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized input/output and attention masks
        """
        input_text, output_text = self.data[idx]

        # Tokenize input and output
        input_ids = self.tokenizer.encode(
            input_text,
            add_special_tokens=True,
            max_length=self.config.max_input_length
        )
        output_ids = self.tokenizer.encode(
            output_text,
            add_special_tokens=True,
            max_length=self.config.max_output_length
        )

        # Create input sequence (input + output for language modeling)
        # This is for autoregressive training
        sequence_ids = input_ids + output_ids[1:]  # Remove BOS from output

        # Pad sequence to max length
        max_seq_len = self.config.max_input_length + self.config.max_output_length
        if len(sequence_ids) < max_seq_len:
            sequence_ids.extend([self.tokenizer.pad_token_id]
                                * (max_seq_len - len(sequence_ids)))
        else:
            sequence_ids = sequence_ids[:max_seq_len]

        # Convert to tensors on the same device as tokenizer
        device = self.tokenizer.device
        input_ids_tensor = torch.tensor(
            sequence_ids[:-1], dtype=torch.long, device=device)  # Input
        target_ids_tensor = torch.tensor(
            sequence_ids[1:], dtype=torch.long, device=device)   # Targets (shifted)
        attention_mask = (input_ids_tensor !=
                          self.tokenizer.pad_token_id).float()

        return {
            'input_ids': input_ids_tensor,
            'labels': target_ids_tensor,
            'attention_mask': attention_mask,
            'input_text': input_text,
            'output_text': output_text
        }

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if not self.conversations:
            return {}

        input_lengths = [len(inp) for inp, _ in self.conversations]
        output_lengths = [len(out) for _, out in self.conversations]

        return {
            'total_conversations': len(self.conversations),
            'train_examples': len(self.train_data),
            'val_examples': len(self.val_data),
            'test_examples': len(self.test_data),
            'avg_input_length': sum(input_lengths) / len(input_lengths),
            'avg_output_length': sum(output_lengths) / len(output_lengths),
            'max_input_length': max(input_lengths),
            'max_output_length': max(output_lengths),
            'min_input_length': min(input_lengths),
            'min_output_length': min(output_lengths)
        }


class DatasetManager:
    """
    Professional dataset manager for DUDUX-GPT training
    Handles multiple datasets and creates efficient data loaders
    """

    def __init__(self, tokenizer, verbose: bool = True):
        """
        Initialize dataset manager

        Args:
            tokenizer: DUDUX tokenizer instance
            verbose: Print initialization info
        """
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.datasets = {}
        self.data_loaders = {}

        if verbose:
            print(f"🗃️ DUDUX Dataset Manager initialized")
            print(f"   🔤 Tokenizer: {tokenizer.encoding_name}")
            print(f"   📚 Vocab Size: {tokenizer.vocab_size:,}")

    def load_dataset(
        self,
        name: str,
        config: DatasetConfig
    ) -> Tuple[DuduxDataset, DuduxDataset]:
        """
        Load and split dataset

        Args:
            name: Dataset name
            config: Dataset configuration

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.verbose:
            print(f"\n📂 Loading dataset: {name}")

        # Create train and validation datasets
        train_dataset = DuduxDataset(
            config=config,
            tokenizer=self.tokenizer,
            split='train',
            verbose=self.verbose
        )

        val_dataset = DuduxDataset(
            config=config,
            tokenizer=self.tokenizer,
            split='val',
            verbose=False  # Avoid duplicate messages
        )

        # Store datasets
        self.datasets[name] = {
            'train': train_dataset,
            'val': val_dataset,
            'config': config
        }

        return train_dataset, val_dataset

    def create_data_loaders(
        self,
        dataset_name: str,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training

        Args:
            dataset_name: Name of loaded dataset
            train_batch_size: Training batch size (uses config if None)
            val_batch_size: Validation batch size (uses config if None)
            num_workers: Number of worker processes

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        dataset_info = self.datasets[dataset_name]
        config = dataset_info['config']

        # Use provided batch sizes or config defaults
        train_bs = train_batch_size or config.batch_size
        val_bs = val_batch_size or config.batch_size
        workers = num_workers or config.num_workers

        # Create data loaders
        train_loader = DataLoader(
            dataset_info['train'],
            batch_size=train_bs,
            shuffle=True,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Drop incomplete batches for stable training
        )

        val_loader = DataLoader(
            dataset_info['val'],
            batch_size=val_bs,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

        # Store loaders
        self.data_loaders[dataset_name] = {
            'train': train_loader,
            'val': val_loader
        }

        if self.verbose:
            print(f"🔄 Data loaders created:")
            print(f"   🏋️ Train: {len(train_loader)} batches of {train_bs}")
            print(f"   ✅ Val: {len(val_loader)} batches of {val_bs}")

        return train_loader, val_loader

    def get_dataset_stats(self, dataset_name: str) -> Dict:
        """Get statistics for a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        return self.datasets[dataset_name]['train'].get_stats()

    def sample_conversations(
        self,
        dataset_name: str,
        n: int = 5,
        split: str = 'train'
    ) -> List[Tuple[str, str]]:
        """
        Get sample conversations from dataset

        Args:
            dataset_name: Name of dataset
            n: Number of samples
            split: Data split to sample from

        Returns:
            List of (input, output) conversation pairs
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        dataset = self.datasets[dataset_name][split]
        if n >= len(dataset.data):
            return dataset.data

        return random.sample(dataset.data, n)


def create_sample_dataset():
    """Create a sample dataset for testing"""
    sample_conversations = [
        ("Hello, how are you?", "I'm doing well, thank you! How can I help you today?"),
        ("What is DUDUX?", "DUDUX is a revolutionary binary neural architecture that uses 0/1 neurons for AI."),
        ("How do binary neurons work?",
         "Binary neurons fire in patterns of 0s and 1s, creating emergent intelligence through population coding."),
        ("Can you explain transformers?",
         "Transformers use attention mechanisms to process sequences and understand relationships between tokens."),
        ("What makes DUDUX special?",
         "DUDUX combines binary neurons with transformer architecture for efficient AI on consumer hardware."),
        ("Tell me about GPU optimization",
         "DUDUX is optimized for GPU acceleration with binary operations that are much faster than float32."),
        ("Goodbye!", "Goodbye! Thanks for learning about DUDUX binary neural networks!"),
    ]

    # Save as conversation format
    os.makedirs('data', exist_ok=True)
    with open('data/sample_conversations.txt', 'w', encoding='utf-8') as f:
        for inp, out in sample_conversations:
            f.write(f"INPUT: {inp}\n")
            f.write(f"OUTPUT: {out}\n\n")

    print("📄 Sample dataset created: data/sample_conversations.txt")
    return 'data/sample_conversations.txt'


def test_dataset():
    """Test DUDUX dataset functionality"""
    print("🧪 Testing DUDUX Dataset")
    print("=" * 50)

    # Import tokenizer
    from tokenizer import DuduxTokenizer

    # Create sample dataset
    dataset_path = create_sample_dataset()

    # Initialize tokenizer
    tokenizer = DuduxTokenizer(verbose=False)

    # Initialize dataset manager
    manager = DatasetManager(tokenizer, verbose=True)

    # Create dataset config
    config = DatasetConfig(
        file_path=dataset_path,
        max_input_length=256,
        max_output_length=256,
        batch_size=4,
        validation_split=0.3
    )

    # Load dataset
    train_dataset, val_dataset = manager.load_dataset('test', config)

    # Create data loaders
    train_loader, val_loader = manager.create_data_loaders('test')

    # Show statistics
    stats = manager.get_dataset_stats('test')
    print(f"\n📊 Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # Show sample conversations
    print(f"\n📋 Sample Conversations:")
    samples = manager.sample_conversations('test', n=3)
    for i, (inp, out) in enumerate(samples, 1):
        print(f"   {i}. Q: {inp}")
        print(f"      A: {out}")

    # Test data loader
    print(f"\n🔄 Testing Data Loader:")
    batch = next(iter(train_loader))
    print(f"   📐 Input shape: {batch['input_ids'].shape}")
    print(f"   🎯 Labels shape: {batch['labels'].shape}")
    print(f"   👁️ Attention shape: {batch['attention_mask'].shape}")
    print(f"   📱 Device: {batch['input_ids'].device}")

    print(f"\n✅ All dataset tests completed successfully!")
    return manager


if __name__ == "__main__":
    # Run tests
    manager = test_dataset()

    print(f"\n🎉 DUDUX Dataset ready for binary neural training!")
