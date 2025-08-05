"""
Tokenizer Module for Binary Neural Network (NNN) Processing

This module provides tokenization capabilities that convert raw text into
sparse distributed representations (SDR) suitable for binary neural networks.

Author: Moussa Ball
Email: moiseball20155@gmail.com
Date: August 5, 2025
"""

import random
import tiktoken
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration class for the tokenizer."""
    encoding_name: str = "cl100k_base"  # GPT-4 tokenizer by default, can also use "gpt2"
    sdr_length: int = 1024
    sparsity: float = 0.02
    cache_size: int = 10000


class BinaryTokenizer:
    """
    A tokenizer that converts text to sparse distributed representations (SDR).

    This tokenizer uses tiktoken for initial tokenization and converts each token ID
    to a binary sparse vector suitable for processing by binary neural networks.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        """
        Initialize the tokenizer with configuration.

        Args:
            config: TokenizerConfig object with tokenizer parameters
        """
        self.config = config or TokenizerConfig()
        self.encoder = tiktoken.get_encoding(self.config.encoding_name)
        self._sdr_cache: Dict[int, List[int]] = {}

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate tokenizer configuration parameters."""
        if self.config.sdr_length <= 0:
            raise ValueError("SDR length must be positive")

        if not (0 < self.config.sparsity < 1):
            raise ValueError("Sparsity must be between 0 and 1")

        num_active_bits = int(self.config.sdr_length * self.config.sparsity)
        if num_active_bits == 0:
            raise ValueError("Sparsity too low: results in 0 active bits")

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize input text into token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        return self.encoder.encode(text)

    def id_to_sdr(self, token_id: int) -> List[int]:
        """
        Convert a token ID to sparse distributed representation (SDR).

        This function creates a deterministic binary vector where only a small
        percentage of bits are active (set to 1). The same token ID will always
        produce the same SDR pattern.

        Args:
            token_id: Integer token ID to convert

        Returns:
            Binary vector as list of integers (0s and 1s)
        """
        # Check cache first
        if token_id in self._sdr_cache:
            return self._sdr_cache[token_id].copy()

        # Generate deterministic SDR using token_id as seed
        random.seed(token_id)

        # Initialize empty SDR
        sdr = [0] * self.config.sdr_length

        # Calculate number of active bits
        num_active = int(self.config.sdr_length * self.config.sparsity)

        # Randomly select positions for active bits
        active_positions = random.sample(
            range(self.config.sdr_length), num_active)

        # Set active bits
        for pos in active_positions:
            sdr[pos] = 1

        # Cache the result if cache is not full
        if len(self._sdr_cache) < self.config.cache_size:
            self._sdr_cache[token_id] = sdr.copy()

        return sdr

    def text_to_sdr_sequence(self, text: str) -> List[List[int]]:
        """
        Convert text to sequence of SDR patterns.

        This is the main pipeline function that combines tokenization and
        SDR conversion into a single operation.

        Args:
            text: Input text string

        Returns:
            List of SDR patterns, one for each token
        """
        token_ids = self.tokenize_text(text)
        sdr_patterns = [self.id_to_sdr(token_id) for token_id in token_ids]
        return sdr_patterns

    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return self.encoder.decode(token_ids)

    def get_vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        return self.encoder.n_vocab

    def get_sdr_stats(self, sdr: List[int]) -> Dict[str, float]:
        """
        Calculate statistics for an SDR pattern.

        Args:
            sdr: Binary vector

        Returns:
            Dictionary with statistics (sparsity, active_bits, etc.)
        """
        active_bits = sum(sdr)
        total_bits = len(sdr)
        sparsity = active_bits / total_bits if total_bits > 0 else 0

        return {
            "active_bits": active_bits,
            "total_bits": total_bits,
            "sparsity": sparsity,
            "density": 1.0 - sparsity
        }

    def clear_cache(self) -> None:
        """Clear the SDR cache."""
        self._sdr_cache.clear()

    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self._sdr_cache)


def demo_tokenizer() -> None:
    """Demonstration of the tokenizer functionality."""
    print("=" * 60)
    print("Binary Tokenizer Demo")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = BinaryTokenizer()

    # Example text
    text = "Hello, how are you today?"
    print(f"Input text: '{text}'")

    # Tokenize
    token_ids = tokenizer.tokenize_text(text)
    print(f"Token IDs: {token_ids}")

    # Convert to SDR sequence
    sdr_sequence = tokenizer.text_to_sdr_sequence(text)
    print(f"Number of SDR patterns: {len(sdr_sequence)}")

    # Show stats for first token
    if sdr_sequence:
        first_sdr = sdr_sequence[0]
        stats = tokenizer.get_sdr_stats(first_sdr)
        print(f"First token SDR stats: {stats}")
        print(f"First 50 bits of first SDR: {first_sdr[:50]}")

    # Decode back
    decoded_text = tokenizer.decode_tokens(token_ids)
    print(f"Decoded text: '{decoded_text}'")

    print(f"Cache size: {tokenizer.get_cache_size()}")
    print("=" * 60)


if __name__ == "__main__":
    demo_tokenizer()
