#!/usr/bin/env python3
"""
DUDUX-GPT Tokenizer
==================

🔤 Professional tokenization for DUDUX binary neural architecture
⚡ TikToken integration with GPU optimization
🎯 Compatible with DuduxGPT model architecture
🧠 Efficient text to token conversion for binary neurons

Authors: DUDUX Research Team
Version: 4.0.0 Professional
Created: August 5, 2025
"""

import torch
import tiktoken
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class DuduxTokenizer:
    """
    Professional tokenizer for DUDUX-GPT binary neural architecture
    Uses TikToken for robust tokenization with GPU optimization
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",  # GPT-4 encoding (100K vocab)
        max_length: int = 4096,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ):
        """
        Initialize DUDUX tokenizer

        Args:
            encoding_name: TikToken encoding name
            max_length: Maximum sequence length
            device: Compute device (auto-detected if None)
            verbose: Print initialization info
        """
        self.encoding_name = encoding_name
        self.max_length = max_length
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if verbose:
            print(f"🔤 Initializing DUDUX Tokenizer...")
            print(f"   📚 Encoding: {encoding_name}")
            print(f"   📏 Max Length: {max_length}")
            print(f"   🎮 Device: {self.device}")

        # Initialize TikToken encoder
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
            self.vocab_size = self.tokenizer.n_vocab
            if verbose:
                print(f"   ✅ TikToken loaded: {self.vocab_size:,} tokens")
        except Exception as e:
            if verbose:
                print(f"   ❌ TikToken error: {e}")
                print(f"   🔄 Falling back to GPT-2 encoding...")
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab

        # Special tokens compatible with DuduxGPT
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2  # Beginning of sequence
        self.eos_token_id = 3  # End of sequence

        if verbose:
            print(f"   🎯 Vocabulary Size: {self.vocab_size:,}")
            print(f"   🚀 Ready for DUDUX-GPT!")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Override default max length

        Returns:
            List of token IDs
        """
        if not text.strip():
            return [self.pad_token_id]

        max_len = max_length or self.max_length

        try:
            # TikToken encoding
            token_ids = self.tokenizer.encode(text)

            # Add special tokens
            if add_special_tokens:
                token_ids = [self.bos_token_id] + \
                    token_ids + [self.eos_token_id]

            # Truncate if too long
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len-1] + [self.eos_token_id]

            return token_ids

        except Exception as e:
            warnings.warn(
                f"Encoding failed for text: {text[:50]}... Error: {e}")
            return [self.unk_token_id]

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text

        Args:
            token_ids: Token IDs (list or tensor)
            skip_special_tokens: Skip special tokens

        Returns:
            Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        try:
            # Filter special tokens
            if skip_special_tokens:
                special_tokens = {self.pad_token_id, self.bos_token_id,
                                  self.eos_token_id, self.unk_token_id}
                filtered_ids = [
                    tid for tid in token_ids if tid not in special_tokens]
            else:
                filtered_ids = token_ids

            # TikToken decoding
            text = self.tokenizer.decode(filtered_ids)
            return text.strip()

        except Exception as e:
            warnings.warn(
                f"Decoding failed for tokens: {token_ids[:10]}... Error: {e}")
            return "<DECODE_ERROR>"

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        return_tensors: bool = True
    ) -> Union[torch.Tensor, List[List[int]]]:
        """
        Encode batch of texts with padding

        Args:
            texts: List of input texts
            add_special_tokens: Add special tokens
            padding: Pad sequences to same length
            return_tensors: Return as tensors

        Returns:
            Padded batch (tensor or list)
        """
        batch_ids = []

        # Encode all texts
        for text in texts:
            token_ids = self.encode(text, add_special_tokens)
            batch_ids.append(token_ids)

        if padding:
            # Pad to same length
            max_len = min(max(len(ids) for ids in batch_ids), self.max_length)

            padded_batch = []
            for token_ids in batch_ids:
                if len(token_ids) < max_len:
                    padded_ids = token_ids + \
                        [self.pad_token_id] * (max_len - len(token_ids))
                else:
                    padded_ids = token_ids[:max_len]
                padded_batch.append(padded_ids)

            batch_ids = padded_batch

        if return_tensors:
            return torch.tensor(batch_ids, dtype=torch.long, device=self.device)

        return batch_ids

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask for padded sequences

        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]

        Returns:
            Attention mask [batch_size, seq_len]
        """
        return (input_ids != self.pad_token_id).float()

    def get_vocab_info(self) -> Dict:
        """Get vocabulary information"""
        return {
            'encoding_name': self.encoding_name,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'device': str(self.device),
            'special_tokens': {
                'pad_token_id': self.pad_token_id,
                'unk_token_id': self.unk_token_id,
                'bos_token_id': self.bos_token_id,
                'eos_token_id': self.eos_token_id
            }
        }

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text tokenization

        Args:
            text: Input text

        Returns:
            Analysis dictionary with statistics
        """
        token_ids = self.encode(text, add_special_tokens=True)
        decoded = self.decode(token_ids, skip_special_tokens=False)

        # Calculate metrics
        char_count = len(text)
        token_count = len(token_ids)
        compression_ratio = char_count / token_count if token_count > 0 else 0

        # Sample token breakdown
        tokens_preview = []
        for tid in token_ids[:10]:
            try:
                token_text = self.decode([tid], skip_special_tokens=False)
                tokens_preview.append(f"ID:{tid} -> '{token_text}'")
            except:
                tokens_preview.append(f"ID:{tid} -> <ERROR>")

        return {
            'original_text': text,
            'token_ids': token_ids,
            'decoded_text': decoded,
            'char_count': char_count,
            'token_count': token_count,
            'compression_ratio': compression_ratio,
            'tokens_preview': tokens_preview,
            'efficiency': f"{compression_ratio:.2f} chars/token"
        }


def test_tokenizer():
    """Test DUDUX tokenizer functionality"""
    print("🧪 Testing DUDUX Tokenizer")
    print("=" * 50)

    # Initialize tokenizer
    tokenizer = DuduxTokenizer(encoding_name="cl100k_base")

    # Test texts for binary neural processing
    test_texts = [
        "Hello, world!",
        "The revolutionary DUDUX-GPT uses binary neurons for intelligence.",
        "Intelligence artificielle et neurosciences cognitives.",
        "🧠 Binary patterns encode meaning through 0/1 operations! ⚡",
        "",  # Empty string
        "A" * 500,  # Long text
    ]

    print("\n📝 Tokenization Analysis:")
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        analysis = tokenizer.analyze_text(text)
        print(
            f"  📊 Length: {analysis['char_count']} chars → {analysis['token_count']} tokens")
        print(f"  🗜️ Efficiency: {analysis['efficiency']}")
        print(f"  🔤 Sample tokens: {analysis['tokens_preview'][:3]}")

    # Test batch processing
    print(f"\n🔢 Batch Processing Test:")
    batch_tensor = tokenizer.encode_batch(test_texts[:3], return_tensors=True)
    attention_mask = tokenizer.create_attention_mask(batch_tensor)

    print(f"  📐 Batch shape: {batch_tensor.shape}")
    print(f"  🎯 Attention mask shape: {attention_mask.shape}")
    print(f"  📱 Device: {batch_tensor.device}")
    print(f"  🔢 Sample tokens: {batch_tensor[0][:10].tolist()}")

    # Test decoding
    print(f"\n🔄 Decoding Test:")
    original_text = "DUDUX binary neurons process information efficiently."
    encoded = tokenizer.encode(original_text)
    decoded = tokenizer.decode(encoded)

    print(f"  📝 Original: {original_text}")
    print(f"  🔢 Encoded: {encoded}")
    print(f"  🔄 Decoded: {decoded}")
    print(f"  ✅ Match: {original_text.strip() == decoded.strip()}")

    # Vocabulary info
    print(f"\n📚 Vocabulary Info:")
    vocab_info = tokenizer.get_vocab_info()
    for key, value in vocab_info.items():
        if key != 'special_tokens':
            print(f"  {key}: {value}")

    print(f"\n✅ All tokenizer tests completed successfully!")
    return tokenizer


if __name__ == "__main__":
    # Run tests
    tokenizer = test_tokenizer()

    print(f"\n🎉 DUDUX Tokenizer ready for binary neural processing!")
