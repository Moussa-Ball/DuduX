#!/usr/bin/env python3
"""
DUDUX-GPT Interactive Inference
===============================

🧠 Interactive inference for DUDUX binary neural architecture
⚡ Real-time text generation with trained BSO model
🎯 Optimized for GTX 1650 4GB GPU
🚀 Professional conversational AI interface

Features:
- Interactive chat interface
- Real-time text generation
- GPU-optimized inference
- Conversation history
- Customizable generation parameters
- Model performance metrics

Authors: DUDUX Research Team
Version: 1.0.0 Professional
Created: August 5, 2025
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import time
import json
from datetime import datetime
import argparse

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model import DuduxGPT
    from tokenizer import DuduxTokenizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class DuduxInference:
    """
    Professional inference engine for DUDUX-GPT
    
    Handles model loading, tokenization, and interactive generation
    with optimizations for GTX 1650 4GB GPU.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 256,
        verbose: bool = True
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.verbose = verbose
        self.device = self._setup_device(device)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        
        # Generation parameters - optimized for speed and coherence
        self.generation_params = {
            'max_new_tokens': 64,  # Reduced for faster generation
            'temperature': 0.7,    # Lower for more deterministic responses
            'top_k': 40,          # Reduced for faster sampling
            'top_p': 0.8,         # Lower for more focused responses
            'do_sample': True,
            'repetition_penalty': 1.05  # Lower penalty for speed
        }
        
        # Performance tracking
        self.inference_stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'total_inference_time': 0.0,
            'average_tokens_per_second': 0.0
        }

    def _setup_device(self, device: str) -> torch.device:
        """Setup inference device"""
        if device == "auto":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        if self.verbose:
            print(f"🖥️ Inference Device Setup:")
            print(f"   Device: {device}")
            
            if device.type == "cuda":
                print(f"   GPU: {torch.cuda.get_device_name()}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                # Clear cache for inference
                torch.cuda.empty_cache()
        
        return device

    def load_model(self):
        """Load trained DUDUX-GPT model and tokenizer"""
        if self.verbose:
            print(f"\n🔄 Loading DUDUX-GPT Model...")
            print(f"   Model Path: {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            if 'config' in checkpoint:
                config = checkpoint['config']
                vocab_size = config.get('vocab_size', 50257)
                d_model = config.get('d_model', 384)
                num_layers = config.get('num_layers', 12)
                num_heads = config.get('num_heads', 12)
                max_seq_len = config.get('max_seq_len', 256)
                dropout = config.get('dropout', 0.1)
            else:
                # Default configuration if not found
                print("   ⚠️ No config found, using default parameters")
                vocab_size, d_model, num_layers, num_heads = 50257, 384, 12, 12
                max_seq_len, dropout = 256, 0.1
            
            # Initialize tokenizer
            self.tokenizer = DuduxTokenizer(
                encoding_name="gpt2",
                max_length=max_seq_len,
                device=self.device,
                verbose=self.verbose
            )
            
            # Initialize model
            self.model = DuduxGPT(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_multiplier=4,
                max_seq_len=max_seq_len,
                dropout=dropout
            )
            
            # Load model weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Model info
            total_params = sum(p.numel() for p in self.model.parameters())
            binary_neurons = self.model.count_binary_neurons()
            
            if self.verbose:
                print(f"   ✅ Model loaded successfully!")
                print(f"   📊 Parameters: {total_params:,}")
                print(f"   🎯 Binary Neurons: {binary_neurons:,}")
                print(f"   📚 Vocabulary: {vocab_size:,}")
                print(f"   🎮 Device: {self.device}")
                
                if 'epoch' in checkpoint:
                    print(f"   📈 Trained Epochs: {checkpoint['epoch']}")
                if 'max_test_acc1' in checkpoint:
                    print(f"   🏆 Best Accuracy: {checkpoint['max_test_acc1']:.2f}%")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None
    ) -> Tuple[str, Dict]:
        """
        Generate text using DUDUX-GPT
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of stop tokens to end generation
            
        Returns:
            Tuple of (generated_text, generation_stats)
        """
        start_time = time.time()
        
        # Initialize generation parameters for speed
        generation_params = {
            'max_new_tokens': max_new_tokens or self.generation_params['max_new_tokens'],
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'do_sample': do_sample,
            'repetition_penalty': repetition_penalty
        }
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Pre-compute repetition penalty indices for speed
        repeated_tokens = set()
        
        # Generate tokens in batch for efficiency
        generated_tokens = []
        current_length = len(input_ids)
        
        # Optimize: disable gradient computation and use optimized inference
        with torch.no_grad():
            with torch.cuda.amp.autocast() if self.device.type == "cuda" else torch.no_grad():
                for step in range(generation_params['max_new_tokens']):
                    if current_length >= self.max_length:
                        break
                    
                    # Single forward pass
                    logits = self.model(input_tensor)
                    next_token_logits = logits[0, -1, :]  # Get last token logits
                    
                    # Fast repetition penalty (only for repeated tokens)
                    if repetition_penalty != 1.0 and repeated_tokens:
                        for token_id in repeated_tokens:
                            next_token_logits[token_id] /= repetition_penalty
                    
                    # Apply temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Optimized sampling
                    if do_sample:
                        # Fast top-k
                        if top_k > 0:
                            values, indices = torch.topk(next_token_logits, top_k)
                            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                            next_token_logits.scatter_(0, indices, values)
                        
                        # Fast top-p
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            cutoff = (cumulative_probs <= top_p).sum().item()
                            if cutoff > 0:
                                cutoff_value = sorted_logits[cutoff-1]
                                next_token_logits[next_token_logits < cutoff_value] = float('-inf')
                        
                        # Sample
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy decoding
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    next_token_id = next_token.item()
                    generated_tokens.append(next_token_id)
                    repeated_tokens.add(next_token_id)
                    
                    # Efficient tensor update
                    input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                    current_length += 1
                    
                    # Early stop check (every 10 tokens for efficiency)
                    if step % 10 == 0 and stop_tokens:
                        partial_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        if any(stop_token in partial_text for stop_token in stop_tokens):
                            break
        
        # Decode generated text
        final_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate stats
        generation_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
        
        generation_stats = {
            'prompt_tokens': len(input_ids),
            'generated_tokens': len(generated_tokens),
            'total_tokens': len(input_ids) + len(generated_tokens),
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        }
        
        # Update global stats
        self.inference_stats['total_generations'] += 1
        self.inference_stats['total_tokens_generated'] += len(generated_tokens)
        self.inference_stats['total_inference_time'] += generation_time
        self.inference_stats['average_tokens_per_second'] = (
            self.inference_stats['total_tokens_generated'] / 
            self.inference_stats['total_inference_time']
        )
        
        return final_text, generation_stats

    def interactive_chat(self):
        """Start interactive chat session"""
        print(f"\n🚀 DUDUX-GPT Interactive Chat")
        print(f"{'='*60}")
        print(f"Type your message and press Enter to generate a response.")
        print(f"Commands:")
        print(f"  /help     - Show this help")
        print(f"  /params   - Show generation parameters")
        print(f"  /set      - Set generation parameters")
        print(f"  /stats    - Show inference statistics")
        print(f"  /history  - Show conversation history")
        print(f"  /save     - Save conversation")
        print(f"  /clear    - Clear conversation history")
        print(f"  /quit     - Exit chat")
        print(f"{'='*60}\n")
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/help':
                        self._show_help()
                    elif user_input == '/params':
                        self._show_params()
                    elif user_input.startswith('/set'):
                        self._set_params(user_input)
                    elif user_input == '/stats':
                        self._show_stats()
                    elif user_input == '/history':
                        self._show_history()
                    elif user_input == '/save':
                        self._save_conversation()
                    elif user_input == '/clear':
                        self._clear_history()
                    elif user_input == '/quit':
                        print("👋 Goodbye!")
                        break
                    else:
                        print("❓ Unknown command. Type /help for available commands.")
                    continue
                
                # Generate response
                print("🧠 DUDUX-GPT: ", end="", flush=True)
                
                response, stats = self.generate(
                    prompt=user_input,
                    **self.generation_params
                )
                
                print(response)
                print(f"   💨 Generated {stats['generated_tokens']} tokens in {stats['generation_time']:.2f}s ({stats['tokens_per_second']:.1f} tok/s)\n")
                
                # Add to conversation history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user': user_input,
                    'assistant': response,
                    'stats': stats
                })
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during generation: {e}")

    def _show_help(self):
        """Show help information"""
        print("\n📖 DUDUX-GPT Chat Commands:")
        print("   /help     - Show this help message")
        print("   /params   - Display current generation parameters")
        print("   /set <param> <value> - Set generation parameter")
        print("   /stats    - Show inference performance statistics")
        print("   /history  - Display conversation history")
        print("   /save     - Save conversation to file")
        print("   /clear    - Clear conversation history")
        print("   /quit     - Exit the chat session")
        print("\n📝 Generation Parameters:")
        print("   max_new_tokens    - Maximum tokens to generate (default: 128)")
        print("   temperature       - Sampling temperature (0.1-2.0, default: 0.8)")
        print("   top_k            - Top-k sampling (1-100, default: 50)")
        print("   top_p            - Nucleus sampling (0.1-1.0, default: 0.9)")
        print("   repetition_penalty - Repetition penalty (1.0-2.0, default: 1.1)")
        print()

    def _show_params(self):
        """Show current generation parameters"""
        print("\n⚙️ Current Generation Parameters:")
        for key, value in self.generation_params.items():
            print(f"   {key}: {value}")
        print()

    def _set_params(self, command: str):
        """Set generation parameters"""
        parts = command.split()
        if len(parts) != 3:
            print("❌ Usage: /set <parameter> <value>")
            return
        
        param, value = parts[1], parts[2]
        
        try:
            if param == 'max_new_tokens':
                self.generation_params[param] = int(value)
            elif param in ['temperature', 'top_p', 'repetition_penalty']:
                self.generation_params[param] = float(value)
            elif param == 'top_k':
                self.generation_params[param] = int(value)
            elif param == 'do_sample':
                self.generation_params[param] = value.lower() in ['true', '1', 'yes']
            else:
                print(f"❌ Unknown parameter: {param}")
                return
            
            print(f"✅ Set {param} = {self.generation_params[param]}")
        except ValueError:
            print(f"❌ Invalid value for {param}: {value}")

    def _show_stats(self):
        """Show inference statistics"""
        stats = self.inference_stats
        print(f"\n📊 Inference Statistics:")
        print(f"   Total Generations: {stats['total_generations']}")
        print(f"   Total Tokens Generated: {stats['total_tokens_generated']:,}")
        print(f"   Total Inference Time: {stats['total_inference_time']:.2f}s")
        print(f"   Average Speed: {stats['average_tokens_per_second']:.1f} tokens/second")
        print()

    def _show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history.")
            return
        
        print(f"\n📝 Conversation History ({len(self.conversation_history)} exchanges):")
        for i, exchange in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            print(f"\n   {i}. [{exchange['timestamp']}]")
            print(f"      👤 You: {exchange['user']}")
            print(f"      🧠 DUDUX-GPT: {exchange['assistant']}")
        print()

    def _save_conversation(self):
        """Save conversation to file"""
        if not self.conversation_history:
            print("📝 No conversation to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'model_info': {
                        'model_path': self.model_path,
                        'device': str(self.device),
                        'generation_params': self.generation_params
                    },
                    'conversation': self.conversation_history,
                    'inference_stats': self.inference_stats
                }, f, indent=2)
            
            print(f"💾 Conversation saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")

    def _clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        print("🗑️ Conversation history cleared.")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="DUDUX-GPT Interactive Inference")
    parser.add_argument("--model", type=str, default="model/dudux_bso_trained.pth",
                       help="Path to trained model file")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--max-length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    
    args = parser.parse_args()
    
    print(f"🧠 DUDUX-GPT Interactive Inference")
    print(f"Version 1.0.0 | August 5, 2025")
    print(f"{'='*60}")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Make sure you have trained a model first using scripts/train.py")
        return
    
    try:
        # Initialize inference engine
        inference = DuduxInference(
            model_path=args.model,
            device=args.device,
            max_length=args.max_length,
            verbose=True
        )
        
        # Load model
        inference.load_model()
        
        # Set custom generation parameters
        inference.generation_params.update({
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p
        })
        
        # Start interactive chat
        inference.interactive_chat()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main()
