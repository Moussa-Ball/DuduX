"""
DUDUX Neural Brain - Inference Script
====================================

🧠 Interactive conversation with trained DUDUX brain
⚡ Real-time pattern matching and response generation
🎯 Load trained model and chat interactively
🚀 GPU-accelerated inference with similarity scoring

Authors: Research Team Dudux  
Version: 1.0.0 Inference Engine
Created: August 5, 2025
"""

import os
import sys
import time
import torch
import warnings
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

# Suppress PyTorch nested tensor warnings
warnings.filterwarnings("ignore", message=".*nested tensors.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model import DuduxBrain
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


@dataclass
class InferenceConfig:
    """Inference configuration parameters"""
    model_path: str = "model/dudux.pth"

    # Response parameters
    similarity_threshold: float = 0.4
    max_response_length: int = 512
    temperature: float = 1.0

    # Device
    device: str = "auto"  # auto, cuda, cpu


class DuduxInference:
    """DUDUX Neural Brain Inference Engine"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.brain: Optional[DuduxBrain] = None
        self.device = self._get_device()
        self.model_info: Dict = {}

    def _get_device(self) -> torch.device:
        """Get appropriate device for inference"""
        if self.config.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = self.config.device
        return torch.device(device_str)

    def load_model(self) -> bool:
        """Load trained DUDUX model"""
        try:
            print(f"🧠 Loading DUDUX Brain from: {self.config.model_path}")

            if not os.path.exists(self.config.model_path):
                print(f"❌ Model file not found: {self.config.model_path}")
                return False

            # Load model data
            model_data = torch.load(
                self.config.model_path, map_location=self.device)

            # Extract configuration and stats
            if 'config' in model_data:
                saved_config = model_data['config']
                self.model_info = {
                    'training_completed': model_data.get('training_completed', 'Unknown'),
                    'final_stats': model_data.get('final_stats', {}),
                    'config': saved_config
                }

                # Initialize brain with saved config
                self.brain = DuduxBrain(
                    n_neurons=saved_config.get('n_neurons', 50000),
                    pattern_dim=saved_config.get('pattern_dim', 1024),
                    memory_size=saved_config.get('memory_size', 10000),
                    device=self.device,
                    verbose=False
                )

                # Load state dict
                self.brain.load_state_dict(model_data['model_state_dict'])
                self.brain.eval()

                # REMOVED: No more original text loading - pure neural generation!
                # if 'original_inputs' in model_data and 'original_responses' in model_data:
                #     self.brain.memory.original_inputs = model_data['original_inputs']
                #     self.brain.memory.original_responses = model_data['original_responses']
                #     print(f"✅ Loaded {len([x for x in model_data['original_inputs'] if x])} original text pairs")
                # else:
                #     print(f"⚠️  No original texts found in model - responses may be limited")
                print(f"🧠 Model loaded - using pure neural generation (no text lookup!)")

                print(f"✅ Model loaded successfully!")
                self._print_model_info()
                return True
            else:
                print(f"❌ Invalid model format: missing configuration")
                return False

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def _print_model_info(self):
        """Print loaded model information"""
        stats = self.model_info.get('final_stats', {})
        config = self.model_info.get('config', {})

        print(f"\n📊 DUDUX Model Information:")
        print(f"{'─'*50}")
        print(f"   🧠 Total Neurons: {stats.get('total_neurons', 'Unknown'):,}")
        print(
            f"   🎯 Pattern Dimension: {stats.get('pattern_dimension', 'Unknown')}")
        print(
            f"   💾 Stored Experiences: {stats.get('stored_experiences', 'Unknown')}")
        print(
            f"   📏 Memory Capacity: {stats.get('memory_capacity', 'Unknown'):,}")
        print(f"   ⚡ Device: {stats.get('device', str(self.device))}")
        print(
            f"   🎓 Training Completed: {self.model_info.get('training_completed', 'Unknown')}")

        if 'similarity_threshold' in config:
            print(
                f"   🎯 Similarity Threshold: {config['similarity_threshold']}")

        print(f"{'─'*50}\n")

    def generate_response(self, input_text: str) -> Tuple[str, float, bool]:
        """Generate response for input text"""
        if self.brain is None:
            return "❌ Model not loaded", 0.0, False

        try:
            # Process input and generate response
            response, similarity, found_match = self.brain.process_input(
                input_text, self.config.similarity_threshold
            )

            return response, similarity, found_match

        except Exception as e:
            return f"❌ Error generating response: {e}", 0.0, False

    def get_brain_stats(self) -> Dict:
        """Get current brain statistics"""
        if self.brain is None:
            return {}

        return self.brain.get_brain_stats()

    def interactive_chat(self):
        """Start interactive chat session"""
        print(f"\n🤖 DUDUX Interactive Chat")
        print(f"{'='*60}")
        print(f"💬 Type your message and press Enter")
        print(f"📝 Commands: 'quit', 'exit', 'stats', 'help'")
        print(f"🎯 Similarity threshold: {self.config.similarity_threshold}")
        print(f"{'='*60}\n")

        conversation_count = 0

        while True:
            try:
                # Get user input
                user_input = input(f"\n🟦 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(
                        f"\n👋 Goodbye! Had {conversation_count} conversations.")
                    break
                elif user_input.lower() == 'stats':
                    self._print_chat_stats()
                    continue
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue

                # Generate response
                start_time = time.time()
                response, similarity, found_match = self.generate_response(
                    user_input)
                response_time = time.time() - start_time

                # Display response with enhanced metrics
                match_symbol = "✅" if found_match else "❌"
                response_type = "🧠 Generated from neural patterns" if found_match else "🤖 Creative neural generation"
                
                print(f"\n🤖 DUDUX: {response}")
                print(f"   📊 [Similarity: {similarity:.3f} | Match: {match_symbol} | {response_type} | Time: {response_time:.2f}s]")

                conversation_count += 1

            except KeyboardInterrupt:
                print(
                    f"\n\n👋 Chat interrupted. Had {conversation_count} conversations.")
                break
            except Exception as e:
                print(f"\n❌ Error in chat: {e}")

    def _print_chat_stats(self):
        """Print current brain statistics"""
        stats = self.get_brain_stats()
        print(f"\n📊 Current Brain Statistics:")
        print(f"{'─'*40}")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                if key == 'memory_usage_percent':
                    print(f"   {key}: {value:.1f}%")
                elif isinstance(value, int) and value > 1000:
                    print(f"   {key}: {value:,}")
                else:
                    print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value}")
        print(f"{'─'*40}")

    def _print_help(self):
        """Print help information"""
        print(f"\n💡 DUDUX Chat Commands:")
        print(f"{'─'*40}")
        print(f"   📝 Type any message to chat with DUDUX")
        print(f"   📊 'stats' - Show brain statistics")
        print(f"   💡 'help' - Show this help")
        print(f"   👋 'quit'/'exit' - End chat session")
        print(f"{'─'*40}")
        print(f"\n🎯 Similarity threshold: {self.config.similarity_threshold}")
        print(f"   ✅ ≥ {self.config.similarity_threshold} = Match found")
        print(f"   ❌ < {self.config.similarity_threshold} = No match")

    def single_inference(self, text: str) -> None:
        """Perform single inference and print result"""
        print(f"\n💬 Input: {text}")

        start_time = time.time()
        response, similarity, found_match = self.generate_response(text)
        response_time = time.time() - start_time

        match_symbol = "✅" if found_match else "❌"
        response_type = "📚 ORIGINAL from dataset" if found_match else "🤖 Generated fallback"
        
        print(f"🤖 Response: {response}")
        print(f"📊 Similarity: {similarity:.3f} | Match: {match_symbol} | {response_type} | Time: {response_time:.3f}s")


def main():
    """Main inference function"""
    print(f"🧠 DUDUX Neural Brain - Inference Engine")
    print(f"Version 1.0.0 | August 5, 2025")
    print(f"{'='*60}")

    # Check device
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  Using device: cuda")
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
    else:
        print(f"🖥️  Using device: cpu")

    # Initialize inference
    config = InferenceConfig()
    inference = DuduxInference(config)

    # Load model
    if not inference.load_model():
        print(f"❌ Failed to load model. Exiting.")
        return

    # Check command line arguments
    if len(sys.argv) > 1:
        # Single inference mode
        text = " ".join(sys.argv[1:])
        inference.single_inference(text)
    else:
        # Interactive chat mode
        inference.interactive_chat()


if __name__ == "__main__":
    main()
