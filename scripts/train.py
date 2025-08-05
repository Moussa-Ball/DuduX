#!/usr/bin/env python3
"""
DUDUX-GPT Training Script
========================

🧠 Professional training for DUDUX binary neural architecture
⚡ GPU-optimized training with real-time metrics
🎯 Compatible with DuduxGPT model architecture
🚀 Advanced training features and monitoring

Authors: DUDUX Research Team
Version: 4.0.0 Professional
Created: August 5, 2025
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
import math

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model import DuduxGPT
    from tokenizer import DuduxTokenizer
    from dataset import DatasetManager, DatasetConfig, create_sample_dataset
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


@dataclass
class TrainingConfig:
    """Professional training configuration for DUDUX-GPT"""

    # Dataset parameters
    dataset_path: str = "data/conversations.txt"
    max_input_length: int = 512
    max_output_length: int = 512
    validation_split: float = 0.1

    # Model parameters
    vocab_size: int = 100277
    d_model: int = 1024
    num_layers: int = 32
    num_heads: int = 32
    max_seq_len: int = 4096
    dropout: float = 0.1

    # Training parameters
    num_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000

    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, linear, constant

    # Logging and saving
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500
    save_path: str = "model/dudux_trained.pth"
    log_path: str = "logs/training.log"

    # Device and performance
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: bool = True  # Use automatic mixed precision
    compile_model: bool = False  # Use torch.compile (requires PyTorch 2.0+)

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


class TrainingLogger:
    """Professional training logger with comprehensive metrics"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.start_time = 0.0
        self.epoch_start_time = 0.0
        self.step_times = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Create directories
        os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    def start_training(self, total_steps: int, model_params: int):
        """Initialize training logging"""
        self.start_time = time.time()
        self.total_steps = total_steps

        print(f"\n🚀 DUDUX-GPT TRAINING STARTED")
        print(f"{'='*60}")
        print(f"📊 Model Parameters: {model_params:,}")
        print(f"🔄 Total Steps: {total_steps:,}")
        print(f"📚 Epochs: {self.config.num_epochs}")
        print(f"🎯 Batch Size: {self.config.batch_size}")
        print(f"⚡ Learning Rate: {self.config.learning_rate}")
        print(f"🎮 Device: {self.config.device}")
        print(f"💾 Mixed Precision: {self.config.mixed_precision}")
        print(f"{'='*60}\n")

    def log_step(
        self,
        epoch: int,
        step: int,
        global_step: int,
        train_loss: float,
        learning_rate: float,
        throughput: float,
        gpu_memory: float = 0.0
    ):
        """Log training step with comprehensive metrics"""
        self.train_losses.append(train_loss)
        self.learning_rates.append(learning_rate)

        # Calculate progress
        progress = global_step / self.total_steps
        elapsed = time.time() - self.start_time
        eta = (elapsed / progress - elapsed) if progress > 0 else 0

        # GPU memory info
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_total = torch.cuda.get_device_properties(
                0).total_memory / 1024**3  # GB
            gpu_info = f"GPU: {gpu_used:.1f}GB/{gpu_total:.1f}GB"

        # Professional progress bar like Transformers
        if step % self.config.log_every == 0:
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            print(f"Epoch {epoch:2d}: {progress*100:5.1f}%|{bar}| "
                  f"Step {global_step:6d}/{self.total_steps} | "
                  f"Loss: {train_loss:.4f} | "
                  f"LR: {learning_rate:.2e} | "
                  f"Throughput: {throughput:.1f} tok/s | "
                  f"{gpu_info} | "
                  f"ETA: {timedelta(seconds=int(eta))}")

    def log_validation(self, epoch: int, val_loss: float, val_perplexity: float):
        """Log validation metrics"""
        self.val_losses.append(val_loss)

        print(f"\n  ✅ Validation Results:")
        print(f"     Loss: {val_loss:.4f}")
        print(f"     Perplexity: {val_perplexity:.2f}")
        print()

    def end_epoch(self, epoch: int, avg_train_loss: float):
        """End epoch logging"""
        epoch_time = time.time() - self.epoch_start_time

        print(f"\n  📊 Epoch {epoch} Summary:")
        print(f"     Duration: {timedelta(seconds=int(epoch_time))}")
        print(f"     Avg Train Loss: {avg_train_loss:.4f}")
        print(
            f"     Train Perplexity: {math.exp(min(avg_train_loss, 10)):.2f}")

    def save_logs(self, final_stats: Dict):
        """Save training logs to file"""
        log_data = {
            'config': self.config.__dict__,
            'training_time': time.time() - self.start_time,
            'final_stats': final_stats,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(self.config.log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"📝 Training logs saved: {self.config.log_path}")
        except Exception as e:
            print(f"⚠️ Failed to save logs: {e}")


class DuduxTrainer:
    """Professional trainer for DUDUX-GPT binary neural architecture"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.logger = TrainingLogger(config)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _setup_device(self) -> torch.device:
        """Setup training device with optimizations"""
        if self.config.device == "auto":
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)

        print(f"🖥️ Device Setup:")
        print(f"   Device: {device}")

        if device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(
                f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

            # Clear cache and optimize
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        return device

    def setup_tokenizer(self):
        """Initialize DUDUX tokenizer"""
        print(f"\n🔤 Initializing Tokenizer...")

        self.tokenizer = DuduxTokenizer(
            encoding_name="cl100k_base",
            max_length=self.config.max_seq_len,
            device=self.device,
            verbose=True
        )

    def setup_model(self):
        """Initialize DUDUX-GPT model"""
        print(f"\n🧠 Initializing DUDUX-GPT Model...")

        self.model = DuduxGPT(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            ff_multiplier=4,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout
        )

        # Move to device
        self.model = self.model.to(self.device)

        # Compile model for speed (PyTorch 2.0+)
        if self.config.compile_model:
            try:
                self.model = torch.compile(self.model)
                print(f"   ⚡ Model compiled for optimization")
            except Exception as e:
                print(f"   ⚠️ Model compilation failed: {e}")

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        print(f"\n⚡ Setting up Optimizer...")

        # Create optimizer
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )

        # Setup learning rate scheduler
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.scheduler.lower() == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )

        # Setup mixed precision scaler
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"   🎯 Mixed precision enabled")

        print(f"   Optimizer: {self.config.optimizer}")
        print(f"   Scheduler: {self.config.scheduler}")
        print(f"   Learning Rate: {self.config.learning_rate}")

    def setup_dataset(self):
        """Setup dataset and data loaders"""
        print(f"\n📂 Setting up Dataset...")

        # Check if dataset exists, create sample if not
        if not os.path.exists(self.config.dataset_path):
            print(f"   Dataset not found, creating sample dataset...")
            self.config.dataset_path = create_sample_dataset()

        # Initialize dataset manager
        dataset_manager = DatasetManager(self.tokenizer, verbose=True)

        # Create dataset config
        dataset_config = DatasetConfig(
            file_path=self.config.dataset_path,
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            shuffle=True
        )

        # Load dataset
        train_dataset, val_dataset = dataset_manager.load_dataset(
            'main', dataset_config)

        # Create data loaders
        self.train_loader, self.val_loader = dataset_manager.create_data_loaders(
            'main',
            train_batch_size=self.config.batch_size,
            val_batch_size=self.config.batch_size,
            num_workers=4
        )

        # Show sample conversations
        print(f"\n📋 Sample Training Data:")
        samples = dataset_manager.sample_conversations('main', n=3)
        for i, (inp, out) in enumerate(samples, 1):
            inp_preview = inp[:60] + "..." if len(inp) > 60 else inp
            out_preview = out[:60] + "..." if len(out) > 60 else out
            print(f"   {i}. Q: {inp_preview}")
            print(f"      A: {out_preview}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single training step"""
        self.model.train()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Forward pass with mixed precision
        if self.config.mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
        else:
            logits = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def validation_step(self) -> Tuple[float, float]:
        """Execute validation step"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_ids, attention_mask)
                        loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                            logits.view(-1, logits.size(-1)), labels.view(-1)
                        )
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )

                # Accumulate metrics
                valid_tokens = (
                    labels != self.tokenizer.pad_token_id).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

        avg_loss = total_loss / \
            total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(min(avg_loss, 10))  # Clamp to prevent overflow

        return avg_loss, perplexity

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.__dict__,
            'val_loss': val_loss,
            'tokenizer_config': self.tokenizer.get_vocab_info(),
            'timestamp': datetime.now().isoformat()
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        if is_best:
            save_path = self.config.save_path
        else:
            save_path = self.config.save_path.replace(
                '.pth', f'_checkpoint_epoch_{epoch}.pth')

        torch.save(checkpoint, save_path)
        print(f"💾 {'Best model' if is_best else 'Checkpoint'} saved: {save_path}")

    def train(self):
        """Main training loop"""
        try:
            # Setup all components
            self.setup_tokenizer()
            self.setup_model()
            self.setup_optimizer()
            self.setup_dataset()

            # Calculate training steps
            steps_per_epoch = len(self.train_loader)
            total_steps = steps_per_epoch * self.config.num_epochs

            # Start training
            model_params = sum(p.numel() for p in self.model.parameters())
            self.logger.start_training(total_steps, model_params)

            # Training loop
            for epoch in range(1, self.config.num_epochs + 1):
                self.logger.epoch_start_time = time.time()
                epoch_loss = 0.0
                self.optimizer.zero_grad()

                for step, batch in enumerate(self.train_loader, 1):
                    step_start_time = time.time()

                    # Training step
                    loss = self.train_step(batch)
                    epoch_loss += loss

                    # Gradient accumulation
                    if step % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if self.config.mixed_precision and self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm)
                            self.optimizer.step()

                        self.optimizer.zero_grad()
                        self.global_step += 1

                    # Calculate throughput
                    step_time = time.time() - step_start_time
                    tokens_per_second = (
                        batch['input_ids'].numel() / step_time) if step_time > 0 else 0

                    # Log progress
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.log_step(
                        epoch, step, self.global_step, loss, current_lr, tokens_per_second
                    )

                    # Validation
                    if self.global_step % self.config.eval_every == 0:
                        val_loss, val_perplexity = self.validation_step()
                        self.logger.log_validation(
                            epoch, val_loss, val_perplexity)

                        # Check for improvement
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            self.save_checkpoint(epoch, val_loss, is_best=True)
                        else:
                            self.patience_counter += 1

                        # Early stopping
                        if self.patience_counter >= self.config.early_stopping_patience:
                            print(
                                f"\n🛑 Early stopping triggered after {self.patience_counter} evaluations without improvement")
                            break

                    # Save checkpoint
                    if self.global_step % self.config.save_every == 0:
                        val_loss, _ = self.validation_step()
                        self.save_checkpoint(epoch, val_loss, is_best=False)

                # End epoch
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.logger.end_epoch(epoch, avg_epoch_loss)

                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Early stopping check
                if self.patience_counter >= self.config.early_stopping_patience:
                    break

            # Final validation and save
            print(f"\n🏁 Training completed!")
            final_val_loss, final_perplexity = self.validation_step()

            final_stats = {
                'total_steps': self.global_step,
                'final_train_loss': avg_epoch_loss,
                'final_val_loss': final_val_loss,
                'final_perplexity': final_perplexity,
                'model_parameters': model_params,
                'best_val_loss': self.best_val_loss
            }

            self.logger.save_logs(final_stats)

            print(f"📊 Final Results:")
            print(f"   Best Validation Loss: {self.best_val_loss:.4f}")
            print(f"   Final Validation Loss: {final_val_loss:.4f}")
            print(f"   Final Perplexity: {final_perplexity:.2f}")

        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted by user")
            if hasattr(self, 'model') and self.model is not None:
                self.save_checkpoint(epoch, float('inf'), is_best=False)
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            raise


def main():
    """Main training function"""
    print(f"🧠 DUDUX-GPT Professional Training")
    print(f"Version 4.0.0 | August 5, 2025")
    print(f"{'='*60}")

    # Optimized training configuration for GTX 1650 4GB
    config = TrainingConfig(
        # Model parameters (optimized for GTX 1650)
        vocab_size=100277,
        d_model=1024,
        num_layers=32,
        num_heads=32,
        max_seq_len=1024,  # Reduced for memory efficiency

        # Training parameters
        num_epochs=3,
        learning_rate=2e-4,
        batch_size=2,  # Small batch for 4GB GPU
        gradient_accumulation_steps=8,  # Effective batch size = 16

        # Optimization
        mixed_precision=True,
        warmup_steps=500,

        # Paths
        dataset_path="data/conversations.txt",
        save_path="model/dudux_trained.pth",
        log_path="logs/training.log"
    )

    # Initialize and start training
    trainer = DuduxTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
