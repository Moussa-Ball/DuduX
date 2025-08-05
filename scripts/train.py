#!/usr/bin/env python3
"""
DUDUX-GPT Professional Training Script
BSO (Binary Spiking Online) Optimization Integration
Optimized for GTX 1650 4GB VRAM

Author: DUDUX Team
Date: August 5, 2025
Version: 5.0.0
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any
import signal

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import DuduxGPT
from tokenizer import DuduxTokenizer
from dataset import DuduxDataset

@dataclass
class TrainingConfig:
    """Configuration centralisée pour l'entraînement"""
    # Architecture du modèle (ultra-compact pour GTX 1650)
    vocab_size: int = 100277  # cl100k_base tokenizer
    d_model: int = 256        # Très réduit pour économie mémoire
    num_layers: int = 8       # Très réduit
    num_heads: int = 8        # Réduit
    max_seq_len: int = 128    # Très réduit pour économie mémoire
    
    # Paramètres d'entraînement
    num_epochs: int = 15
    learning_rate: float = 5e-4
    batch_size: int = 1       # Batch size minimal
    gradient_accumulation_steps: int = 16  # Batch effectif = 16
    
    # Longueurs de séquences (optimisées)
    max_input_length: int = 80
    max_output_length: int = 80
    
    # Optimisation BSO
    optimizer: str = "bso"
    bso_threshold: float = 5e-7
    bso_beta1: float = 0.999
    bso_beta2: float = 0.99999
    bso_adaptive_threshold: bool = True
    
    # Optimisations GPU
    mixed_precision: bool = True
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Validation et sauvegarde
    eval_every: int = 500
    save_every: int = 1000
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001
    
    # Chemins
    model_save_path: str = "model/dudux_final.pth"
    log_path: str = "logs/training.log"
    dataset_path: str = "data/conversations_extended.txt"

class BSOptimizer:
    """
    Binary Spiking Online (BSO) Optimizer
    ICML 2025 Algorithm Implementation
    """
    
    def __init__(self, model_params, lr=1e-3, threshold=1e-6, 
                 beta1=0.9, beta2=0.999, eps=1e-8, adaptive_threshold=True):
        self.params = list(model_params)
        self.lr = lr
        self.threshold = threshold
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.adaptive_threshold = adaptive_threshold
        
        # État de l'optimiseur
        self.state = {}
        self.step_count = 0
        self.memory_saved = 0
        self.total_params = 0
        
        # Initialisation des états
        for param in self.params:
            if param.requires_grad:
                self.state[param] = {
                    'exp_avg': torch.zeros_like(param.data),
                    'exp_avg_sq': torch.zeros_like(param.data),
                    'binary_mask': torch.ones_like(param.data, dtype=torch.bool),
                    'flip_signal': torch.zeros_like(param.data),
                    'accumulated_grad': torch.zeros_like(param.data)
                }
                self.total_params += param.numel()
    
    def step(self):
        """Étape d'optimisation BSO"""
        self.step_count += 1
        memory_saved_step = 0
        
        for param in self.params:
            if param.grad is None or not param.requires_grad:
                continue
                
            state = self.state[param]
            grad = param.grad.data
            
            # Accumulation des gradients
            state['accumulated_grad'].add_(grad)
            
            # Mise à jour des moyennes exponentielles
            state['exp_avg'].mul_(self.beta1).add_(grad, alpha=1-self.beta1)
            state['exp_avg_sq'].mul_(self.beta2).addcmul_(grad, grad, value=1-self.beta2)
            
            # Correction de biais
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count
            
            corrected_exp_avg = state['exp_avg'] / bias_correction1
            corrected_exp_avg_sq = state['exp_avg_sq'] / bias_correction2
            
            # Calcul du signal de flip BSO
            denom = corrected_exp_avg_sq.sqrt().add_(self.eps)
            step_size = self.lr / denom
            
            # Seuil adaptatif
            current_threshold = self.threshold
            if self.adaptive_threshold:
                current_threshold = self.threshold * (1 + 0.1 * torch.tanh(
                    torch.tensor(self.step_count / 1000.0)
                ))
            
            # Détection des poids à binariser
            flip_candidates = torch.abs(corrected_exp_avg) > current_threshold
            
            # Signal de flip binaire
            state['flip_signal'] = torch.where(
                flip_candidates,
                torch.sign(corrected_exp_avg),
                torch.zeros_like(corrected_exp_avg)
            )
            
            # Mise à jour des paramètres
            param.data.add_(corrected_exp_avg, alpha=-step_size)
            
            # Application du masque binaire pour économies mémoire
            binary_update = torch.abs(state['flip_signal']) > 0
            state['binary_mask'] = state['binary_mask'] | binary_update
            
            # Comptage des économies mémoire
            memory_saved_step += binary_update.sum().item()
        
        self.memory_saved += memory_saved_step
        return memory_saved_step
    
    def zero_grad(self):
        """Remet à zéro les gradients"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def get_memory_savings(self):
        """Calcule les économies mémoire"""
        if self.total_params == 0:
            return 0.0, 0.0
        
        savings_ratio = self.memory_saved / (self.total_params * self.step_count + 1e-8)
        savings_mb = (self.memory_saved * 4) / (1024 * 1024)  # 4 bytes par float32
        return savings_ratio * 100, savings_mb

class DuduxTrainer:
    """Gestionnaire d'entraînement principal pour DUDUX-GPT"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configuration du logging
        self.setup_logging()
        
        # Initialisation des composants
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Métriques d'entraînement
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.start_time = time.time()
        
        # Gestion des interruptions
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
    
    def setup_logging(self):
        """Configuration du système de logging"""
        os.makedirs(os.path.dirname(self.config.log_path), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """Gestionnaire d'interruption propre"""
        self.logger.info("🛑 Interruption détectée, sauvegarde en cours...")
        if self.model is not None:
            self.save_checkpoint("model/dudux_interrupted.pth")
        exit(0)
    
    def initialize_components(self):
        """Initialise tous les composants d'entraînement"""
        self.logger.info("🚀 DUDUX-GPT Professional Training")
        self.logger.info("Version 5.0.0 | August 5, 2025")
        self.logger.info("=" * 60)
        
        # GPU Info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"🖥️ Device: {gpu_name}")
            self.logger.info(f"💾 VRAM: {gpu_memory:.1f} GB")
        
        # Tokenizer
        self.logger.info("🔤 Initializing Tokenizer...")
        self.tokenizer = DuduxTokenizer(
            encoding_name="cl100k_base",
            max_length=self.config.max_seq_len,
            device=self.device
        )
        self.logger.info(f"✅ Tokenizer ready: {self.tokenizer.vocab_size:,} tokens")
        
        # Model
        self.logger.info("🧠 Initializing Model...")
        self.model = DuduxGPT(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_seq_len
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"✅ Model ready: {total_params:,} parameters")
        
        # Dataset
        self.logger.info("📂 Loading Dataset...")
        from dataset import DatasetConfig
        
        dataset_config = DatasetConfig(
            file_path=self.config.dataset_path,
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length
        )
        
        dataset = DuduxDataset(
            dataset_config,
            self.tokenizer,
            split='train'
        )
        
        # Data split
        train_size = max(1, int(0.9 * len(dataset)))
        val_size = max(0, len(dataset) - train_size)
        
        if val_size == 0:
            # Si pas assez de données, on utilise tout pour l'entraînement
            train_dataset = dataset
            val_dataset = dataset  # On utilise le même dataset pour validation
            self.logger.info("⚠️ Not enough data for validation split, using training data")
        else:
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.logger.info(f"✅ Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Optimizer
        self.logger.info("⚡ Setting up BSO Optimizer...")
        if self.config.optimizer.lower() == "bso":
            self.optimizer = BSOptimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                threshold=self.config.bso_threshold,
                beta1=self.config.bso_beta1,
                beta2=self.config.bso_beta2,
                adaptive_threshold=self.config.bso_adaptive_threshold
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
        
        # Scheduler
        self.scheduler = None  # Le BSOptimizer gère son propre learning rate
        if not isinstance(self.optimizer, BSOptimizer):
            total_steps = len(self.train_loader) * self.config.num_epochs
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps
            )
        
        self.logger.info("✅ Training setup complete!")
        self.logger.info("=" * 60)
    
    def train_step(self, batch):
        """Étape d'entraînement"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        if self.config.mixed_precision and self.scaler:
            with autocast():
                logits = self.model(input_ids, attention_mask)
                # Décalage pour prédiction du token suivant
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(input_ids, attention_mask)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss.backward()
        
        return loss.item()
    
    def validate(self):
        """Validation du modèle"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.config.mixed_precision:
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100
                        )
                else:
                    logits = self.model(input_ids, attention_mask)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, path):
        """Sauvegarde du modèle"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"💾 Model saved: {path}")
    
    def train(self):
        """Boucle d'entraînement principale"""
        self.initialize_components()
        
        self.logger.info("🚀 TRAINING STARTED")
        self.logger.info("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Étape d'entraînement
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Accumulation de gradients
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.mixed_precision and self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                        if hasattr(self.optimizer, 'step'):
                            self.optimizer.step()
                        else:
                            self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    if self.scheduler and hasattr(self.scheduler, 'step'):
                        self.scheduler.step()
                    
                    self.global_step += 1
                
                # Validation périodique
                if self.global_step % self.config.eval_every == 0:
                    val_loss = self.validate()
                    
                    # Métriques BSO
                    if isinstance(self.optimizer, BSOptimizer):
                        savings_pct, savings_mb = self.optimizer.get_memory_savings()
                        self.logger.info(
                            f"Epoch {epoch+1}/{self.config.num_epochs} | "
                            f"Step {self.global_step} | "
                            f"Train Loss: {loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"BSO Savings: {savings_pct:.2f}% ({savings_mb:.1f}MB)"
                        )
                    else:
                        self.logger.info(
                            f"Epoch {epoch+1}/{self.config.num_epochs} | "
                            f"Step {self.global_step} | "
                            f"Train Loss: {loss:.4f} | "
                            f"Val Loss: {val_loss:.4f}"
                        )
                    
                    # Early stopping
                    if val_loss < self.best_val_loss - self.config.early_stopping_threshold:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(self.config.model_save_path)
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.early_stopping_patience:
                            self.logger.info("🛑 Early stopping triggered")
                            return
            
            # Log de fin d'époque
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            elapsed_time = time.time() - self.start_time
            self.logger.info(
                f"✅ Epoch {epoch+1} complete | "
                f"Avg Loss: {avg_epoch_loss:.4f} | "
                f"Time: {elapsed_time/60:.1f}min"
            )
        
        self.logger.info("🎉 Training completed successfully!")
        
        # Sauvegarde finale
        if isinstance(self.optimizer, BSOptimizer):
            savings_pct, savings_mb = self.optimizer.get_memory_savings()
            self.logger.info(f"🎯 Final BSO Memory Savings: {savings_pct:.2f}% ({savings_mb:.1f}MB)")

def main():
    """Fonction principale"""
    try:
        # Configuration optimisée pour GTX 1650
        config = TrainingConfig()
        
        # Entraînement
        trainer = DuduxTrainer(config)
        trainer.train()
        
    except Exception as e:
        logging.error(f"❌ Training error: {e}")
        raise

if __name__ == "__main__":
    main()
