/**
 * Python code templates for model architectures
 */

export class PythonTemplates {
  /**
   * Get GPT model architecture template
   */
  static getGPTArchitecture(): string {
    return `"""
GPT-style transformer model architecture
Configurable decoder-only transformer for language modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTConfig:
    """Configuration for GPT model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 512,
        layers: int = 6,
        heads: int = 6,
        dim: int = 384,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.layers = layers
        self.heads = heads
        self.dim = dim
        self.dropout = dropout
        self.head_dim = dim // heads
        
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_length, config.max_length))
            .view(1, 1, config.max_length, config.max_length)
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.dim, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 4 * config.dim)
        self.fc2 = nn.Linear(4 * config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.ff = FeedForward(config)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """GPT-style transformer language model"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        
        # Validate and clamp sequence length to prevent position embedding index errors
        if T > self.config.max_length:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Input sequence length {T} exceeds max_length {self.config.max_length}. Truncating to {self.config.max_length}.")
            input_ids = input_ids[:, :self.config.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.config.max_length]
            if labels is not None:
                labels = labels[:, :self.config.max_length]
            T = self.config.max_length
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(token_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
        
        return {'logits': logits, 'loss': loss}
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max_length
            input_ids_cond = input_ids if input_ids.size(1) <= self.config.max_length else input_ids[:, -self.config.max_length:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self(input_ids_cond)
                logits = outputs['logits']
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Validate token is in vocabulary range
            if next_token.item() >= self.config.vocab_size:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Generated token {next_token.item()} exceeds vocab size "
                    f"{self.config.vocab_size}. Clamping to valid range."
                )
                next_token = torch.clamp(next_token, 0, self.config.vocab_size - 1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_gpt_model(config_dict: dict) -> GPTModel:
    """Create GPT model from config dictionary"""
    config = GPTConfig(
        vocab_size=config_dict.get('vocab_size', 32000),
        max_length=config_dict.get('max_length', 512),
        layers=config_dict.get('layers', 6),
        heads=config_dict.get('heads', 6),
        dim=config_dict.get('dim', 384),
        dropout=config_dict.get('dropout', 0.1),
    )
    model = GPTModel(config)
    print(f"Created GPT model with {model.count_parameters():,} parameters")
    return model


if __name__ == '__main__':
    # Test model creation
    config = GPTConfig(
        vocab_size=32000,
        max_length=512,
        layers=6,
        heads=6,
        dim=384,
        dropout=0.1,
    )
    model = GPTModel(config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids, labels=labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
`;
  }

  /**
   * Get nano model template (500K parameters)
   */
  static getNanoModel(): string {
    return `"""
Nano GPT model (500K parameters)
Perfect for learning and quick testing
"""

from .gpt import create_gpt_model


def create_nano_model(model_config=None):
    """
    Create nano GPT model with 500K parameters
    
    Architecture:
    - 3 layers
    - 4 attention heads
    - 128 hidden dimension
    - 5K vocabulary (or actual vocab size from tokenizer)
    - 256 max sequence length
    
    Hardware Requirements:
    - Any CPU
    - 2GB RAM minimum
    - Training time: 1-2 minutes
    
    Note:
    - vocab_size should match your trained tokenizer's vocabulary size
    - The default value (5000) is overridden by load_model_from_config()
      to use the actual tokenizer vocabulary size
    - Mismatched vocab sizes cause poor generation quality
    """
    config = {
        'vocab_size': 5000,  # Will be overridden by actual tokenizer vocab
        'max_length': 256,
        'layers': 3,
        'heads': 4,
        'dim': 128,
        'dropout': 0.1,
    }
    
    # Override with provided config (e.g., actual vocab size from tokenizer)
    if model_config:
        config.update(model_config)
    
    return create_gpt_model(config)


if __name__ == '__main__':
    model = create_nano_model()
    print(f"Nano model created with {model.count_parameters():,} parameters")
`;
  }

  /**
   * Get tiny model template (5M parameters)
   */
  static getTinyModel(): string {
    return `"""
Tiny GPT model (5M parameters)
Optimized for prototyping and small projects
"""

from .gpt import create_gpt_model


def create_tiny_model(model_config=None):
    """
    Create tiny GPT model with 5M parameters
    
    Architecture:
    - 4 layers
    - 4 attention heads
    - 256 hidden dimension
    - 10K vocabulary (or actual vocab size from tokenizer)
    - 512 max sequence length
    
    Hardware Requirements:
    - CPU or basic GPU
    - 4GB RAM minimum
    - Training time: 5-15 minutes
    
    Note:
    - vocab_size should match your trained tokenizer's vocabulary size
    - The default value (10000) is overridden by load_model_from_config()
      to use the actual tokenizer vocabulary size
    - Mismatched vocab sizes cause poor generation quality
    """
    config = {
        'vocab_size': 10000,  # Will be overridden by actual tokenizer vocab
        'max_length': 512,
        'layers': 4,
        'heads': 4,
        'dim': 256,
        'dropout': 0.2,
    }
    
    # Override with provided config (e.g., actual vocab size from tokenizer)
    if model_config:
        config.update(model_config)
    
    return create_gpt_model(config)


if __name__ == '__main__':
    model = create_tiny_model()
    print(f"Tiny model created with {model.count_parameters():,} parameters")
`;
  }

  /**
   * Get small model template (100M parameters)
   */
  static getSmallModel(): string {
    return `"""
Small GPT model (100M parameters)
Optimized for single GPU training with good performance
"""

from .gpt import create_gpt_model


def create_small_model(model_config=None):
    """
    Create small GPT model with 100M parameters
    
    Architecture:
    - 12 layers
    - 12 attention heads
    - 768 hidden dimension
    - 32K vocabulary (or actual vocab size from tokenizer)
    - 1024 max sequence length
    
    Hardware Requirements:
    - NVIDIA RTX 3060 (12GB) or better
    - 16GB RAM minimum
    - Training time: 2-6 hours
    
    Note:
    - vocab_size should match your trained tokenizer's vocabulary size
    - The default value (32000) is overridden by load_model_from_config()
      to use the actual tokenizer vocabulary size
    - Mismatched vocab sizes cause poor generation quality
    """
    config = {
        'vocab_size': 32000,  # Will be overridden by actual tokenizer vocab
        'max_length': 1024,
        'layers': 12,
        'heads': 12,
        'dim': 768,
        'dropout': 0.1,
    }
    
    # Override with provided config (e.g., actual vocab size from tokenizer)
    if model_config:
        config.update(model_config)
    
    return create_gpt_model(config)


if __name__ == '__main__':
    model = create_small_model()
    print(f"Small model created with {model.count_parameters():,} parameters")
`;
  }

  /**
   * Get base model template (1B parameters)
   */
  static getBaseModel(): string {
    return `"""
Base GPT model (1B parameters)
Optimized for multi-GPU training with high quality
"""

from .gpt import create_gpt_model


def create_base_model(model_config=None):
    """
    Create base GPT model with 1B parameters
    
    Architecture:
    - 24 layers
    - 16 attention heads
    - 1536 hidden dimension
    - 50K vocabulary (or actual vocab size from tokenizer)
    - 2048 max sequence length
    
    Hardware Requirements:
    - NVIDIA A100 (40GB) or 2x RTX 4090
    - 64GB RAM minimum
    - Training time: 1-3 days
    
    Note:
    - vocab_size should match your trained tokenizer's vocabulary size
    - The default value (50000) is overridden by load_model_from_config()
      to use the actual tokenizer vocabulary size
    - Mismatched vocab sizes cause poor generation quality
    """
    config = {
        'vocab_size': 50000,  # Will be overridden by actual tokenizer vocab
        'max_length': 2048,
        'layers': 24,
        'heads': 16,
        'dim': 1536,
        'dropout': 0.1,
    }
    
    # Override with provided config (e.g., actual vocab size from tokenizer)
    if model_config:
        config.update(model_config)
    
    return create_gpt_model(config)


if __name__ == '__main__':
    model = create_base_model()
    print(f"Base model created with {model.count_parameters():,} parameters")
`;
  }

  /**
   * Get model config loader
   */
  static getModelConfig(): string {
    return `"""
Model configuration management
Loads model config from llm.config.js with validation and hardware checks
"""

import json
import subprocess
import platform
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigLoader:
    """
    Configuration loader for LLM training
    Loads and validates llm.config.js file
    """
    
    def __init__(self, config_path: str = 'llm.config.js'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load JavaScript config file using Node.js"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\\n"
                f"Make sure you're running from the project root directory."
            )
        
        # Use Node.js to parse the config file
        js_code = f"""
        const config = require('./{self.config_path}');
        console.log(JSON.stringify(config));
        """
        
        try:
            result = subprocess.run(
                ['node', '-e', js_code],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.config_path.parent
            )
            config = json.loads(result.stdout)
            return config
        except FileNotFoundError:
            raise RuntimeError(
                "Node.js not found. Please install Node.js to load config files.\\n"
                "Visit: https://nodejs.org/"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to load config file:\\n{e.stderr}\\n"
                f"Check your llm.config.js for syntax errors."
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse config as JSON: {e}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate model config
        self._validate_model_config()
        
        # Validate training config
        self._validate_training_config()
        
        # Validate data config
        self._validate_data_config()
        
        # Validate tokenizer config
        self._validate_tokenizer_config()
        
        # Check hardware compatibility
        self._check_hardware_compatibility()
    
    def _validate_model_config(self):
        """Validate model configuration"""
        model = self.config.get('model', {})
        
        if not model:
            raise ConfigValidationError("Missing 'model' section in config")
        
        # Check required fields
        required = ['type', 'vocab_size', 'max_length', 'layers', 'heads', 'dim']
        for field in required:
            if field not in model:
                raise ConfigValidationError(f"Missing required field: model.{field}")
        
        # Validate types
        if model['type'] not in ['gpt', 'bert', 't5']:
            raise ConfigValidationError(
                f"Invalid model type: {model['type']}. "
                f"Must be one of: gpt, bert, t5"
            )
        
        # Validate dimensions
        if model['dim'] % model['heads'] != 0:
            raise ConfigValidationError(
                f"model.dim ({model['dim']}) must be divisible by "
                f"model.heads ({model['heads']})"
            )
        
        # Validate positive values
        for field in ['vocab_size', 'max_length', 'layers', 'heads', 'dim']:
            if model[field] <= 0:
                raise ConfigValidationError(f"model.{field} must be positive")
        
        # Validate dropout
        dropout = model.get('dropout', 0.1)
        if not 0 <= dropout < 1:
            raise ConfigValidationError(
                f"model.dropout must be between 0 and 1, got {dropout}"
            )
    
    def _validate_training_config(self):
        """Validate training configuration"""
        training = self.config.get('training', {})
        
        if not training:
            raise ConfigValidationError("Missing 'training' section in config")
        
        # Validate positive values
        positive_fields = [
            'batch_size', 'learning_rate', 'max_steps',
            'eval_interval', 'save_interval', 'gradient_clip'
        ]
        for field in positive_fields:
            if field in training and training[field] <= 0:
                raise ConfigValidationError(f"training.{field} must be positive")
        
        # Validate optimizer
        optimizer = training.get('optimizer', 'adamw')
        if optimizer not in ['adamw', 'adam', 'sgd']:
            raise ConfigValidationError(
                f"Invalid optimizer: {optimizer}. "
                f"Must be one of: adamw, adam, sgd"
            )
        
        # Validate warmup steps
        warmup = training.get('warmup_steps', 0)
        if warmup < 0:
            raise ConfigValidationError("training.warmup_steps must be non-negative")
    
    def _validate_data_config(self):
        """Validate data configuration"""
        data = self.config.get('data', {})
        
        if not data:
            raise ConfigValidationError("Missing 'data' section in config")
        
        # Validate max_length and stride
        max_length = data.get('max_length', 512)
        stride = data.get('stride', 256)
        
        if max_length <= 0:
            raise ConfigValidationError("data.max_length must be positive")
        
        if stride <= 0:
            raise ConfigValidationError("data.stride must be positive")
        
        if stride > max_length:
            raise ConfigValidationError(
                f"data.stride ({stride}) cannot be greater than "
                f"data.max_length ({max_length})"
            )
        
        # Validate val_split
        val_split = data.get('val_split', 0.1)
        if not 0 <= val_split < 1:
            raise ConfigValidationError(
                f"data.val_split must be between 0 and 1, got {val_split}"
            )
    
    def _validate_tokenizer_config(self):
        """Validate tokenizer configuration"""
        tokenizer = self.config.get('tokenizer', {})
        
        if not tokenizer:
            raise ConfigValidationError("Missing 'tokenizer' section in config")
        
        # Validate tokenizer type
        tok_type = tokenizer.get('type', 'bpe')
        if tok_type not in ['bpe', 'wordpiece', 'unigram']:
            raise ConfigValidationError(
                f"Invalid tokenizer type: {tok_type}. "
                f"Must be one of: bpe, wordpiece, unigram"
            )
        
        # Validate vocab_size
        vocab_size = tokenizer.get('vocab_size', 32000)
        if vocab_size <= 0:
            raise ConfigValidationError("tokenizer.vocab_size must be positive")
    
    def _check_hardware_compatibility(self):
        """Check if hardware is compatible with config"""
        import torch
        
        model = self.config.get('model', {})
        training = self.config.get('training', {})
        
        # Estimate memory requirements (rough approximation)
        params = self._estimate_parameters()
        batch_size = training.get('batch_size', 32)
        max_length = model.get('max_length', 512)
        
        # Rough memory estimate in GB
        # params * 4 bytes (fp32) + activations + gradients + optimizer states
        memory_gb = (params * 4 * 4) / (1024 ** 3)  # 4x for gradients, optimizer
        memory_gb += (batch_size * max_length * model.get('dim', 512) * 4) / (1024 ** 3)
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            if memory_gb > 8:
                print(
                    f"⚠️  Warning: No GPU detected. Model requires ~{memory_gb:.1f}GB memory.\\n"
                    f"   Training on CPU will be very slow. Consider using a smaller model."
                )
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if memory_gb > gpu_memory * 0.8:  # Leave 20% headroom
                print(
                    f"⚠️  Warning: Model may not fit in GPU memory.\\n"
                    f"   Estimated: {memory_gb:.1f}GB, Available: {gpu_memory:.1f}GB\\n"
                    f"   Consider: reducing batch_size, enabling mixed_precision, "
                    f"or using gradient_accumulation"
                )
    
    def _estimate_parameters(self) -> int:
        """Estimate number of model parameters"""
        model = self.config.get('model', {})
        
        vocab_size = model.get('vocab_size', 32000)
        max_length = model.get('max_length', 512)
        layers = model.get('layers', 6)
        dim = model.get('dim', 384)
        
        # Rough parameter count
        # Embeddings: vocab_size * dim + max_length * dim
        # Each layer: 4 * dim^2 (attention) + 8 * dim^2 (FFN)
        # Output: dim * vocab_size (tied with input embedding)
        
        embeddings = vocab_size * dim + max_length * dim
        per_layer = 12 * dim * dim
        total = embeddings + layers * per_layer
        
        return total
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_tokenizer_config(self) -> Dict[str, Any]:
        """Get tokenizer configuration"""
        return self.config.get('tokenizer', {})
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpoint configuration"""
        return self.config.get('checkpoints', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def get_plugins(self) -> list:
        """Get enabled plugins"""
        return self.config.get('plugins', [])


def get_tokenizer_vocab_size(tokenizer_path: Path) -> Optional[int]:
    """
    Read vocabulary size from tokenizer.json
    
    Args:
        tokenizer_path: Path to tokenizer.json file
        
    Returns:
        Vocabulary size if tokenizer exists and is valid, None otherwise
    """
    if not tokenizer_path.exists():
        return None
    
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        # Extract vocab from tokenizer data
        if 'model' not in tokenizer_data:
            raise ValueError("Invalid tokenizer format: missing 'model' key")
        
        if 'vocab' not in tokenizer_data['model']:
            raise ValueError("Invalid tokenizer format: missing 'vocab' key in model")
        
        vocab_size = len(tokenizer_data['model']['vocab'])
        return vocab_size
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupted tokenizer file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading tokenizer: {e}")


def load_model_from_config(config_path: str = 'llm.config.js'):
    """Load model based on config file"""
    from .architectures import nano, tiny, small, base, gpt
    import json
    from pathlib import Path
    
    config = ConfigLoader(config_path)
    model_config = config.get_model_config()
    
    # Auto-detect vocab size from tokenizer if available
    tokenizer_path = Path('tokenizer/tokenizer.json')
    config_vocab_size = model_config.get('vocab_size', 32000)
    
    try:
        actual_vocab_size = get_tokenizer_vocab_size(tokenizer_path)
        
        if actual_vocab_size is not None:
            if actual_vocab_size != config_vocab_size:
                print(f"\\n⚠️  Vocab size mismatch detected!")
                print(f"   Config: {config_vocab_size:,} | Tokenizer: {actual_vocab_size:,}")
                print(f"   Using actual tokenizer vocab size: {actual_vocab_size:,}")
                model_config['vocab_size'] = actual_vocab_size
            else:
                print(f"✓ Vocab size: {actual_vocab_size:,}")
        else:
            print(f"⚠️  Tokenizer not found, using config vocab size: {config_vocab_size:,}")
            
    except ValueError as e:
        print(f"⚠️  Error reading tokenizer: {e}")
        print(f"   Using config vocab size: {config_vocab_size:,}")
    
    # Get model type and size
    model_type = model_config.get('type', 'gpt')
    size = model_config.get('size', 'small')

    # Route by architecture type first
    if model_type == 'bert':
        from .architectures.bert import create_bert_model
        return create_bert_model(model_config)
    elif model_type == 't5':
        from .architectures.t5 import create_t5_model
        return create_t5_model(model_config)

    # GPT: route by size
    if size == 'nano':
        return nano.create_nano_model(model_config)
    elif size == 'tiny':
        return tiny.create_tiny_model(model_config)
    elif size == 'small':
        return small.create_small_model(model_config)
    elif size == 'base':
        return base.create_base_model(model_config)
    elif size == 'custom':
        return gpt.create_gpt_model(model_config)
    else:
        raise ValueError(f"Unknown model size: {size}")


if __name__ == '__main__':
    # Test config loading
    try:
        config = ConfigLoader()
        print("✓ Config loaded successfully")
        print(f"  Model: {config.get('model.type')} ({config.get('model.size')})")
        print(f"  Parameters: ~{config._estimate_parameters() / 1_000_000:.0f}M")
        print(f"  Batch size: {config.get('training.batch_size')}")
        print(f"  Max steps: {config.get('training.max_steps')}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
`;
  }

  /**
   * Get BERT model architecture template
   */
  static getBERTArchitecture(): string {
    return `"""
BERT-style transformer model architecture
Configurable encoder-only (bidirectional) transformer for masked language modeling
and downstream NLP tasks (classification, NER, QA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BERTConfig:
    """Configuration for BERT model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 512,
        layers: int = 6,
        heads: int = 6,
        dim: int = 384,
        dropout: float = 0.1,
        type_vocab_size: int = 2,       # segment A / segment B
        pad_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.layers = layers
        self.heads = heads
        self.dim = dim
        self.dropout = dropout
        self.type_vocab_size = type_vocab_size
        self.pad_token_id = pad_token_id
        self.head_dim = dim // heads

        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"


class BERTEmbeddings(nn.Module):
    """Token + position + token-type embeddings with LayerNorm"""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.token_embedding    = nn.Embedding(config.vocab_size,      config.dim, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_length,      config.dim)
        self.token_type_embedding = nn.Embedding(config.type_vocab_size, config.dim)
        self.norm    = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.token_type_embedding(token_type_ids)
        )
        return self.dropout(self.norm(x))


class BERTAttention(nn.Module):
    """Bidirectional (unmasked) multi-head self-attention"""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.heads    = config.heads
        self.head_dim = config.head_dim
        self.dim      = config.dim

        self.qkv     = nn.Linear(config.dim, 3 * config.dim)
        self.proj    = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.dim, dim=2)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # attention_mask: (B, T) with 1=keep, 0=pad  →  (B, 1, 1, T)
        if attention_mask is not None:
            scores = scores + (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9

        attn = self.dropout(F.softmax(scores, dim=-1))
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class BERTLayer(nn.Module):
    """Single BERT encoder layer: attention → FFN, both with pre-norm residuals"""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.dim, eps=1e-12)
        self.attn = BERTAttention(config)
        self.ln2  = nn.LayerNorm(config.dim, eps=1e-12)
        self.fc1  = nn.Linear(config.dim, 4 * config.dim)
        self.fc2  = nn.Linear(4 * config.dim, config.dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        residual = x
        x = self.fc2(self.drop(F.gelu(self.fc1(self.ln2(x)))))
        return residual + self.drop(x)


class BERTModel(nn.Module):
    """
    BERT encoder stack.
    Returns per-token hidden states and the pooled [CLS] representation.
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config     = config
        self.embeddings = BERTEmbeddings(config)
        self.layers     = nn.ModuleList([BERTLayer(config) for _ in range(config.layers)])
        self.pooler_fc  = nn.Linear(config.dim, config.dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if input_ids.size(1) > self.config.max_length:
            import logging
            logging.getLogger(__name__).warning(
                f"Truncating sequence from {input_ids.size(1)} to {self.config.max_length}"
            )
            input_ids = input_ids[:, :self.config.max_length]
            if attention_mask  is not None: attention_mask  = attention_mask[:, :self.config.max_length]
            if token_type_ids  is not None: token_type_ids  = token_type_ids[:, :self.config.max_length]

        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Pooled output: tanh of the [CLS] (position 0) hidden state
        pooled = torch.tanh(self.pooler_fc(x[:, 0]))
        return {'last_hidden_state': x, 'pooler_output': pooled}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class BERTForMaskedLM(nn.Module):
    """BERT with a masked-language-modelling head for pre-training"""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.bert    = BERTModel(config)
        self.mlm_fc  = nn.Linear(config.dim, config.dim)
        self.mlm_ln  = nn.LayerNorm(config.dim, eps=1e-12)
        self.mlm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying
        self.mlm_head.weight = self.bert.embeddings.token_embedding.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        x       = outputs['last_hidden_state']
        logits  = self.mlm_head(self.mlm_ln(F.gelu(self.mlm_fc(x))))

        loss = None
        if labels is not None:
            # labels == -100 at non-masked positions → ignored by cross_entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return {'logits': logits, 'loss': loss, 'pooler_output': outputs['pooler_output']}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def generate(self, *args, **kwargs):
        """BERT is an encoder-only model and does not support autoregressive generation."""
        raise NotImplementedError(
            "BERT is an encoder-only (bidirectional) model and does not support autoregressive generation.\\n"
            "Use model(input_ids, labels=masked_labels) for masked language modelling instead.\\n"
            "For text generation, use a GPT or T5 model."
        )


def create_bert_model(config_dict: dict) -> BERTForMaskedLM:
    """Create a BERT-for-MLM model from a config dictionary"""
    config = BERTConfig(
        vocab_size      = config_dict.get('vocab_size', 32000),
        max_length      = config_dict.get('max_length', 512),
        layers          = config_dict.get('layers', 6),
        heads           = config_dict.get('heads', 6),
        dim             = config_dict.get('dim', 384),
        dropout         = config_dict.get('dropout', 0.1),
        type_vocab_size = config_dict.get('type_vocab_size', 2),
    )
    model = BERTForMaskedLM(config)
    print(f"Created BERT model with {model.count_parameters():,} parameters")
    return model


if __name__ == '__main__':
    config = BERTConfig(vocab_size=32000, max_length=128, layers=4, heads=4, dim=256)
    model  = BERTForMaskedLM(config)
    print(f"Parameters: {model.count_parameters():,}")

    B, T   = 2, 64
    ids    = torch.randint(0, config.vocab_size, (B, T))
    mask   = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()
    labels[:, ::2] = -100   # only predict every other token

    out = model(ids, attention_mask=mask, labels=labels)
    print(f"Logits: {out['logits'].shape}  Loss: {out['loss'].item():.4f}")
`;
  }

  /**
   * Get T5 model architecture template
   */
  static getT5Architecture(): string {
    return `"""
T5-style encoder-decoder transformer architecture
Text-to-Text Transfer Transformer — frames every NLP task as seq2seq.
Uses relative position biases, RMS LayerNorm, and gated GEGLU feed-forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class T5Config:
    """Configuration for T5 model"""
    def __init__(
        self,
        vocab_size: int  = 32000,
        max_length: int  = 512,
        layers: int      = 6,       # applies to both encoder and decoder
        heads: int       = 6,
        dim: int         = 384,
        ff_dim: int      = 0,       # 0 → auto (4 × dim)
        dropout: float   = 0.1,
        num_buckets: int = 32,      # relative position bias buckets
        max_distance: int = 128,    # max relative distance for bias
        pad_token_id: int = 0,
        eos_token_id: int = 1,
    ):
        self.vocab_size   = vocab_size
        self.max_length   = max_length
        self.layers       = layers
        self.heads        = heads
        self.dim          = dim
        self.ff_dim       = ff_dim if ff_dim > 0 else 4 * dim
        self.dropout      = dropout
        self.head_dim     = dim // heads
        self.num_buckets  = num_buckets
        self.max_distance = max_distance
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"


class T5RMSNorm(nn.Module):
    """Root-mean-square layer norm (no bias, no mean subtraction — as in the T5 paper)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class T5RelativePositionBias(nn.Module):
    """Learned relative position bias — shared across all layers in encoder/decoder"""

    def __init__(self, config: T5Config, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets   = config.num_buckets
        self.max_distance  = config.max_distance
        self.heads         = config.heads
        self.bias_emb      = nn.Embedding(config.num_buckets, config.heads)

    @staticmethod
    def _bucket(relative_position, bidirectional, num_buckets, max_distance):
        n    = -relative_position
        half = num_buckets // 2
        if bidirectional:
            ret = (n < 0).long() * half
            n   = n.abs()
        else:
            ret = torch.zeros_like(n)
            n   = n.clamp(min=0)

        max_exact = half // 2
        is_small  = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float().clamp(min=1) / max_exact)
            / math.log(max_distance / max_exact)
            * (half - max_exact)
        ).long().clamp(max=half - 1)
        return ret + torch.where(is_small, n, val_if_large)

    def forward(self, query_len, key_len, device):
        q_pos = torch.arange(query_len, device=device).unsqueeze(1)
        k_pos = torch.arange(key_len,   device=device).unsqueeze(0)
        rel   = k_pos - q_pos                                          # (Q, K)
        buckets = self._bucket(rel, self.bidirectional, self.num_buckets, self.max_distance)
        bias  = self.bias_emb(buckets)                                 # (Q, K, H)
        return bias.permute(2, 0, 1).unsqueeze(0)                      # (1, H, Q, K)


class T5Attention(nn.Module):
    """Multi-head attention with optional cross-attention and relative position bias"""

    def __init__(self, config: T5Config, has_relative_bias: bool = False, bidirectional: bool = True):
        super().__init__()
        self.heads    = config.heads
        self.head_dim = config.head_dim
        self.dim      = config.dim

        self.q = nn.Linear(config.dim, config.dim, bias=False)
        self.k = nn.Linear(config.dim, config.dim, bias=False)
        self.v = nn.Linear(config.dim, config.dim, bias=False)
        self.o = nn.Linear(config.dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.pos_bias = T5RelativePositionBias(config, bidirectional) if has_relative_bias else None

    def forward(self, x, key_value=None, attention_mask=None):
        B, T, _ = x.shape
        kv_src  = key_value if key_value is not None else x
        S       = kv_src.size(1)

        def split(t): return t.view(B, -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(kv_src)), split(self.v(kv_src))
        scores  = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # (B, H, T, S)

        if self.pos_bias is not None:
            scores = scores + self.pos_bias(T, S, x.device)

        if attention_mask is not None:
            scores = scores + (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9

        attn = self.dropout(F.softmax(scores, dim=-1))
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.dropout(self.o(out))


class T5FFN(nn.Module):
    """Gated GEGLU feed-forward (T5v1.1 / Flan-T5 style)"""

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_gate = nn.Linear(config.dim, config.ff_dim, bias=False)
        self.wi_proj = nn.Linear(config.dim, config.ff_dim, bias=False)
        self.wo      = nn.Linear(config.ff_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.wo(F.gelu(self.wi_gate(x)) * self.wi_proj(x)))


class T5EncoderLayer(nn.Module):
    def __init__(self, config: T5Config, has_relative_bias: bool = False):
        super().__init__()
        self.ln_attn = T5RMSNorm(config.dim)
        self.attn    = T5Attention(config, has_relative_bias=has_relative_bias, bidirectional=True)
        self.ln_ffn  = T5RMSNorm(config.dim)
        self.ffn     = T5FFN(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_attn(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln_ffn(x))
        return x


class T5DecoderLayer(nn.Module):
    def __init__(self, config: T5Config, has_relative_bias: bool = False):
        super().__init__()
        self.ln_self = T5RMSNorm(config.dim)
        self.self_attn = T5Attention(config, has_relative_bias=has_relative_bias, bidirectional=False)

        self.ln_cross = T5RMSNorm(config.dim)
        self.cross_attn = T5Attention(config, has_relative_bias=False, bidirectional=True)

        self.ln_ffn = T5RMSNorm(config.dim)
        self.ffn    = T5FFN(config)

        # Pre-register causal mask as a buffer (avoids re-allocation every forward pass)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.max_length, config.max_length)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x, encoder_hidden, decoder_mask=None, encoder_mask=None):
        # Causal self-attention — slice pre-registered buffer instead of allocating
        T = x.size(1)
        causal = self.causal_mask[:, :, :T, :T]
        if decoder_mask is not None:
            causal = causal * decoder_mask.unsqueeze(1).unsqueeze(2)
        x = x + self.self_attn(self.ln_self(x), attention_mask=causal)
        # Cross-attention over encoder output
        x = x + self.cross_attn(self.ln_cross(x), key_value=encoder_hidden, attention_mask=encoder_mask)
        x = x + self.ffn(self.ln_ffn(x))
        return x


class T5Encoder(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.layers = nn.ModuleList([
            T5EncoderLayer(config, has_relative_bias=(i == 0))
            for i in range(config.layers)
        ])
        self.final_norm = T5RMSNorm(config.dim)

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.final_norm(x)


class T5Decoder(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.layers = nn.ModuleList([
            T5DecoderLayer(config, has_relative_bias=(i == 0))
            for i in range(config.layers)
        ])
        self.final_norm = T5RMSNorm(config.dim)

    def forward(self, x, encoder_hidden, decoder_mask=None, encoder_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_hidden, decoder_mask, encoder_mask)
        return self.final_norm(x)


class T5Model(nn.Module):
    """
    T5 encoder-decoder model.
    Suitable for seq2seq tasks: summarisation, translation, QA, text generation.
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.config    = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.encoder   = T5Encoder(config)
        self.decoder   = T5Decoder(config)
        self.lm_head   = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying: shared embedding for encoder, decoder input, and lm_head
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _truncate(self, ids, mask, name):
        if ids.size(1) > self.config.max_length:
            import logging
            logging.getLogger(__name__).warning(f"Truncating {name} from {ids.size(1)} to {self.config.max_length}")
            ids  = ids[:, :self.config.max_length]
            mask = mask[:, :self.config.max_length] if mask is not None else None
        return ids, mask

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        input_ids,         attention_mask         = self._truncate(input_ids,         attention_mask,         'encoder')
        decoder_input_ids, decoder_attention_mask = self._truncate(decoder_input_ids, decoder_attention_mask, 'decoder')

        enc_hidden = self.encoder(self.embedding(input_ids), attention_mask)
        dec_hidden = self.decoder(self.embedding(decoder_input_ids), enc_hidden, decoder_attention_mask, attention_mask)
        logits     = self.lm_head(dec_hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return {'logits': logits, 'loss': loss, 'encoder_hidden_state': enc_hidden}

    def generate(self, input_ids, attention_mask=None, max_new_tokens=64, temperature=1.0, top_k=None):
        """Greedy / top-k autoregressive decoding"""
        self.eval()
        B = input_ids.size(0)
        enc_hidden = self.encoder(self.embedding(input_ids), attention_mask)
        # Start with pad token as the first decoder input
        dec_ids = torch.full((B, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                dec_hidden = self.decoder(self.embedding(dec_ids), enc_hidden, encoder_mask=attention_mask)
                logits     = self.lm_head(dec_hidden[:, -1, :]) / temperature
                if top_k is not None:
                    v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                dec_ids    = torch.cat([dec_ids, next_token], dim=1)
                if (next_token == self.config.eos_token_id).all():
                    break

        return dec_ids[:, 1:]   # strip the initial pad token

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_t5_model(config_dict: dict) -> T5Model:
    """Create T5 model from a config dictionary"""
    config = T5Config(
        vocab_size   = config_dict.get('vocab_size', 32000),
        max_length   = config_dict.get('max_length', 512),
        layers       = config_dict.get('layers', 6),
        heads        = config_dict.get('heads', 6),
        dim          = config_dict.get('dim', 384),
        dropout      = config_dict.get('dropout', 0.1),
    )
    model = T5Model(config)
    print(f"Created T5 model with {model.count_parameters():,} parameters")
    return model


if __name__ == '__main__':
    config = T5Config(vocab_size=32000, max_length=128, layers=4, heads=4, dim=256)
    model  = T5Model(config)
    print(f"Parameters: {model.count_parameters():,}")

    B, T_enc, T_dec = 2, 32, 16
    enc_ids = torch.randint(1, config.vocab_size, (B, T_enc))
    dec_ids = torch.randint(1, config.vocab_size, (B, T_dec))
    labels  = dec_ids.clone(); labels[:, ::2] = -100

    out = model(enc_ids, dec_ids, labels=labels)
    print(f"Logits: {out['logits'].shape}  Loss: {out['loss'].item():.4f}")
`;
  }

  /**
   * Get architectures __init__.py
   */
  static getArchitecturesInit(): string {
    return `"""
Model architectures package
"""

from importlib.util import find_spec

from .gpt import GPTModel, GPTConfig, create_gpt_model
from .nano import create_nano_model
from .tiny import create_tiny_model
from .small import create_small_model
from .base import create_base_model

__all__ = [
    'GPTModel',
    'GPTConfig',
    'create_gpt_model',
    'create_nano_model',
    'create_tiny_model',
    'create_small_model',
    'create_base_model',
]

if find_spec(f"{__name__}.bert") is not None:
    from .bert import BERTModel, BERTForMaskedLM, BERTConfig, create_bert_model

    __all__.extend([
        'BERTModel',
        'BERTForMaskedLM',
        'BERTConfig',
        'create_bert_model',
    ])

if find_spec(f"{__name__}.t5") is not None:
    from .t5 import T5Model, T5Config, create_t5_model

    __all__.extend([
        'T5Model',
        'T5Config',
        'create_t5_model',
    ])
`;
  }

  /**
   * Get models __init__.py
   */
  static getModelsInit(): string {
    return `"""
Models package
"""

from .config import ConfigLoader, ConfigValidationError, load_model_from_config
from .architectures import *  # Re-export the architectures available in this scaffold.
from .architectures import __all__ as _architecture_exports

__all__ = [
    'ConfigLoader',
    'ConfigValidationError',
    'load_model_from_config',
    *_architecture_exports,
]
`;
  }
}
