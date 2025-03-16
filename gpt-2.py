import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass



@dataclass
class GPTConfig:

    block_size: int = 12 # T -> context length to be processed 
    vocab_size: int = 50304 # 50257 but nearest power of 2 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: float = True



