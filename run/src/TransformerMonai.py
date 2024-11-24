import torch
import torch.nn as nn
from generative.networks.nets import DecoderOnlyTransformer
from src.utils import *
import torch.nn.functional as F

class MonaiDecoderOnlyModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, vocab_size=64, max_len=64):
        super().__init__()
        print("Initialized monai transformer")
        self.transformer = DecoderOnlyTransformer(
            num_tokens=vocab_size,
            max_seq_len=max_len,
            attn_layers_dim=d_model,
            attn_layers_depth=num_layers,
            attn_layers_heads=nhead,
            with_cross_attention=False,
            embedding_dropout_rate=0.1
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size)

    

    def generate(self, idx, max_new_tokens):
        """
        Generate a sequence of tokens based on the initial context `idx`.

        Args:
            idx (torch.Tensor): Initial context of shape (B, T) with indices.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: Generated sequence of shape (B, T + max_new_tokens).
        """
        self.eval()  # Ensure the model is in evaluation mode

        with torch.no_grad():  # Disable gradient computation for inference
            for _ in range(max_new_tokens):
                # Crop `idx` to the last `block_size` tokens for efficient processing
                idx_cond = idx[:, -block_size:]
                
                # Forward pass to get logits
                logits = self(idx_cond)
                
                # Focus only on the last time step (prediction for the next token)
                logits = logits[:, -1, :]  # Shape: (B, vocab_size)
                
                # Convert logits to probabilities using softmax
                probs = F.softmax(logits, dim=-1)  # Shape: (B, vocab_size)
                
                # Sample the next token from the probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)
                
                # Append the sampled token to the sequence
                idx = torch.cat((idx, idx_next), dim=1)  # Shape: (B, T + 1)

        return idx

    def forward(self, src, tgt=None):
        
        logits = self.transformer(src)
        
        B, T, C = logits.shape  
        logits = logits.reshape(B * T, C)  

        logits = self.lm_head(logits)
        
        logits = logits.view(B, T, -1)

        return logits
