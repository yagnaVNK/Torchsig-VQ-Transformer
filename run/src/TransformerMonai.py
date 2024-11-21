import torch
import torch.nn as nn
from generative.networks.nets import DecoderOnlyTransformer
from src.utils import *

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
        self.eval()
        # Initialize generated sequence with idx as a batch
        batch_size = idx.size(0)
        generated_sequence = idx.clone().to(device=device)  # Start with given idx of shape (4, 1)
        print("generating from MONAI transformer")
        for _ in range(1, max_new_tokens):
            with torch.no_grad():
                # Forward pass through the model
                output = self(generated_sequence)  # Output shape: (batch_size, seq_len, vocab_size)
                
                # Get the last token's logits for each sequence in the batch
                next_tokens = torch.argmax(output[:, -1, :], dim=-1)  # Shape: (batch_size,)
                
                # Append the predicted tokens to the sequence
                next_tokens = next_tokens.unsqueeze(1)  # Shape: (batch_size, 1)
                generated_sequence = torch.cat([generated_sequence, next_tokens], dim=1)  # Shape: (batch_size, seq_len + 1)

                # Stop if the generated sequence reaches max length
                if generated_sequence.size(1) >= max_new_tokens:
                    break
        
        return generated_sequence  # Shape: (4, 64)

    def forward(self, src, tgt=None):
        
        logits = self.transformer(src)
        
        B, T, C = logits.shape  
        logits = logits.reshape(B * T, C)  

        logits = self.lm_head(logits)
        
        logits = logits.view(B, T, -1)

        return logits
