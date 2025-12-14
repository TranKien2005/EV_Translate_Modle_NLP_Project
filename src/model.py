"""
Complete Transformer Model for Machine Translation
"""

import torch
import torch.nn as nn
import math
from src.layers import EncoderLayer, DecoderLayer


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sinusoidal functions
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder: Stack of N encoder layers
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional mask
        Returns:
            [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder: Stack of N decoder layers
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch_size, tgt_seq_len, d_model]
            encoder_output: [batch_size, src_seq_len, d_model]
            src_mask: Source mask
            tgt_mask: Target mask (look-ahead)
        Returns:
            [batch_size, tgt_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Complete Transformer Model for Sequence-to-Sequence Translation
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_pad_mask(self, seq, pad_idx=None):
        """
        Create padding mask
        Args:
            seq: [batch_size, seq_len]
        Returns:
            mask: [batch_size, 1, 1, seq_len]
        """
        if pad_idx is None:
            pad_idx = self.pad_idx
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_look_ahead_mask(self, size):
        """
        Create look-ahead mask for decoder
        Args:
            size: sequence length
        Returns:
            mask: [1, 1, size, size]
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.uint8)
        return mask == 0
    
    def forward(self, src, tgt):
        """
        Forward pass
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]
        Returns:
            output: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Create masks
        src_mask = self.create_pad_mask(src)
        tgt_mask = self.create_pad_mask(tgt)
        
        # Create look-ahead mask for decoder
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(tgt.device)
        tgt_mask = tgt_mask & look_ahead_mask.unsqueeze(0)
        
        # Embedding and positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Encoder
        encoder_output = self.encoder(src_embedded, src_mask)
        
        # Decoder
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output
    
    def encode(self, src):
        """
        Encode source sequence (for inference)
        Args:
            src: [batch_size, src_seq_len]
        Returns:
            encoder_output: [batch_size, src_seq_len, d_model]
        """
        src_mask = self.create_pad_mask(src)
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        encoder_output = self.encoder(src_embedded, src_mask)
        return encoder_output, src_mask
    
    def decode(self, tgt, encoder_output, src_mask):
        """
        Decode one step (for inference)
        Args:
            tgt: [batch_size, tgt_seq_len]
            encoder_output: [batch_size, src_seq_len, d_model]
            src_mask: Source mask
        Returns:
            output: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        tgt_mask = self.create_pad_mask(tgt)
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(tgt.device)
        tgt_mask = tgt_mask & look_ahead_mask.unsqueeze(0)
        
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        
        return output
