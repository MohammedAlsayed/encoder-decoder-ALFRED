# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import Tuple

class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, params, device):
        super(Encoder, self).__init__()
        self.device = device
        self.hidden_size = params.hidden_size
        
        if params.glove:
            print("using glove embeddings")
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(params.pre_embeddings).float(), freeze=True)
        else:
            print("using regular embeddings")
            self.embedding = nn.Embedding(params.n_words, params.embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(params.embedding_dim, params.hidden_size, dropout=params.dropout, batch_first=True) 

    def forward(self, input, hidden, length):
        embedded = self.embedding(input)
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length, enforce_sorted=False, batch_first=True)
        output, hidden = self.lstm(embedded, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
    
    def initHidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return (h0, c0)

class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.embedding_action = nn.Embedding(params.n_actions, params.embedding_dim, padding_idx=0)
        self.embedding_target = nn.Embedding(params.n_targets, params.embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(params.embedding_dim, params.hidden_size, batch_first=True)
        self.fc_action = nn.Linear(params.hidden_size, params.n_actions)
        self.fc_target = nn.Linear(params.hidden_size, params.n_targets)
        

    def forward(self, action_input, target_input, hidden):
        # concat the embedding of action and target
        action_output = self.embedding_action(action_input)
        target_output = self.embedding_target(target_input)
        output = action_output + target_output
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        action = self.fc_action(output)
        target = self.fc_target(output)

        return (action, target), hidden

class AttentionDecoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, params):
        super(AttentionDecoder, self).__init__()
        self.n_actions = params.n_actions
        self.n_targets = params.n_targets
        self.output_size = params.output_size
        self.hidden_size = params.hidden_size
        self.dropout_p = params.dropout
        self.local_attention = params.local_attention

        self.embedding_action = nn.Embedding(self.n_actions, self.hidden_size, padding_idx=0)
        self.embedding_target = nn.Embedding(self.n_targets, self.hidden_size, padding_idx=0)
    
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)

        self.dropout = nn.Dropout(self.dropout_p)
        self.a = nn.Linear(self.hidden_size * 2, 1)    
        self.fc_action = nn.Linear(self.hidden_size, self.n_actions)
        self.fc_target = nn.Linear(self.hidden_size, self.n_targets)

        

    def forward(self, idx, action_input, target_input, hidden_d, hidden_e):
        # concat the embedding of action and target
        action_output = self.embedding_action(action_input)
        target_output = self.embedding_target(target_input)
        output = action_output + target_output
        output = self.dropout(output)
        output = F.relu(output)
        
        output, hidden = self.lstm(output, hidden_d)
        
        batch_size = action_output.shape[0]
        hidden_d = hidden[0].view(batch_size,1,-1) # switch batch dim to the begning
        
        # local attention
        if self.local_attention:
            hidden_e = hidden_e[:, idx-2:idx+2, :]

        hidden_d = hidden_d.expand(-1, hidden_e.shape[1], -1) # expand decoder dim to the same as all encoder hidden states
                                                              # in other words, replicate the decoding embedding across input_size or local attention size 

        x = torch.cat((hidden_e, hidden_d), dim=2) # concat last decode hidden with all encode hiddens, hidden dimesion doubles

        a = self.a(x) # FC layer to a scaler

        attn_weights = F.softmax(a, dim=1) # softmax across sequence (sum of sequence probability = 1)
        
        attn = torch.mul(hidden_e, attn_weights) # multiply every encoder hidden output by it's atten_weight 

        attn_out = torch.sum(attn, dim=1) # sum all of the weigthed attn

        action = self.fc_action(attn_out)
        target = self.fc_target(attn_out)

        return (action, target), hidden




class TransformerModel(nn.Module):

    def __init__(self, params, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(params.n_words, d_model)
        self.d_model = d_model
        self.action_decoder = nn.Linear(d_model, params.n_actions)
        self.target_decoder = nn.Linear(d_model, params.n_targets)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        actions_out = self.action_decoder(output)
        targets_out = self.target_decoder(output)
        return actions_out, targets_out


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)