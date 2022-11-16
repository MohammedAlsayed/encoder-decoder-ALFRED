# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import torch

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
        
        self.lstm = nn.LSTM(params.embedding_dim, params.hidden_size, dropout=params.dropout, batch_first=True)
        self.fc_action = nn.Linear(params.hidden_size, params.n_actions)
        self.fc_target = nn.Linear(params.hidden_size, params.n_targets)
        

    def forward(self, action_input, target_input, hidden):
        # concat the embedding of action and target
        action_output = self.embedding_action(action_input)
        target_output = self.embedding_target(target_input)
        output = action_output + target_output
        output, hidden = self.lstm(output, hidden)
        action = self.fc_action(output)
        target = self.fc_target(output)

        return (action, target), hidden


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self):
        pass

    def forward(self, x):
        pass
