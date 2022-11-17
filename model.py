# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import torch
import torch.nn.functional as F

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

        self.embedding_action = nn.Embedding(self.n_actions, self.hidden_size, padding_idx=0)
        self.embedding_target = nn.Embedding(self.n_targets, self.hidden_size, padding_idx=0)
    
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)

        self.dropout = nn.Dropout(self.dropout_p)
        self.a = nn.Linear(self.hidden_size * 2, 1)    
        self.fc_action = nn.Linear(self.hidden_size, self.n_actions)
        self.fc_target = nn.Linear(self.hidden_size, self.n_targets)

        

    def forward(self, action_input, target_input, hidden_d, hidden_e, length):
        # concat the embedding of action and target
        action_output = self.embedding_action(action_input)
        target_output = self.embedding_target(target_input)
        output = action_output + target_output
        output = self.dropout(output)
        output = F.relu(output)
        
        output, hidden = self.lstm(output, hidden_d)
        
        batch_size = action_output.shape[0]
        hidden_d = hidden[0].view(batch_size,1,-1) # make batch dim at 0

        hidden_d = hidden_d.expand(-1, hidden_e.shape[1], -1) # expand decoder dim to the same as all encoder hidden states
                                                              # in other words, replicate the decoding embedding across input_size 

        x = torch.cat((hidden_e, hidden_d), dim=2) # concat last decode hidden with all encode hiddens, hidden dimesion doubles

        a = self.a(x) # FC layer to a scaler

        attn_weights = F.softmax(a, dim=1) # softmax across sequence (sum of sequence probability = 1)
        
        attn = torch.mul(hidden_e, attn_weights) # multiply every encoder hidden output by it's atten_weight 

        attn_out = torch.sum(attn, dim=1) # sum all of the weigthed attn

        action = self.fc_action(attn_out)
        target = self.fc_target(attn_out)

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
