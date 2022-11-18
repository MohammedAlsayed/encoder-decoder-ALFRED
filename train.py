import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from model import Encoder
from model import Decoder
from model import AttentionDecoder
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import wandb
import json
import itertools
from nltk.stem.porter import *

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match,
    encode_data,
    LCS,
    load_glove_model
)

# zero reserved for padding
BOS_token = 1
EOS_token = 2

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    f = open("lang_to_sem_data.json")
    data_json = json.load(f)
    # creating data tokens
    v2id, id2v, input_size = build_tokenizer_table(data_json['train'])
    a2id, id2a, t2id, id2t, output_size = build_output_tables(data_json['train'])
    args.n_words = len(v2id)
    args.n_actions = len(a2id)
    args.n_targets = len(t2id)
    args.output_size = output_size

    print("sequence length = ", input_size)
    print("output length = ", output_size)

    if args.glove:
        not_found = 0
        stemmer = PorterStemmer()
        glove_model = load_glove_model("glove.6B.300d.txt")
        embedding_matrix = np.zeros((len(v2id), args.embedding_dim))
        np.random.seed(10)
        for word in v2id.keys():
            index = v2id[word]
            if word in glove_model:
                vector = glove_model[word]
            elif word.lower() in glove_model:
                vector = glove_model[word.lower()]
            elif stemmer.stem(word) in glove_model:
                vector = glove_model[stemmer.stem(word)]
            else:
                not_found += 1
                vector = np.random.rand(args.embedding_dim)
            embedding_matrix[index] = vector
        args.pre_embeddings = embedding_matrix
        print(f"total words found in Glove ={len(v2id)-not_found} out of {len(v2id)}!")

    # preparing train and valid data
    x_train, y_train, l_train, train_output_lengths = encode_data(data_json['train'], v2id, a2id, t2id, input_size, output_size)
    x_valid, y_valid, l_valid, valid_output_lengths = encode_data(data_json['valid_seen'], v2id, a2id, t2id, input_size, output_size)

    # args.train_input_length = train_input_length

    train_dataset = TensorDataset(torch.from_numpy(x_train[0:3000]), torch.from_numpy(y_train[0:3000]), torch.from_numpy(l_train[0:3000]), torch.from_numpy(train_output_lengths[0:3000]))
    val_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid), torch.from_numpy(l_valid), torch.from_numpy(valid_output_lengths))
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader, v2id


def setup_model(args, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #

    encoder = Encoder(args, device)
    decoder = Decoder(args)
    attentionDecoder = AttentionDecoder(args)
    model = {}
    model['encoder'] = encoder
    model['decoder'] = decoder
    model['attentionDecoder'] = attentionDecoder
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #

    criterion = torch.nn.CrossEntropyLoss()

    encoder_optimizer = torch.optim.Adam(model['encoder'].parameters(), lr=args.learning_rate)

    if args.attention:
        decoder_optimizer = torch.optim.Adam(model['attentionDecoder'].parameters(), lr=args.learning_rate)
    else:    
        decoder_optimizer = torch.optim.Adam(model['decoder'].parameters(), lr=args.learning_rate)
    

    optimizer = {}
    optimizer['encoder_optimizer'] = encoder_optimizer
    optimizer['decoder_optimizer'] = decoder_optimizer

    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """
    
    encoder = model['encoder']
    decoder = model['decoder']
    attentionDecoder = model['attentionDecoder']
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_action_acc = 0.0
    epoch_target_acc = 0.0

    # iterate over each batch in the dataloader
    # length is the original length of the
    for (inputs, labels, length, o_length) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        batch_size = inputs.shape[0]
        encoder_hidden = encoder.initHidden(batch_size)
        encoder_output, encoder_hidden = encoder(inputs, encoder_hidden, length)
        
        decoder_action_input = torch.tensor([[BOS_token]]*batch_size, device=device)
        decoder_target_input = torch.tensor([[BOS_token]]*batch_size, device=device)

        decoder_hidden = encoder_hidden
        
        # for longest common subsequence calculations, save the output of the decoder
        action_output = np.zeros((batch_size, args.output_size), dtype=np.int32) # decoded sequence of actions
        target_output = np.zeros((batch_size, args.output_size), dtype=np.int32) # decoded sequence of targets

        loss = 0
        # decode action, target one by one accross batch
        for di in range(args.output_size):

            if args.attention:
                decoder_output, decoder_hidden = attentionDecoder(di, decoder_action_input, decoder_target_input, decoder_hidden, encoder_output)
            else:
                decoder_output, decoder_hidden = decoder(decoder_action_input, decoder_target_input, decoder_hidden)            
            
            action_label = labels[:, 0:1, di:di+1].view(batch_size, -1) # slice action labels across batch = BATCH_SIZE x 1
            target_label = labels[:, 1:2, di:di+1].view(batch_size, -1) # slice target labels across batch = BATCH_SIZE x 1 
            
            # transforming label to one hot encoding for criterion
            action_label_encoded = F.one_hot(action_label.to(torch.int64), num_classes=args.n_actions)
            target_label_encoded = F.one_hot(target_label.to(torch.int64), num_classes=args.n_targets)
           
            # summing action and target loss of one decode step
            loss += criterion(decoder_output[0].view(batch_size, -1), action_label_encoded.view(batch_size, -1).float()) 
            loss += criterion(decoder_output[1].view(batch_size, -1), target_label_encoded.view(batch_size, -1).float())
            
            
            # saving the decode output for LCS score computation
            action_output[:, di:di+1] = torch.argmax(F.log_softmax(decoder_output[0].view(batch_size, -1), dim=1), dim=1).view(batch_size,1)
            target_output[:, di:di+1] = torch.argmax(F.log_softmax(decoder_output[1].view(batch_size, -1), dim=1), dim=1).view(batch_size,1)

            # Teacher forcing
            # change the next input of the decoder to the ground truth
            decoder_action_input = action_label
            decoder_target_input = target_label
        
        # step optimizer and compute gradients during training
        if training:
            loss.backward()
            
            optimizer['encoder_optimizer'].step()
            optimizer['decoder_optimizer'].step()

            optimizer['encoder_optimizer'].zero_grad()
            optimizer['decoder_optimizer'].zero_grad()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        # em = output == labels
        # prefix_em = prefix_em(output, labels)

        # find LCS score across the batch 
        action_acc = LCS(action_output, labels[:, 0:1, :], o_length)
        target_acc = LCS(target_output, labels[:, 1:2, :], o_length)
        total_acc = (action_acc + target_acc) / 2
        
        # logging
        epoch_acc += total_acc
        epoch_action_acc += action_acc
        epoch_target_acc += target_acc
        epoch_loss += loss.item()
    
    epoch_loss /= len(loader)
    epoch_acc /= len(loader)
    epoch_action_acc /= len(loader)
    epoch_target_acc /= len(loader)

    return epoch_loss, epoch_acc, epoch_action_acc, epoch_target_acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model['encoder'].eval()
    if args.attention:
        model['attentionDecoder'].eval()
    else:
        model['decoder'].eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc, val_action_acc, val_target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc, val_action_acc, val_target_acc


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model['encoder'].train()

    if args.attention:
        model['attentionDecoder'].train()
    else:
        model['decoder'].train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc, train_action_acc, train_target_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        # some logging
        wandb.log({"train loss": train_loss})
        wandb.log({"train total LCS acc": train_acc})
        wandb.log({"train LCS action acc": train_action_acc})
        wandb.log({"train LCS target acc": train_target_acc})
        
        print(f"\ntrain loss : {train_loss}")
        print(f"train total LCS acc : {train_acc}")
        print(f"train LCS action acc : {train_action_acc}")
        print(f"train LCS target acc : {train_target_acc}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            print(f"\nval loss : {val_loss} | val total LCS acc: {val_acc}")
            print(f"val LCS action acc : {val_action_acc} | val LCS target acc: {val_target_acc}")
            
            wandb.log({"val loss": val_loss})
            wandb.log({"val total LCS acc": val_acc})
            wandb.log({"val LCS action acc": val_action_acc})
            wandb.log({"val LCS target acc": val_target_acc})

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):

    wandb.init(project="hw3", entity="alsayedm")
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, v2id = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device)
    print(model)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc, _ , _ = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--attention", action="store_true", help="global attention on decoder")
    parser.add_argument("--local_attention", action="store_true", help="local attention on decoder")
    parser.add_argument("--glove", action="store_true", help="use glove word embedding")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=16, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )
    parser.add_argument("--learning_rate", default=0.001, help="learning rate", type=float)
    parser.add_argument("--embedding_dim", default=300, help="number of embedding dimensions", type=int)
    parser.add_argument("--dropout", default=0.33, help="dropout rate of the neural net", type=float)
    parser.add_argument("--hidden_size", default=256, help="LSTM hidden dimension size", type=int)

    args = parser.parse_args()
    
    print(f"using attention: {args.attention}, local attention: {args.local_attention}, glove embeddings: {args.glove}")

    main(args)
