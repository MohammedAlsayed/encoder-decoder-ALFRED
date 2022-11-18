import re
import torch
import numpy as np
from collections import Counter
from rouge import Rouge

def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    for episode in train:
        padded_len = 2  # start/end
        for inst, _ in episode:
            inst = preprocess_string(inst)
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1 # calculate the length of the episode
        padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        max(padded_lens)
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    output_size = []
    for episode in train:
        output_size.append(len(episode))
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)

    actions_to_index = {a: i+3 for i, a in enumerate(actions)}
    actions_to_index["<pad>"] = 0
    actions_to_index["<start>"] = 1
    actions_to_index["<end>"] = 2

    targets_to_index = {t: i+3 for i, t in enumerate(targets)}
    targets_to_index["<pad>"] = 0
    targets_to_index["<start>"] = 1
    targets_to_index["<end>"] = 2

    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets, max(output_size)+2 # because of <start> & <end> & tokens

def encode_data(data, v2i, a2id, t2id, input_size, output_size):
    episodes = []
    actions = []
    targets = []
    input_length = []
    output_lengths = []
    rows = len(data)
    for episode in data:
        idx = 0 # to limit the number of actions (DEBUG purposes)
        i = []
        a = []
        t = []
        i.append(v2i['<start>'])
        a.append(a2id['<start>'])
        t.append(a2id['<start>'])

        for inst, label in episode:
            # if idx >= output_size-2:
            #     break
            for word in inst.split():
                i.append(v2i[word] if word in v2i else v2i["<unk>"])

            a.append(a2id[label[0]])
            t.append(t2id[label[1]])
            idx += 1
        
        i.append(v2i['<end>'])
        a.append(v2i['<end>'])
        t.append(v2i['<end>'])
        input_length.append(len(i))
        episodes.append(i)
        actions.append(a)
        targets.append(t)
        output_lengths.append(len(a))

    # 2322 episode looked weird LOL
    x = np.zeros((rows, input_size), dtype=np.int32) # number of episodes x input_size
    y = np.zeros((rows, 2, output_size), dtype=np.int32) # number of episodes x num_labels x number of instructions
    l = np.asarray(input_length)
    output_lengths = np.asarray(output_lengths)

    n_early_cutoff = 0
    for idx, e in enumerate(episodes):
        if len(e) <= input_size:
            x[idx, 0:len(e)] = e
        else:
            x[idx, 0:input_size] = e[0:input_size]
            n_early_cutoff += 1
    
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, input_size)
    )
    
    for idx, a in enumerate(actions):
        y[idx, 0, 0:len(a)] = a

    for idx, t in enumerate(targets):
        y[idx, 1, 0:len(t)] = t

    return x, y, l, output_lengths
    

def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1 

    seq_length = len(gt_labels)
    
    for i in range(seq_length):
        if predicted_labels[i] != gt_labels[i]:
            break
    
    pm = (1.0 / seq_length) * i

    return pm

# predicted outputs, true_lables, number of outputs (to skip padding)
def LCS(predicted, labels, o_length):
    rouge = Rouge()
    total_score = 0
    for bi in range(predicted.shape[0]):
        length = o_length[bi].item()
        
        p = predicted[bi]
        p = p[0:length]
        
        l = labels[bi]
        l = l[:, 0:length][0]
        
        p = " ".join(map(str, p.tolist()))
        l = " ".join(map(str, l.tolist()))

        score =  rouge.get_scores(p, l)[0]['rouge-l']['f']
        total_score += score
    
    return score / predicted.shape[0]


def load_glove_model(glove_path):
    print("Loading Glove 300 Model")
    glove_model = {}
    with open(glove_path,'rb') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0].decode()
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model