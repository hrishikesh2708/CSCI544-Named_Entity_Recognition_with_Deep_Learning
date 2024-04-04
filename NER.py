# %%
#libraries
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import (
    Dataset, DataLoader, random_split, SequentialSampler
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# %%
PROJECT_ROOT = './'
DATA_ROOT = './data'
TRAIN_DATA = './data/train'
DEV_DATA = './data/dev'
TEST_DATA = './data/test'
SAVED_MODELS_PATH = './saved_models'
GLOVE_PATH = './glove.6B.100d.gz'
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE_1 = 128
LEARNING_RATE_1 = 1.5
BATCH_SIZE_2 = 64
LEARNING_RATE_2 = 0.8

# %%
class BLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100,
                 word_embeddings=None, **kwargs):
        super(BLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size

        if word_embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(word_embeddings,
                                                          freeze=True,
                                                          padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim + 1,  # +1 for case_bool
            hidden_size=kwargs.get('hidden_size', 256),
            num_layers=kwargs.get('num_layers', 1),
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=kwargs.get('dropout', 0.33))
        self.linear = nn.Linear(
            in_features=2*kwargs.get('hidden_size', 256),
            out_features=128
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=self.tagset_size),
        )
        self.elu = nn.ELU()

    def forward(self, sentences, case_bool, lengths):
        x = self.embedding(sentences)
        case_bool = torch.unsqueeze(case_bool, dim=2)
        x = torch.cat([x, case_bool], dim=2)
        x = pack_padded_sequence(x, lengths, batch_first=True,
                                 enforce_sorted=False)

        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)

        x = self.linear(x)
        x = self.elu(x)
        x = self.classifier(x)

        return x

# %%
def read_data(filepath):
    print(f"INFO: Reading data from {filepath}")
    min_freq_thresh = 1
    with open(filepath, 'r') as file:
        lines = file.readlines()

    lines = [line.rstrip('\n') for line in lines]
    lines = [line.split(' ') for line in lines]
    counts = {}
    tags = []
    for line in tqdm(lines):
        if len(line) == 3:
            _, word, tag = line
            word = word.lower()
            tags.append(tag)
            if counts.get(word) is not None:
                counts[word] += 1
            else:
                counts[word] = 1
    counts = {k: v for k, v in sorted(counts.items(),
                                      key=lambda item: item[1],
                                      reverse=True)}

    words = ['<unk>']

    unknown_count = 0

    for i, (word, count) in enumerate(counts.items(), start=1):
        if count >= min_freq_thresh:
            words.append(word)
        else:
            unknown_count += count

    tagset = list(set(tags))
    tagset.sort()
    tag_to_idx = {tag: i for i, tag in enumerate(tagset)}
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
    tag_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(list(tag_to_idx.keys())),
        y=tags)
    tagset.__setitem__(-1, '<PAD>')

    return lines, words, tag_to_idx, idx_to_tag, tag_weights


def load_glove_vec():
    glove_file = pd.read_csv(GLOVE_PATH, sep=" ",quoting=3, header=None, index_col=0)
    embedding_glove = {key: val.values for key, val in glove_file.T.items()}
    embedding_glove['<unk>'] = np.zeros((100,))
    return embedding_glove


class NERDataset(Dataset):
    def __init__(self, filepath, vocab, tagset, no_targets=False):
        self.filepath = filepath 
        self.no_targets = no_targets 
        self.vocab = vocab 
        self.tagset = tagset
        self.sentences, self.targets = self.get_sentences()
        self.vocab_map = {word: i for (i, word) in enumerate(self.vocab)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        case_bool = torch.Tensor([
            0 if i.lower() == i else 1 for i in sentence
        ])
        sentence = torch.LongTensor([self.vocab_map.get(i.lower(), 0)
                                     for i in sentence])
        n_words = len(sentence)
        if not self.no_targets:
            targets = self.targets[idx]
            targets = torch.Tensor([self.tagset[i] for i in targets])
            return (sentence, case_bool, n_words), \
                targets.type(torch.LongTensor)
        else:
            return (sentence, case_bool, n_words), \
                None

    def get_sentences(self):
        with open(self.filepath, 'r') as file:
            lines = file.readlines()

        lines = [line.rstrip('\n') for line in lines]
        lines = [line.split(' ') for line in lines]
        dataset = []
        datum = []
        target = []
        targets = []
        for line in lines:
            if len(line) == 1:
                dataset.append(datum)
                datum = []
                if not self.no_targets:
                    targets.append(target)
                    target = []
            else:
                datum.append(line[1])
                if not self.no_targets:
                    target.append(line[2])
        dataset.append(datum)
        if not self.no_targets:
            targets.append(target)
        return dataset, targets


def collate_fn(data):
    max_len = max([l for (_, _, l), _ in data])
    batch_size = len(data)
    sentences_batched = torch.zeros((batch_size, max_len), dtype=torch.long)
    case_batched = torch.zeros((batch_size, max_len), dtype=torch.bool)
    lengths_batched = []
    targets_batched = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, ((sentence, case_bool, length), target) in enumerate(data):
        pad_length = max_len - length
        padding = torch.nn.ConstantPad1d((0, pad_length), 0)
        tag_padding = torch.nn.ConstantPad1d((0, pad_length), -1)
        sentence = padding(sentence)
        sentences_batched[i, :] = sentence
        case_bool = padding(case_bool)
        case_batched[i, :] = case_bool
        if target is not None:
            target = tag_padding(target)
            targets_batched[i, :] = target
        lengths_batched.append(length)
    sentences_batched = torch.Tensor(sentences_batched)
    case_batched = torch.Tensor(case_batched)
    lengths_batched = torch.Tensor(lengths_batched)
    targets_batched = torch.Tensor(targets_batched)
    return (sentences_batched, case_batched, lengths_batched), targets_batched


def get_dataloaders(train_data, split=True, **kwargs):
    train_dataset = NERDataset(train_data, kwargs['vocab'], kwargs['tagset'])
    if split:
        train_len = math.floor(0.8 * len(train_dataset))
        val_len = len(train_dataset) - train_len
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_len, val_len],
            torch.Generator().manual_seed(RANDOM_SEED))
        val_dataloader = DataLoader(
            val_dataset, batch_size=kwargs.get('batch_size', 128),
            shuffle=False, collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )
    train_dataloader = DataLoader(
        train_dataset, batch_size=kwargs.get('batch_size', 128),
        shuffle=True, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    if split:
        return train_dataloader, val_dataloader
    else:
        return train_dataloader, None


def train_model(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        num_epochs=30,
        lr_scheduler=None
):
    device = DEVICE

    model = model.to(device)

    for epoch in range(num_epochs):
        metrics = {'train_acc': 0, 'train_loss': 0.0, 'val_acc': 0, 'val_loss': 0.0}
        for i, ((X, case, lengths), y) in enumerate(tqdm(train_dataloader)):
            model.train()
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)           
            case = case.to(device)
            outputs = model(X, case, lengths)
            outputs = outputs.permute(0, 2, 1)  
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            metrics['train_acc'] += \
                (torch.argmax(outputs, axis=1) == y).float().sum() / sum(lengths)
            metrics['train_loss'] += loss

        metrics['train_acc'] /= len(train_dataloader)
        metrics['train_loss'] /= len(train_dataloader)
        for i, ((X, case, lengths), y) in enumerate(tqdm(val_dataloader)):
            model.eval()
            X = X.to(device) 
            y = y.to(device)
            case = case.to(device)
            outputs = model(X, case, lengths)
            outputs = outputs.permute(0, 2, 1)
            metrics['val_acc'] += \
                (torch.argmax(outputs, axis=1) == y).float().sum() / sum(lengths)
            metrics['val_loss'] += loss

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(metrics['val_loss'])
            else:
                lr_scheduler.step()

        metrics['val_acc'] /= len(val_dataloader)
        metrics['val_loss'] /= len(val_dataloader)

        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print("Mode\tLoss\tAcc")
        print(f"Train\t{metrics['train_loss']:.2f}\t{metrics['train_acc']:.2f}")
        print(f"Valid\t{metrics['val_loss']:.2f}\t{metrics['val_acc']:.2f}")

    return model, metrics


def generate_outputs(model, test_file, out_file,
                     no_targets=False, conll_eval=False, **kwargs):
    try:
        os.remove(out_file)
    except FileNotFoundError:
        pass

    vocab = kwargs['vocab']
    tag_to_idx = kwargs['tag_to_idx']
    idx_to_tag = kwargs['idx_to_tag']
    test_dataset = NERDataset(test_file, vocab,
                              tag_to_idx, no_targets)
    sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_1,
                                 shuffle=False, collate_fn=collate_fn,
                                 sampler=sampler)

    model = model.to(DEVICE)

    for i, ((X, case_bool, lengths), y) in enumerate(tqdm(test_dataloader)):
        model.eval()
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        case_bool = case_bool.to(DEVICE)

        output = model(X, case_bool, lengths)
        output = torch.argmax(output, axis=2)

        with open(out_file, 'a') as file:
            for j in range(len(output)):
                for k in range(int(lengths[j])):
                    sentence_idx = i * BATCH_SIZE_1 + j
                    if conll_eval:
                        file.write(
                            f'{k + 1} {test_dataset.sentences[sentence_idx][k]}'
                            f' {idx_to_tag[int(y[j][k])]} '
                            f'{idx_to_tag[int(output[j][k])]}\n')
                    else:
                        file.write(
                            f'{k + 1} {test_dataset.sentences[sentence_idx][k]}'
                            f' {idx_to_tag[int(output[j][k])]}\n')
                file.write('\n')

    return

# %%
def Task1():
    _, vocab, tag_to_idx, idx_to_tag, tag_weight = read_data(TRAIN_DATA)
    tag_weight = torch.Tensor(tag_weight).to(DEVICE)
    num_tags = len(tag_to_idx)
    train_dataloader, val_dataloader = get_dataloaders(
        TRAIN_DATA, vocab=vocab, tagset=tag_to_idx, batch_size=BATCH_SIZE_1)
    model = BLSTM(len(vocab), num_tags, embedding_dim=100)
    criterion = nn.CrossEntropyLoss(weight=tag_weight, ignore_index=-1)
    optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_1,
                            momentum=0.1)
    model, metrics = train_model(
        model=model,
        optimizer=optim,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=20,
    )
    torch.save(model, os.path.join(SAVED_MODELS_PATH, 'blstm1.pt'))
    return


def Task2():
    _, vocab, tag_to_idx, idx_to_tag, tag_weight = read_data(TRAIN_DATA)
    glove_vec = load_glove_vec()
    word_embeddings = torch.Tensor(np.vstack(list(glove_vec.values())))
    num_tags = len(tag_to_idx)
    tag_weight = torch.Tensor(tag_weight).to(DEVICE)
    train_dataloader, val_dataloader = get_dataloaders(
        TRAIN_DATA, vocab=list(glove_vec.keys()), tagset=tag_to_idx,
        batch_size=BATCH_SIZE_2
    )
    model = BLSTM(
        vocab_size=len(list(glove_vec.keys())),
        tagset_size=num_tags,
        embedding_dim=100,
        word_embeddings=word_embeddings
    )
    criterion = nn.CrossEntropyLoss(weight=tag_weight, ignore_index=-1)
    optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_2,
                            momentum=0.5)
    model, metrics = train_model(
        model=model,
        optimizer=optim,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=30,
    )
    torch.save(model, os.path.join(SAVED_MODELS_PATH, 'blstm2.pt'))
    return


def Evaluation():
    print("****** Task 1 ******")
    _, vocab, tag_to_idx, idx_to_tag, _ = read_data(TRAIN_DATA)
    model = torch.load(os.path.join(SAVED_MODELS_PATH, 'blstm1.pt'))
    model.eval()
    print("INFO: Running inference on the dev set")
    generate_outputs(model, DEV_DATA, 'dev1.out', conll_eval=False,
                     vocab=vocab, idx_to_tag=idx_to_tag, tag_to_idx=tag_to_idx)
    print("INFO: Running inference on the test set")
    generate_outputs(model, TEST_DATA, 'test1.out', conll_eval=False,
                     vocab=vocab, idx_to_tag=idx_to_tag, tag_to_idx=tag_to_idx,
                     no_targets=True)
    print("****** Task 2 ******")
    _, vocab, tag_to_idx, idx_to_tag, _ = read_data(TRAIN_DATA)
    glove_vec = load_glove_vec()

    model = torch.load(os.path.join(SAVED_MODELS_PATH, 'blstm2.pt'))
    model.eval()
    print("INFO: Running inference on the dev set")
    generate_outputs(model, DEV_DATA, 'dev2.out', conll_eval=False,
                     vocab=list(glove_vec.keys()), idx_to_tag=idx_to_tag,
                     tag_to_idx=tag_to_idx)
    print("INFO: Running inference on the test set")
    generate_outputs(model, TEST_DATA, 'test2.out', conll_eval=False,
                     vocab=list(glove_vec.keys()), idx_to_tag=idx_to_tag,
                     tag_to_idx=tag_to_idx, no_targets=True)
    return

# %%
Task1()

# %%
Task2()

# %%
Evaluation()

# # %%
# !python eval.py -p dev1.out -g ./data/dev

# # %%
# !python eval.py -p dev2.out -g ./data/dev


# bonus task
import torch.nn.functional as F

class BLSTMWithCNN(nn.Module):
    def __init__(self, vocab_size, tagset_size, char_vocab_size, char_embedding_dim,
                 word_embedding_dim=100, word_embeddings=None, cnn_output_dim=128,
                 num_cnn_layers=1, cnn_kernel_sizes=[3], hidden_size=256, num_layers=1,
                 dropout=0.33):
        super(BLSTMWithCNN, self).__init__()
        self.vocab_size = vocab_size
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.tagset_size = tagset_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_cnn_layers = num_cnn_layers
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.cnn_output_dim = cnn_output_dim

        # Word embeddings
        if word_embeddings is None:
            self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=True, padding_idx=0)

        # Character embeddings and CNN
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(char_embedding_dim, cnn_output_dim, kernel_size=kernel_size)
            for kernel_size in cnn_kernel_sizes
        ])

        # BLSTM
        self.lstm = nn.LSTM(
            input_size=word_embedding_dim + len(cnn_kernel_sizes) * cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Linear and classifier layers
        self.linear = nn.Linear(2 * hidden_size, 128)
        self.classifier = nn.Linear(128, tagset_size)

    def forward(self, sentences, char_sequences, lengths):
        # Word embeddings
        word_embedded = self.word_embedding(sentences)

        # Character embeddings and CNN
        char_embedded = self.char_embedding(char_sequences)
        char_embedded = char_embedded.permute(0, 2, 1)  # Reshape for CNN
        cnn_outputs = []
        for conv_layer in self.conv_layers:
            cnn_output = F.relu(conv_layer(char_embedded))
            cnn_output, _ = torch.max(cnn_output, dim=2)
            cnn_outputs.append(cnn_output)
        cnn_outputs = torch.cat(cnn_outputs, dim=1)

        # Concatenate word embeddings and CNN outputs
        combined = torch.cat((word_embedded, cnn_outputs), dim=2)

        # BLSTM
        packed = pack_padded_sequence(combined, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # Linear and classifier layers
        linear_out = self.linear(lstm_out)
        linear_out = F.elu(linear_out)
        logits = self.classifier(linear_out)

        return logits
    
    # Train the model

# _, vocab, tag_to_idx, idx_to_tag, tag_weight = read_data(train_data_loc)
# glove_vec = load_glove_vec(glove_embed_loc)
# char_vocab_size = len(char_to_idx)  # Assuming char_to_idx is defined
# model = BLSTMWithCNN(
#     vocab_size=len(glove_vec),
#     tagset_size=len(tag_to_idx),
#     char_vocab_size=char_vocab_size,
#     char_embedding_dim=30,
#     word_embedding_dim=100,
#     word_embeddings=torch.Tensor(np.vstack(list(glove_vec.values()))),
#     cnn_output_dim=128,
#     num_cnn_layers=2,  # You can tune these hyperparameters
#     cnn_kernel_sizes=[3, 5],
#     hidden_size=256,
#     num_layers=1,
#     dropout=0.33
# )
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor(tag_weight).to(device), ignore_index=-1)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.5)
# train_dataloader, val_dataloader = get_dataloaders(
#         train_data_loc, vocab=vocab, tagset=tag_to_idx, batch_size=batch_sz_1)
# train_model(
#     model=model,
#     optimizer=optimizer,
#     criterion=criterion,
#     train_dataloader=train_dataloader,
#     val_dataloader=val_dataloader,
#     num_epochs=30,
# )

# # Evaluate the model on the development data
# model.eval()
# generate_outputs(
#     model=model,
#     test_file=dev_data_loc,
#     out_file='dev_predictions.out',
#     no_targets=False,
#     conll_eval=False,
#     vocab=list(glove_vec.keys()),
#     idx_to_tag=idx_to_tag,
#     tag_to_idx=tag_to_idx
# )

# # Compute precision, recall, and F1 score on the development data
# !python eval.py -p dev_predictions.out -g ./data/dev

# # Generate predictions on the test data
# generate_outputs(
#     model=model,
#     test_file=test_data_loc,
#     out_file='test_predictions.out',
#     no_targets=True,
#     conll_eval=False,
#     vocab=list(glove_vec.keys()),
#     idx_to_tag=idx_to_tag,
#     tag_to_idx=tag_to_idx
# )