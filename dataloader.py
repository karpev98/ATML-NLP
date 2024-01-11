import numpy as np
from transformers import BertTokenizer, BertModel, PreTrainedTokenizerBase, PreTrainedTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)


class TransformerDataset(Dataset):
    def __init__(self, X, labels, z, max_length, converter):
        super().__init__()
        self.X = X
        self.labels = labels
        self.max_length = max_length
        self.converter = converter
        self.z = z

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        text = self.X[item]
        label = self.labels[item]
        encoding = self.converter.tokenize_data(text)
        categories = self.converter.translate_labels(label)
        return encoding, categories, self.z[item]


def split(X, y, seed, z=None, train_size=0.7, test_size=0.3, val_size=0.2):
    X_val = None
    y_val = None
    Z_train, Z_val, Z_test = None, None, None
    if train_size is None and test_size is None:
        raise AttributeError()
    elif train_size is None:
        train_size = 1. - test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, shuffle=True,
                                                        train_size=train_size)
    if z is not None:
        Z_train, Z_test = train_test_split(z, stratify=y, random_state=seed, shuffle=True, train_size=train_size)
    if val_size is not None:
        if z is not None:
            Z_train, Z_val = train_test_split(Z_train, stratify=y_train, random_state=seed, shuffle=True,
                                              test_size=val_size)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, random_state=seed,
                                                          shuffle=True, test_size=val_size)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (Z_train, Z_val, Z_test)


def get_dataset(
        df: str | pd.DataFrame = 'dataset/EDOS_1M_balanced.pkl',
        verbose=0,
        seed=0
):
    generator = np.random.default_rng(seed)
    df = df
    if isinstance(df, str):
        if df.endswith('pkl'):
            df = pd.read_pickle(df)
        else:
            df = pd.read_csv(df)
    groups = []
    labels = []
    confidence = []
    # tokenizer = BertTokenizer.from_pretrained()
    if verbose:
        break_line = '#' * 80
        logging.info(break_line)
        logging.info(f'Generating dataset...')
    for name, group in df.groupby(by='eb+_emot'):
        group = group.reset_index(drop=True)
        groups.append(group.loc[:, 'uttr'])
        confidence.append(group.loc[:, 'label_confidence'])
        labels.append(group.loc[:, 'eb+_emot'])

    groups = np.array(groups, dtype=str)
    labels = np.array(labels, dtype=str)

    confidence = np.array(confidence, dtype=np.float32)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), (Z_train, Z_val, Z_test) = split(groups.reshape(-1),
                                                                                           labels.reshape(-1), seed,
                                                                                           confidence.reshape(-1))

    bert_model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)


    train_dataset = TransformerDataset(X_train, y_train, Z_train, max_length=100)
    test_dataset = TransformerDataset(X_test, y_test, Z_test, max_length=100)
    val_dataset = TransformerDataset(X_val, y_val, Z_val, max_length=100)

    return train_dataset, test_dataset, val_dataset




if __name__ == '__main__':
    train_loader, *_ = get_dataset()
    print(train_loader.__getitem__(0))
