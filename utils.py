import time

import numpy as np
import pandas as pd
import whisper
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from transformers import AutoConfig, AutoModel
import torch
from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave
import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

def split(X, y, seed, z=None, train_size=0.7, test_size=0.3, val_size=0.2):
    """
    This function split the dataset using train_test_split stratified
    :param X:
    :param y:
    :param seed:
    :param z:
    :param train_size:
    :param test_size:
    :param val_size:
    :return:
    """
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


def get_edos_dataset_balanced(
        df: str | pd.DataFrame = 'dataset/EDOS_1M_balanced.pkl'
):
    """
    This function returns the edos balanced dataset divided in inputs, labels, confidence levels
    :param df: pandas DataFrame of edos data
    :return: inputs, labels, confidence levels
    """
    df = df
    if isinstance(df, str):
        if df.endswith('pkl'):
            df = pd.read_pickle(df)
        else:
            df = pd.read_csv(df)
    groups = []
    labels = []
    confidence = []
    for name, group in df.groupby(by='eb+_emot'):
        group = group.reset_index(drop=True)
        groups.append(group.loc[:, 'uttr'])
        confidence.append(group.loc[:, 'label_confidence'])
        labels.append(group.loc[:, 'eb+_emot'])

    groups = np.array(groups, dtype=str)
    labels = np.array(labels, dtype=str)

    confidence = np.array(confidence, dtype=np.float32)
    return groups, labels, confidence


class Agglomerate:
    def __init__(self, device):
        self.device = device

    def __call__(self, *args, **kwargs):
        pass


class Concat(Agglomerate):
    def __init__(self, device, name='bert-base-uncased'):
        super(Concat, self).__init__(device)
        config = AutoConfig.from_pretrained(name)
        config.update({"output_hidden_states": True})
        self.encoder = AutoModel.from_pretrained(name, config=config)
        self.encoder.to(device)
        self.encoder.eval()
        for param in self.encoder.parameters(recurse=True):
            param.requires_grad = False

    def __call__(self, batch):
        with torch.no_grad():
            x = self.encoder(batch['input_ids'].to(self.device),
                             batch['attention_mask'].to(self.device))  # TODO: check against ** batch
            hidden_states = torch.stack(x["hidden_states"])
            concatenated = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]), -1)
            concatenated = concatenated[:, 0]
        return concatenated


class EmotionClassifierConcat(nn.Module):
    def __init__(self, output_size, device, hidden_layers, activation_fn=nn.ReLU, dropout_value=0.):
        super(EmotionClassifierConcat, self).__init__()
        self.classification = nn.Sequential(
        )
        for i in range(len(hidden_layers)):
            curr = hidden_layers[i]
            if i == 0:
                self.classification.append(nn.LazyLinear(curr))
                self.classification.append(activation_fn())
                if dropout_value > 0:
                    self.classification.append(nn.Dropout(dropout_value))
                continue
            prev = hidden_layers[i - 1]
            self.classification.append(nn.Linear(prev, curr))
            self.classification.append(activation_fn())
            if dropout_value > 0:
                self.classification.append(nn.Dropout(dropout_value))

        self.classification.append(nn.Linear(hidden_layers[-1], output_size))
        # self.classification.append(nn.Softmax(dim=-1))

        self.device = device

    def forward(self, concatenated):
        return self.classification(concatenated)


def compute_metrics(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy().argmax(-1)
    accuracy = accuracy_score(y_true, y_pred)
    f1_score_value = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1_score_value


def decode(input_, y_true, y_pred, confidence, tokenizer, category_to_emotion, top_n=5):
    input_ = input_[0].detach().cpu()
    y_true = y_true[0].detach().cpu().item()
    y_pred = y_pred[0].detach().cpu().numpy()
    true_confidence = confidence[0]
    pred_confidence = y_pred.max()
    top_5_emotions = None
    if top_n > 0:
        index = y_pred.argsort()[-top_n:]
        top_5_emotions = [category_to_emotion[i] for i in index]
    y_pred = y_pred.argmax()
    input_ = tokenizer.decode(input_)
    y_true = category_to_emotion[y_true]
    y_pred = category_to_emotion[y_pred]
    return input_, y_true, y_pred, true_confidence, pred_confidence, top_5_emotions


def top(y_true, y_pred, top_n=5):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = np.argsort(y_pred, axis=-1)[:, -top_n:]
    return np.mean(np.isin(y_true[:, np.newaxis], y_pred))


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_val = np.inf
        self.counter = 0

    def __call__(self, val: float):
        save = False
        if val < self.best_val:
            self.counter = 0
            self.best_val = val
            save = True
        else:
            self.counter += 1
        return self.counter >= self.patience, save


def dialogue_edos_text(model: nn.Module, agglomerate: Agglomerate, tokenizer, category_to_emotion, top_n=5):
    model.eval()
    with torch.no_grad():
        while True:
            user_text = input('ENTER SENTENCE:')
            if user_text.lower() == 'exit' or len(user_text) == 0:
                print('BYE!')
                return
            print(f'USER: {user_text}')
            assert isinstance(user_text, str)
            batch = tokenizer(user_text, padding=True, return_tensors='pt')
            batch = agglomerate(batch)
            prediction = model.forward(batch)

            prediction = nn.functional.softmax(prediction, dim=-1).cpu().numpy().flatten()

            predicated_label = prediction.argmax()
            probability = np.max(prediction)
            predicated_label = category_to_emotion[predicated_label]
            top_5_prediction = prediction.argsort()[-top_n:]
            top_5_probability = prediction[top_5_prediction][::-1]
            top_5 = np.vectorize(lambda x: category_to_emotion[x])(top_5_prediction)[::-1]
            dictionary = {key: value for key, value in zip(top_5, top_5_probability)}
            print(f'BOT: {predicated_label} with prob: {probability}')
            print(f'TOP {top_n}: {dictionary}')
            time.sleep(2)


def dialogue_edos_mic(model: nn.Module, agglomerate: Agglomerate, tokenizer, category_to_emotion, top_n=5):
    model.eval()
    with torch.no_grad():
        print('RECORDING IN...', end=' ')
        for i in reversed(range(1, 4)):
            print(i, end=' ')
            time.sleep(1)
        print('RECORDING STARTED')
        record_to_file('demo_simple.wav')
        user_text = parse_audio('demo_simple.wav')
        print('USER:', user_text)
        assert isinstance(user_text, str)
        batch = tokenizer(user_text, padding=True, return_tensors='pt')
        batch = agglomerate(batch)
        prediction = model.forward(batch)

        prediction = nn.functional.softmax(prediction, dim=-1).cpu().numpy().flatten()

        predicated_label = prediction.argmax()
        probability = np.max(prediction)
        predicated_label = category_to_emotion[predicated_label]
        top_5_prediction = prediction.argsort()[-top_n:]
        top_5_probability = prediction[top_5_prediction][::-1]
        top_5 = np.vectorize(lambda x: category_to_emotion[x])(top_5_prediction)[::-1]
        dictionary = {key: value for key, value in zip(top_5, top_5_probability)}
        print(f'BOT: {predicated_label} with prob: {probability}')
        print(f'TOP {top_n}: {dictionary}')
        time.sleep(2)


def test_edos_text_model(model: nn.Module, agglomerate: Agglomerate, test_dataloader, category_to_emotion, top_n=5):
    model.eval()
    with torch.no_grad():
        real_labels = []
        out_labels = []
        accuracies = []
        f1_scores = []
        top_5_ = []
        for batch, labels, confidence in tqdm(test_dataloader):
            batch = agglomerate(batch)
            out = model.forward(batch)
            ac, f1 = compute_metrics(labels, out)
            accuracies.append(ac)
            f1_scores.append(f1)
            real_labels.append(labels.cpu().numpy())
            out = nn.functional.softmax(out, -1)
            top_5_.append(top(labels, out, top_n))
            out_labels.append(out.argmax(-1).cpu().numpy())

    return real_labels, out_labels, accuracies, f1_scores, top_5_


class OurDataset(Dataset):
    def __init__(self, X, y, z):
        self.X = X
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.z[item]


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100


# TAKEN FROM: https://stackoverflow.com/questions/892199/detect-record-audio-in-python (cryo)
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    "Trim the blank spots at the start and end"

    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r


def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 120:
            break
    print(num_silent)
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


speech_to_text = whisper.load_model("base.en", in_memory=True)
options = whisper.DecodingOptions(fp16=False, language="en")


def parse_audio(path):
    global speech_to_text, options
    audio = whisper.load_audio(path, 16_000)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(speech_to_text.device)
    result = whisper.decode(speech_to_text, mel, options)
    return result.text
