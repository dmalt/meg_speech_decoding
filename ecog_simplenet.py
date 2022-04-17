import math

import h5py
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy.signal
import sklearn
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tqdm

DATA_FOLDER = "/home/altukhov/Data/speech_dl/Procenko"

FILES_LIST = [
    f"{DATA_FOLDER}/Patient001_test0042021.03.11_19.12.59.hdf5",
    f"{DATA_FOLDER}/Patient001_test0052021.03.11_19.26.24.hdf5",
    f"{DATA_FOLDER}/Patient001_test0062021.03.11_19.42.41.hdf5",
    f"{DATA_FOLDER}/Patient001_test0072021.03.11_20.40.02.hdf5",
]


def data_generator(
    X, Y, batch_size, lag_backward, lag_forward, shuffle=True, infinite=True
):
    assert len(X) == len(Y) or len(Y) == 0
    total_lag = lag_backward + lag_forward
    all_batches = math.ceil((X.shape[0] - total_lag) / batch_size)
    batch = 0
    random_core = np.arange(lag_backward, X.shape[0] - lag_forward)
    while True:
        if shuffle:
            np.random.shuffle(random_core)
        for batch in range(all_batches):
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size
            if batch_end >= len(random_core):
                batch_end = None
            batch_samples = random_core[batch_start:batch_end]

            batch_x = np.array(
                [X[i - lag_backward : i + lag_forward] for i in batch_samples]
            )
            batch_x = np.swapaxes(batch_x, 1, 2)

            if len(Y) > 0:
                batch_y = Y[[batch_samples]]
                yield (batch_x, batch_y)
            else:
                yield batch_x

        if not infinite:
            break


def notch_filtering(ecog, frequency):
    ecog_filtered = ecog
    for w0 in [50, 100, 150]:
        norch_b, norch_a = scipy.signal.iirnotch(w0, Q=10, fs=frequency)
        ecog_filtered = scipy.signal.filtfilt(
            norch_b, norch_a, ecog_filtered, axis=0
        )
    return ecog_filtered


def extract_sound_log_envelope(sound, frequency):
    LOW_PASS_FREQUENCY = 200
    HIGH_PASS_FREQUENCY = 200
    blp, alp = scipy.signal.butter(
        3, LOW_PASS_FREQUENCY / (frequency / 2), btype="low", analog=False
    )
    bhp, ahp = scipy.signal.butter(
        5, HIGH_PASS_FREQUENCY / (frequency / 2), btype="high", analog=False
    )

    sound_filtered = scipy.signal.filtfilt(bhp, ahp, sound)
    envelope = scipy.signal.filtfilt(blp, alp, np.log(np.abs(sound_filtered)))
    return envelope


def envelope_signals(signals):
    enveloped_signals = np.zeros(signals.shape)
    for i in range(signals.shape[1]):
        enveloped_signals[:, i] = np.abs(scipy.signal.hilbert(signals[:, i]))
    return enveloped_signals


def remove_eyes_artifacts(ecog, frequency):
    HIGH_PASS_FREQUENCY = 20
    bgamma, agamma = scipy.signal.butter(
        5, HIGH_PASS_FREQUENCY / (frequency / 2), btype="high"
    )
    return scipy.signal.filtfilt(bgamma, agamma, ecog, axis=0)


def extract_mfccs(sound, sr, i_mel, lngt):
    S = librosa.feature.melspectrogram(
        y=sound, sr=sr, n_mels=16, fmax=4000, hop_length=256
    )
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=16)
    the_mfcc = scipy.signal.resample(x=mfccs[i_mel, :].T, num=lngt).T
    return the_mfcc


FREQUENCY = 19200
DOWNSAMPLING_COEF = 10


def load_ecog():
    X = []
    Y = []

    for filepath in tqdm(FILES_LIST):
        with h5py.File(filepath, "r+") as input_file:
            data = input_file["RawData"]["Samples"][()]

        ecog = data[:, :30].astype("double")
        ecog = scipy.signal.decimate(ecog, DOWNSAMPLING_COEF, axis=0)
        ecog = remove_eyes_artifacts(ecog, int(FREQUENCY / DOWNSAMPLING_COEF))
        ecog = notch_filtering(ecog, int(FREQUENCY / DOWNSAMPLING_COEF))

        sound = data[:, 31].astype("double")
        sound_envelope = extract_mfccs(
            sound=sound, sr=FREQUENCY, i_mel=1, lngt=ecog.shape[0]
        ).T

        X.append(ecog)
        Y.append(sound_envelope)

    X = np.concatenate(X)
    Y = np.concatenate(Y).reshape((-1, 1))

    X = sklearn.preprocessing.scale(X, copy=False)
    Y = sklearn.preprocessing.scale(Y, copy=False)

    X = X.astype("float32")
    Y = Y.astype("float32")
    return X, Y


def load_meg():
    filepath = "/home/altukhov/Data/speech/rawdata/meg.h5"
    with h5py.File(filepath, "r+") as input_file:
        meg = input_file["RawData"]["Samples"][()].astype("double")
        sound = input_file["RawData"]["Audio"][()].astype("double")

    sound_envelope = extract_mfccs(
        sound=sound, sr=FREQUENCY, i_mel=1, lngt=meg.shape[0]
    ).T

    X = meg
    Y = sound_envelope[:, np.newaxis]

    X = sklearn.preprocessing.scale(X, copy=False)
    Y = sklearn.preprocessing.scale(Y, copy=False)

    X = X.astype("float32")
    Y = Y.astype("float32")
    return X, Y


# X, Y = load_ecog()
X, Y = load_meg()
# assert False


# np.save(f'{DATA_FOLDER}/X.npy', X);
# np.save(f'{DATA_FOLDER}/Y.npy', Y);

# X = np.load(f'{DATA_FOLDER}/X.npy');
# Y = np.load(f'{DATA_FOLDER}/Y.npy');

FREQUENCY = int(FREQUENCY / DOWNSAMPLING_COEF)


assert X.shape[0] == Y.shape[0]

print(f"{round(X.shape[0] / FREQUENCY / 60, 1)} min")

ntr = int(X.shape[0] * 0.50)

X_test = X[ntr:]
Y_test = Y[ntr:]

X_train = X[:ntr]
Y_train = Y[:ntr]


class EnvelopeDetector(nn.Module):
    def __init__(self, in_channels):
        super(self.__class__, self).__init__()
        self.FILTERING_SIZE = 75
        self.ENVELOPE_SIZE = 75
        self.conv_filtering = nn.Conv1d(
            in_channels,
            in_channels,
            bias=False,
            kernel_size=self.FILTERING_SIZE,
            groups=in_channels,
        )
        self.conv_envelope = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=self.ENVELOPE_SIZE,
            groups=in_channels,
        )
        self.pre_envelope_batchnorm = torch.nn.BatchNorm1d(
            in_channels, affine=False
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv_filtering(x)
        x = self.pre_envelope_batchnorm(x)
        # x = self.relu(x)
        x = torch.abs(x)
        x = self.conv_envelope(x)
        return x


class SimpleNet(nn.Module):
    def __init__(
        self, in_channels, output_channels, lag_backward, lag_forward
    ):
        super(self.__class__, self).__init__()
        self.ICA_CHANNELS = 20
        self.relu = torch.nn.Sigmoid()

        self.total_input_channels = self.ICA_CHANNELS
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward

        self.decim = 5
        self.final_out_features = self.ICA_CHANNELS * int(
            (lag_backward + lag_forward) / self.decim
        )

        self.ica = nn.Conv1d(in_channels, self.ICA_CHANNELS, 1)

        self.detector = EnvelopeDetector(self.ICA_CHANNELS)
        self.unmixed_batchnorm = torch.nn.BatchNorm1d(
            self.ICA_CHANNELS, affine=False
        )

        self.features_batchnorm = torch.nn.BatchNorm1d(
            self.final_out_features, affine=False
        )

        self.pre_output_features = int(self.final_out_features / 2)

        self.not_yet_output_features = int(self.pre_output_features / 2)

        self.wights_second = nn.Linear(
            self.final_out_features, self.pre_output_features
        )

        self.wights_third = nn.Linear(
            self.pre_output_features, self.not_yet_output_features
        )

        self.wights_fourth = nn.Linear(
            self.not_yet_output_features, output_channels
        )

    def forward(self, inputs):
        all_inputs = self.ica(inputs)
        # self.unmixed_channels = all_inputs.cpu().data.numpy()
        # all_inputs = self.unmixed_batchnorm(all_inputs)

        # print(inputs.shape)
        # print(all_inputs.shape)
        detected_envelopes = self.detector(all_inputs)

        #        features  = detected_envelopes[:, :, ::10].contiguous()
        print(detected_envelopes.shape)
        features = detected_envelopes[:, :, ::self.decim].contiguous()
        # print(f"{features.shape=}")
        features = features.view(features.size(0), -1)
        # features = self.features_batchnorm(features)
        # self.pre_out = features.cpu().data.numpy()
        preoutput = self.wights_second(features)
        notyetoutput = self.wights_third(self.relu(preoutput))
        output = self.wights_fourth(self.relu(notyetoutput))

        return output


# writer = SummaryWriter()

LAG_BACKWARD = 175
LAG_FORWARD = 175


BATCH_SIZE = 100

model = SimpleNet(
    X_train.shape[1], Y_train.shape[1], LAG_BACKWARD, LAG_FORWARD
)

print(
    "Trainable params: ",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)
print(
    "Total params: ",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

train_generator = data_generator(
    X_train,
    Y_train,
    BATCH_SIZE,
    LAG_BACKWARD,
    LAG_FORWARD,
    shuffle=True,
    infinite=True,
)
test_generator = data_generator(
    X_test,
    Y_test,
    BATCH_SIZE,
    LAG_BACKWARD,
    LAG_FORWARD,
    shuffle=True,
    infinite=True,
)

loss_history_train = []
loss_history_test = []
corr_history_train = []
corr_history_test = []


def process_batch(generator, is_train):
    if is_train:
        model.train()
    else:
        model.eval()

    x_batch, y_batch = next(generator)
    assert x_batch.shape[0] == y_batch.shape[0]
    assert y_batch.shape[1] == 1
    # assert x_batch.shape[1] == 30
    #    x_batch = torch.FloatTensor(x_batch).cuda()
    #    y_batch = torch.FloatTensor(y_batch).cuda()
    x_batch = torch.FloatTensor(x_batch)
    y_batch = torch.FloatTensor(y_batch)

    if is_train:
        optimizer.zero_grad()

    y_predicted = model(x_batch)
    loss = loss_function(y_predicted, y_batch)

    if is_train:
        loss.backward()
        optimizer.step()

    assert y_predicted.shape[0] == y_batch.shape[0]
    assert y_predicted.shape[1] == y_batch.shape[1] == 1

    # train_or_test = "train" if is_train else "test"

    correlation = np.corrcoef(
        y_predicted.cpu().detach().numpy(),
        y_batch.cpu().detach().numpy(),
        rowvar=False,
    )[0, 1]
    if not iteration % 250:
        if is_train:
            print("Train")
        else:
            print("Test")
        print(f"{x_batch.shape=}")
        # plt.plot(y_predicted.detach().numpy())
        # plt.plot(y_batch.detach().numpy())
        # plt.legend(["predicted", "batch"])
        # plt.show()

    # writer.add_scalar(f'{train_or_test}/loss', loss.cpu().detach().numpy(), iteration)
    # writer.add_scalar(f'{train_or_test}/correlation', correlation, iteration)

    # if is_train:
    #     loss_history_train.append(correlation)
    # else:
    #     loss_history_test.append(correlation)

    if is_train:
        loss_history_train.append(loss.detach().numpy())
        corr_history_train.append(correlation)
    else:
        loss_history_test.append(loss.detach().numpy())
        corr_history_test.append(correlation)


for iteration in tqdm(range(3000)):
    process_batch(train_generator, True)
    with torch.no_grad():
        process_batch(test_generator, False)

    if iteration % 250 == 0:
        eval_lag = min(1000, iteration)
        print(f"Train loss: {np.mean(loss_history_train[-eval_lag:])}")
        print(f"Test loss: {np.mean(loss_history_test[-eval_lag:])}")
        print(f"Train corr: {np.mean(corr_history_train[-eval_lag:])}")
        print(f"Test corr: {np.mean(corr_history_test[-eval_lag:])}")
        print("#")

    if iteration % 5000 == 0:
        torch.save(model.state_dict(), "speech_net")


Y_predicted = []
Y_batch = []

# model.cuda()

for index, (x_batch, y_batch) in tqdm(
    enumerate(
        data_generator(
            X_test,
            Y_test,
            BATCH_SIZE,
            LAG_BACKWARD,
            LAG_FORWARD,
            shuffle=False,
            infinite=False,
        )
    )
):
    #### Train
    model.eval()

    if index < 1000:
        continue  # skip first samples

    if index > 1400:
        break  # skip all the rest

    x_batch = torch.FloatTensor(x_batch)
    y_predicted = model(x_batch).cpu().data.numpy()
    assert x_batch.shape[0] == y_predicted.shape[0]
    Y_predicted.append(y_predicted)
    Y_batch.append(y_batch)

Y_predicted = np.concatenate(Y_predicted, axis=0)
Y_batch = np.concatenate(Y_batch, axis=0)

print(
    "Correlation   val",
    np.corrcoef(Y_predicted[:, 0], Y_batch[:, 0], rowvar=False)[0, 1],
)

plt.figure(figsize=(16, 3))
plt.plot(Y_predicted, label="predicted", alpha=0.7)
plt.show()
