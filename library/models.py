import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from library.type_aliases import ChanBatchTensor, SigBatchTensor

from .config_schema import FeatureExtractorConfig, SimpleNetConfig

log = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    def __init__(self, cfg: FeatureExtractorConfig):
        super(self.__class__, self).__init__()
        assert cfg.filtering_size % 2 == 1, "conv weights must be odd"
        assert cfg.envelope_size % 2 == 1, "envelope size must be odd"
        self.cfg = cfg

        self.unmixing_layer = nn.Conv1d(cfg.in_channels, cfg.hidden_channels, kernel_size=1)
        self.unmixed_channels_batchnorm = nn.BatchNorm1d(cfg.hidden_channels, affine=False)
        self.detector = self._create_envelope_detector(cfg.hidden_channels)

    def _create_envelope_detector(self, nch: int) -> nn.Sequential:
        # 1. Learn band pass flter
        # 2. Centering signals
        # 3. Abs
        # 4. low pass to get envelope
        kern_conv, kern_env = self.cfg.filtering_size, self.cfg.envelope_size
        pad_conv, pad_env = int(self.cfg.filtering_size / 2), int(self.cfg.envelope_size / 2)
        return nn.Sequential(
            nn.Conv1d(nch, nch, kernel_size=kern_conv, bias=False, groups=nch, padding=pad_conv),
            nn.BatchNorm1d(nch, affine=False),
            nn.LeakyReLU(-1),  # just absolute value
            nn.Conv1d(nch, nch, kernel_size=kern_env, groups=nch, padding=pad_env),
        )

    def get_conv_filtering_weights(self) -> npt.NDArray[Any]:
        conv_params = self.detector[0].cpu().parameters()
        return np.squeeze(list(conv_params)[0].detach().numpy())

    def get_spatial(self) -> npt.NDArray[Any]:
        return self.unmixing_layer.weight.cpu().detach().numpy()[:, :, 0].T

    def forward(self, x: SigBatchTensor) -> ChanBatchTensor:
        self._unmixed_channels = self.unmixing_layer(x)
        unmix_scaled = self.unmixed_channels_batchnorm(self._unmixed_channels)
        detected_envelopes = self.detector(unmix_scaled)
        features = detected_envelopes[:, :, :: self.cfg.downsampling].contiguous()
        return features


class SimpleNet(nn.Module):
    def __init__(self, cfg: SimpleNetConfig):
        super(self.__class__, self).__init__()
        self.is_use_lstm = cfg.use_lstm
        ext_cfg = cfg.feature_extractor
        self.feature_extractor = FeatureExtractor(ext_cfg)

        window_size = cfg.lag_backward + cfg.lag_forward + 1
        final_out_features = (window_size // ext_cfg.downsampling + 1) * ext_cfg.hidden_channels

        assert window_size > ext_cfg.filtering_size
        assert window_size > ext_cfg.envelope_size

        self.features_batchnorm = nn.BatchNorm1d(final_out_features, affine=False)
        if cfg.use_lstm:
            assert cfg.hidden_channels % 2 == 0, "hidden channels number must be even"
            hc = ext_cfg.hidden_channels
            self.lstm = nn.LSTM(hc, hc // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.fc_layer = nn.Linear(final_out_features, cfg.out_channels)

    def forward(self, x: SigBatchTensor) -> ChanBatchTensor:
        features = self.feature_extractor(x)
        if self.is_use_lstm:
            features = self.lstm(features.transpose(1, 2))[0].transpose(1, 2)
        features = features.reshape((features.shape[0], -1))
        self.features_scaled = self.features_batchnorm(features)
        output = self.fc_layer(self.features_scaled)
        return output


class SimpleNetConv(nn.Module):
    def __init__(self, cfg: SimpleNetConfig, conv_len=20, pooling_size=4):
        super(self.__class__, self).__init__()
        ext_cfg = cfg.feature_extractor

        window_size = cfg.lag_backward + cfg.lag_forward + 1
        # final_out_features = (window_size // ext_cfg.downsampling + 1) * ext_cfg.hidden_channels
        extractor_n_features_per_branch = window_size // ext_cfg.downsampling + 1
        final_out_features = (
            (extractor_n_features_per_branch - conv_len + 1) // pooling_size
        ) * ext_cfg.hidden_channels

        log.debug(f"{extractor_n_features_per_branch=}")
        log.debug(f"{final_out_features=}")

        assert window_size > ext_cfg.filtering_size
        assert window_size > ext_cfg.envelope_size

        self.feature_extractor = FeatureExtractor(ext_cfg)
        self.conv_block = nn.Sequential(
            self.feature_extractor,
            nn.Conv1d(
                in_channels=ext_cfg.hidden_channels,
                out_channels=ext_cfg.hidden_channels,
                kernel_size=conv_len,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(final_out_features, affine=False),
            nn.Linear(final_out_features, cfg.out_channels),
        )

    def forward(self, x: SigBatchTensor) -> ChanBatchTensor:
        features = self.conv_block(x)
        features = features.reshape((features.shape[0], -1))
        return self.head(features)


class EEGNet(nn.Module):
    """Taken from https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb"""
    def __init__(self):
        super(EEGNet, self).__init__()
        T = 120  # number of time samples
        conv1_length = 64
        f1 = 16

        # Layer 1
        self.conv1 = nn.Conv2d(1, f1, (1, conv1_length), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(f1, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 7, 1)

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1, 4 * 2 * 7)
        x = F.sigmoid(self.fc1(x))
        return x


#################################
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, ngroups, group_size, dropout=0.1):
#         super(self.__class__,self).__init__()

#         base = np.array([2**i - 1 for i in range(ngroups+2)])
#         k = base[2:]
#         p = base[1:-1]
#         nchan = ngroups * group_size

#         self.layers = nn.ModuleList()
#         for i in range(ngroups):
#             conv = nn.Conv1d(in_channels=in_channels, out_channels=group_size, kernel_size=k[i], padding=p[i])
#             self.layers.append(conv)
#         self.relu = nn.ReLU()
#         self.shuffle = nn.Conv1d(in_channels=nchan, out_channels=in_channels, kernel_size=1)

#         self.dropout = nn.Dropout(dropout)
#         self.residual = nn.Conv1d(in_channels=in_channels, out_channels=nchan, kernel_size=1)

#     def forward(self, inputs):

#         res = self.residual(inputs)
#         x = []
#         for layer in self.layers:
#             x.append(layer(inputs))
#         x = torch.cat(x, dim=1)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = x + res
#         x = self.shuffle(x)
#         x = self.relu(x)

#         return x

# class FeatureExtractor(nn.Module):
#     def __init__(self, frame_total, stride):
#         super(self.__class__,self).__init__()

#         k_out = frame_total // stride - 3

#         ngroups = 4
#         group_size = 15
#         nchan = ngroups*group_size

#         self.feature_extractor = nn.ModuleList()

#         self.feature_extractor.append(nn.Conv1d(in_channels=6, out_channels=5, kernel_size=1))

#         self.feature_extractor.append(ConvBlock(5, ngroups, group_size, dropout=0.4))
#         self.feature_extractor.append(ConvBlock(5, ngroups, group_size, dropout=0.4))
#         self.feature_extractor.append(ConvBlock(5, ngroups, group_size, dropout=0.4))

#         self.feature_extractor.append(nn.Conv1d(in_channels=5, out_channels=nchan, kernel_size=1))
#         self.feature_extractor.append(nn.AvgPool1d(kernel_size=88, stride=22))

#         self.feature_extractor.append(nn.Conv1d(in_channels=nchan, out_channels=80, kernel_size=k_out))
#         self.feature_extractor.append(nn.ReLU(inplace=True))

#     def forward(self, inputs):

#         x = inputs
#         for layer in self.feature_extractor:
#             x = layer(x)
#             #print(x.size())

#         return x

# class Encoder(nn.Module):
#     N_MELS = 80

#     def __init__(self, frame_total, stride):
#         super(self.__class__,self).__init__()

#         # (B, )
#         self.feature_extractor = FeatureExtractor(frame_total, stride)

#         self.lstm = nn.LSTM(80, 80, num_layers=1, batch_first=True, bidirectional=False)
#         self.lstm_dropout = nn.Dropout(0.3)

#         self.regressor = nn.Sequential(
#             nn.Linear(80, 80),
#             nn.ReLU(inplace=True),
#             nn.Linear(80, self.N_MELS)
#         )

#     def forward(self, inputs):
#         #inputs = x.transpose(1, 2)

#         # (B, nchannels, eeg_d) -> (B, e_d, s_d)
#         x = self.feature_extractor(inputs)

#         # (B, e_d, s_d) -> (B, s_d, e_d)
#         x = x.transpose(1, 2)

#         self.lstm.flatten_parameters()
#         x, _ = self.lstm(x)

#         x = self.lstm_dropout(x)

#         # (B, s_d, e_d) -> (B, s_d, nmel)
#         outputs = self.regressor(x)

#         return outputs[:, 0, :]


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dense_block_layers, filter_size):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.dense_block_layers = dense_block_layers

        self.conv_layers = nn.ModuleList()
        for _ in range(self.dense_block_layers):
            out_channels = in_channels + growth_rate
            self.conv_layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(in_channels),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels,
                        growth_rate,
                        kernel_size=filter_size,
                        padding=filter_size // 2,
                    ),
                )
            )
            in_channels = out_channels
        self.out_channels = out_channels

    def forward(self, x):
        all_outputs = x
        for i in range(self.dense_block_layers):
            current_out = self.conv_layers[i](all_outputs)
            all_outputs = torch.cat((all_outputs, current_out), 1)
        return all_outputs


class DenseNet(nn.Module):
    DENSE_BLOCKS = 3
    DENSE_BLOCKS_LAYERS = 4
    FILTERS_SIZE = 15
    TRANSTION_MEAN_POOL_SIZE = 2
    HIDDEN_CHANNELS = 50
    GROWTH_RATE = 25

    def __init__(self, in_channels, out_channels, lag_backward, lag_forward):
        super(DenseNet, self).__init__()

        self.dense_blocks_list = nn.ModuleList()
        self.transition_blocks_list = nn.ModuleList()

        self.unmixing_layer = nn.Conv1d(in_channels, self.HIDDEN_CHANNELS, 1)

        for i in range(self.DENSE_BLOCKS):
            self.dense_blocks_list.append(
                DenseBlock(
                    self.HIDDEN_CHANNELS,
                    self.GROWTH_RATE,
                    self.DENSE_BLOCKS_LAYERS,
                    self.FILTERS_SIZE,
                )
            )
            self.transition_blocks_list.append(
                self._create_transition_layer(
                    self.dense_blocks_list[-1].out_channels,
                    self.HIDDEN_CHANNELS,
                )
            )

        final_out_features = 6250  # KOSTYL

        self.features_batchnorm = nn.BatchNorm1d(final_out_features, affine=False)
        self.fc_layer = nn.Linear(final_out_features, out_channels)
        self.lstm = nn.LSTM(50, int(50 / 2), num_layers=1, batch_first=True, bidirectional=True)

    def _create_transition_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(self.TRANSTION_MEAN_POOL_SIZE),
        )

    def forward(self, x):
        unmixed_channels = self.unmixing_layer(x)
        dense_block_out = unmixed_channels
        for i in range(self.DENSE_BLOCKS):
            dense_block_out = self.dense_blocks_list[i](dense_block_out)
            dense_block_out = self.transition_blocks_list[i](dense_block_out)

        features = dense_block_out
        features = self.lstm(features.transpose(1, 2))[0].transpose(1, 2)
        features = features.reshape((features.shape[0], -1))
        features_scaled = self.features_batchnorm(features)
        output = self.fc_layer(features_scaled)
        return output
