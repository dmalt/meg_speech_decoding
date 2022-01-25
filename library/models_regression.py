import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class SimpleNet(nn.Module):
    DOWNSAMPLING =  10
    HIDDEN_CHANNELS = 30
    FILTERING_SIZE = 25
    ENVELOPE_SIZE = 15
    FC_HIDDEN_FEATURES = 100

    assert FILTERING_SIZE % 2 == 1, "conv weights must be odd"
    assert ENVELOPE_SIZE % 2 == 1, "conv weights must be odd"

    def __init__(self, in_channels, out_channels, lag_backward, lag_forward, use_lstm=True):
        super(self.__class__, self).__init__()
        window_size = lag_backward + lag_forward + 1
        self.final_out_features = (window_size // self.DOWNSAMPLING + 1) * self.HIDDEN_CHANNELS
        self.use_lstm = use_lstm
        
        assert window_size > self.FILTERING_SIZE
        assert window_size > self.ENVELOPE_SIZE

        self.unmixing_layer = nn.Conv1d(in_channels, self.HIDDEN_CHANNELS, 1)
        self.unmixed_channels_batchnorm = torch.nn.BatchNorm1d(self.HIDDEN_CHANNELS, affine=False)
        self.detector = self._create_envelope_detector(self.HIDDEN_CHANNELS)
        self.features_batchnorm = torch.nn.BatchNorm1d(self.final_out_features, affine=False)
        self.lstm = nn.LSTM(self.HIDDEN_CHANNELS, int(self.HIDDEN_CHANNELS / 2), num_layers=1, batch_first=True, bidirectional=True)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.final_out_features, out_channels),
#             nn.Linear(self.final_out_features, self.FC_HIDDEN_FEATURES),
#             nn.ReLU(),
#             nn.Linear(self.FC_HIDDEN_FEATURES, self.FC_HIDDEN_FEATURES),
#             nn.ReLU(),
#             nn.Linear(self.FC_HIDDEN_FEATURES, self.FC_HIDDEN_FEATURES),
#             nn.ReLU(),
#             nn.Linear(self.FC_HIDDEN_FEATURES, out_channels),
        )

    def _create_envelope_detector(self, in_channels):  
        # 1. Learn band pass flter
        # 2. Centering signals
        # 3. Abs
        # 4. low pass to get envelope
        return nn.Sequential(
            nn.Conv1d(in_channels, in_channels,kernel_size=self.FILTERING_SIZE,
                      bias=False,  groups=in_channels,
                      padding=int(self.FILTERING_SIZE / 2)),
            nn.BatchNorm1d(in_channels, affine=False),
            nn.LeakyReLU(-1), # just absolute value
            #nn.PReLU(),
            #LogActivation(),
            nn.Conv1d(in_channels, in_channels, kernel_size=self.ENVELOPE_SIZE,
                      groups=in_channels, padding=int(self.ENVELOPE_SIZE / 2)),
        )

    def forward(self, x):
        self.unmixed_channels = self.unmixing_layer(x)
        unmixed_channels_scaled = self.unmixed_channels_batchnorm(self.unmixed_channels)
        detected_envelopes = self.detector(unmixed_channels_scaled)
        features = detected_envelopes[:, :, ::self.DOWNSAMPLING].contiguous()
        if self.use_lstm:
            features = self.lstm(features.transpose(1, 2))[0].transpose(1, 2)
        features = features.reshape((features.shape[0], -1))
        self.features_scaled = self.features_batchnorm(features)
        output = self.fc_layer(self.features_scaled)
        return output
    
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
            self.conv_layers.append(nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels, growth_rate, kernel_size=filter_size, padding=int(filter_size / 2)),
            ))
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
            self.dense_blocks_list.append(DenseBlock(self.HIDDEN_CHANNELS, self.GROWTH_RATE, self.DENSE_BLOCKS_LAYERS, self.FILTERS_SIZE))
            self.transition_blocks_list.append(self._create_transition_layer(self.dense_blocks_list[-1].out_channels, self.HIDDEN_CHANNELS))

        self.final_out_features = 6250  # KOSTYL

        self.features_batchnorm = torch.nn.BatchNorm1d(self.final_out_features, affine=False)
        self.fc_layer = nn.Linear(self.final_out_features, out_channels) 
        self.lstm = nn.LSTM(50, int(50 / 2), num_layers=1, batch_first=True, bidirectional=True)

    def _create_transition_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool1d(self.TRANSTION_MEAN_POOL_SIZE)
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