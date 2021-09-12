import torch
import torch.nn as nn
import torch.nn.functional as F

class Mel2WordSimple(nn.Module):
    MEL_PRE_DONSAMPLING = 10
    MELS_CONV_SIZE = 200
    HIDDEN_CHANNELS = 100
    MEL_POST_DONSAMPLING = 10
    
    def __init__(self, in_channels, out_channels):
        super(self.__class__,self).__init__()

        self.mels2features = nn.Sequential(
            nn.Conv2d(1, self.HIDDEN_CHANNELS, kernel_size=(in_channels, 10)),
        )        
        self.max_pool = torch.nn.MaxPool1d(self.MEL_POST_DONSAMPLING)
        
        self.lstm = nn.LSTM(self.HIDDEN_CHANNELS, self.HIDDEN_CHANNELS, num_layers=1, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(200, out_channels),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x[:, :, :, ::self.MEL_PRE_DONSAMPLING]
        mels_features = self.mels2features(x)
        mels_features = mels_features.squeeze(2)
        mels_features = self.max_pool(mels_features)
        mels_features = mels_features.transpose(1, 2)
        lstm_out, (lstm_hidden_state_h, lstm_hidden_state_c) = self.lstm(mels_features)
        lstm_hidden_state_h = lstm_hidden_state_h.transpose(0, 1)
        features = lstm_hidden_state_h.reshape((lstm_hidden_state_h.size(0), -1)) # flatten
        output = self.fc_layer(features)
        return output
