import torch
import torch.nn as nn
from torchvision import models

class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        self.resnext = models.resnext50_32x4d(pretrained=True)
        self.resnext.fc = nn.Identity()
        self.lstm = nn.LSTM(2048, 256, batch_first=True)
        self.classifier = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, seq_len, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        features = self.resnext(x)
        features = features.view(batch, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        output = self.classifier(lstm_out[:, -1, :])
        return self.sigmoid(output)
