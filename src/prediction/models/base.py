import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    name="mlp_encoder"
    """
    Simple 1 layer MLP for the action encoding
    """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, input):
        return self.output(input)


def init_weights(m):
    cn = m.__class__.__name__
    if cn.find("ConvDecoder") != -1 or cn.find("ConvEncoder") != -1 or cn.find("ConvLSTM") != -1:
        return
    if cn.find("Conv") != -1 or cn.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif cn.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        ic = config.channels
        if config.model_use_mask:
            # RGB + mask channel
            ic += 1

        self._l1 = nn.Sequential(
            nn.Conv2d(ic, 16, 5,padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self._img_layers = nn.Sequential(
            self._l1,
            nn.Conv2d(16, 3, 3,padding=1),
            nn.Sigmoid()
        )

    def forward(self, images, masks):
        frames = images
        if self._config.model_use_mask:
            frames = torch.cat([images, masks], 1)
        return self._img_layers(frames)

if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    atn = Attention(config)
    images = torch.ones((4, 3, 64, 64))
    masks = torch.ones((4, 1, 64, 64))
    masks = atn(images, masks)
    import ipdb; ipdb.set_trace()