"""VGG encoder, decoder models"""
import torch
import torch.nn as nn


class VGGLayer(nn.Module):
    def __init__(self, nin, nout):
        super(VGGLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, dim, nc=1, multiview=False):
        """
        Dim should be for a single image
        Multiview assumes 2 images concatenated
        """
        super(Encoder, self).__init__()
        self._multiview = multiview
        self.dim = dim
        # 128 x 128
        self.c1 = nn.Sequential(
            VGGLayer(nc, 64),
            VGGLayer(64, 64),
        )
        # 64 x 64
        self.c2 = nn.Sequential(
            VGGLayer(64, 128),
            VGGLayer(128, 128),
        )
        # 32 x 32
        self.c3 = nn.Sequential(
            VGGLayer(128, 256),
            VGGLayer(256, 256),
            VGGLayer(256, 256),
        )
        # 16 x 16
        self.c4 = nn.Sequential(
            VGGLayer(256, 512),
            VGGLayer(512, 512),
            VGGLayer(512, 512),
        )
        # 8 x 8
        self.c5 = nn.Sequential(
            VGGLayer(512, 512),
            VGGLayer(512, 512),
            VGGLayer(512, 512),
        )
        if self._multiview:
            # 8 x 4
            self.c6_multiview = nn.Sequential(
                nn.Conv2d(512, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),  # 5x1
                nn.Conv2d(512, dim, (5, 1), 1),
                nn.BatchNorm2d(dim),
                nn.Tanh(),
            )
            # 5 x 1 for multiview 2 image
        else:
            # 4 x 4
            self.c6 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0), nn.BatchNorm2d(dim), nn.Tanh()
            )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input)  # 128 -> 64
        h2 = self.c2(self.mp(h1))  # 64 -> 32
        h3 = self.c3(self.mp(h2))  # 32 -> 16
        h4 = self.c4(self.mp(h3))  # 16 -> 8
        h5 = self.c5(self.mp(h4))  # 8 -> 4
        if self._multiview:
            h6 = self.c6_multiview(self.mp(h5))
        else:
            h6 = self.c6(self.mp(h5))  # 4 -> 1
        return h6.view(-1, self.dim), [h1, h2, h3, h4, h5]


# split the input into 2 images, upsample them, and then recombine
up = nn.UpsamplingNearest2d(scale_factor=2)


def multiview_upsample(x):
    H = x.shape[-2]
    x1, x2 = x[:, :, : H // 2], x[:, :, H // 2 :]
    x1_ = up(x1)
    x2_ = up(x2)
    final_img = torch.cat([x1_, x2_], dim=-2)
    assert final_img.shape[-2] == 2 * H, final_img.shape
    return final_img


class Decoder(nn.Module):
    def __init__(self, dim, nc=1, multiview=False):
        super(Decoder, self).__init__()
        self._multiview = multiview
        self.dim = dim
        if multiview:
            # 1 x 1 -> 8 x 4
            self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, (8, 4), 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            # 1 x 1 -> 4 x 4
            self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )
        # 8 x 8
        self.upc2 = nn.Sequential(
            VGGLayer(512 * 2, 512), VGGLayer(512, 512), VGGLayer(512, 512)
        )
        # 16 x 16
        self.upc3 = nn.Sequential(
            VGGLayer(512 * 2, 512), VGGLayer(512, 512), VGGLayer(512, 256)
        )
        # 32 x 32
        self.upc4 = nn.Sequential(
            VGGLayer(256 * 2, 256), VGGLayer(256, 256), VGGLayer(256, 128)
        )
        # 64 x 64
        self.upc5 = nn.Sequential(VGGLayer(128 * 2, 128), VGGLayer(128, 64))
        # 128 x 128
        self.upc6 = nn.Sequential(
            VGGLayer(64 * 2, 64), nn.ConvTranspose2d(64, nc, 3, 1, 1), nn.Sigmoid()
        )
        if multiview:
            self.up = multiview_upsample
        else:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))  # 1 -> 4
        up1 = self.up(d1)  # 4 -> 8
        d2 = self.upc2(torch.cat([up1, skip[4]], 1))  # 8 x 8
        up2 = self.up(d2)  # 8 -> 16
        d3 = self.upc3(torch.cat([up2, skip[3]], 1))  # 16 x 16
        up3 = self.up(d3)  # 16 -> 32
        d4 = self.upc4(torch.cat([up3, skip[2]], 1))  # 32 x 32
        up4 = self.up(d4)  # 32 -> 64
        d5 = self.upc5(torch.cat([up4, skip[1]], 1))  # 64 x 64
        up5 = self.up(d5)  # 64 -> 128
        output = self.upc6(torch.cat([up5, skip[0]], 1))  # 128 x 128 or 256 x 128
        return output


def encoder_test():
    import numpy as np
    from torchvision.transforms import ToTensor

    # Singleview Test
    print("Single view Test:")
    img = np.random.normal(size=(2, 128, 128, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    print(tensor.shape)
    enc = Encoder(dim=128, nc=3, multiview=False)
    out = enc(tensor)
    print(out[0].shape)

    # Multiview Test
    print("Multiview Test:")
    img = np.random.normal(size=(2, 256, 128, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    print(tensor.shape)
    enc = Encoder(dim=128, nc=3, multiview=True)
    out = enc(tensor)
    print(out[0].shape)


def decoder_test():
    import numpy as np
    from torchvision.transforms import ToTensor

    print("Single view Test:")
    img = np.random.normal(size=(2, 128, 128, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    enc = Encoder(dim=128, nc=3, multiview=False)
    out = enc(tensor)
    dec = Decoder(dim=128, nc=3, multiview=False)
    out = dec(out)
    print(out.shape)

    print("Multiview Test:")
    img = np.random.normal(size=(2, 256, 128, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    enc = Encoder(dim=128, nc=3, multiview=True)
    out = enc(tensor)
    dec = Decoder(dim=128, nc=3, multiview=True)
    out = dec(out)
    print(out.shape)


if __name__ == "__main__":
    decoder_test()
