import torch
import torch.nn as nn
import torch.nn.functional as F
from src.prediction.models.cdna import apply_cdna_kernels_torch

RELU_SHIFT = 1e-7

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    name = "encoder_64"
    def __init__(self, dim, nc=1, multiview=False, dropout=None, use_skip=True):
        super(Encoder, self).__init__()
        self.use_skip = use_skip
        self._multiview = multiview
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
            vgg_layer(nc, 64),
            vgg_layer(64, 64),
        )
        # 32 x 32
        self.c2 = nn.Sequential(
            vgg_layer(64, 128),
            vgg_layer(128, 128),
        )
        # 16 x 16
        self.c3 = nn.Sequential(
            vgg_layer(128, 256),
            vgg_layer(256, 256),
            vgg_layer(256, 256),
        )
        # 8 x 8
        self.c4 = nn.Sequential(
            vgg_layer(256, 512),
            vgg_layer(512, 512),
            vgg_layer(512, 512),
        )

        if self._multiview:
            # 8 x 4
            self.c5_multiview = nn.Sequential(
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
            self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0), nn.BatchNorm2d(dim), nn.Tanh()
            )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self._dropout = dropout
        if dropout is not None:
            self._dropout = nn.Dropout2d(p=dropout)

    def forward(self, input):
        dropout = lambda x: x
        if self._dropout is not None:
            dropout = self._dropout
        h1 = dropout(self.c1(input))  # 64 -> 32
        h2 = dropout(self.c2(self.mp(h1)))  # 32 -> 16
        h3 = dropout(self.c3(self.mp(h2)))  # 16 -> 8
        h4 = dropout(self.c4(self.mp(h3)))  # 8 -> 4
        if self._multiview:
            h5 = self.c5_multiview(self.mp(h4))
        else:
            h5 = self.c5(self.mp(h4))  # 4 -> 1
        if self.use_skip:
            return h5.view(-1, self.dim), [h1, h2, h3, h4]
        return h5.view(-1, self.dim), None


class ConvEncoder(nn.Module):
    name = "conv_encoder_64"
    def __init__(self, dim, nc=1):
        """Encoder for convolutional lstm

        Args:
            dim ([type]): number of output channels
            nc (int, optional): number of input channels
        """
        super(ConvEncoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
            vgg_layer(nc, 64),
            vgg_layer(64, 64),
        )
        # 32 x 32
        self.c2 = nn.Sequential(
            vgg_layer(64, 128),
            vgg_layer(128, 128),
        )
        # 16 x 16
        self.c3 = nn.Sequential(
            vgg_layer(128, 256),
            vgg_layer(256, 256),
            vgg_layer(256, 256),
        )
        # 8 x 8
        self.c4 = nn.Sequential(
            vgg_layer(256, 512),
            vgg_layer(512, 512),
            vgg_layer(512, dim)
        )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        """Returns an Cx8x8 feature map where C is g_dim
        """
        h1 = self.c1(input)  # 64
        h2 = self.c2(self.mp(h1))  # 32
        h3 = self.c3(self.mp(h2))  # 16
        h4 = self.c4(self.mp(h3))  # 8
        return h4, [h1, h2, h3, h4]


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
    name = "decoder_64"
    def __init__(self, dim, nc=1, multiview=False, use_skip=True):
        super(Decoder, self).__init__()
        self.use_skip = use_skip
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
            vgg_layer(512 * 2, 512), vgg_layer(512, 512), vgg_layer(512, 256)
        )
        # 16 x 16
        self.upc3 = nn.Sequential(
            vgg_layer(256 * 2, 256), vgg_layer(256, 256), vgg_layer(256, 128)
        )
        # 32 x 32
        self.upc4 = nn.Sequential(vgg_layer(128 * 2, 128), vgg_layer(128, 64))
        # 64 x 64
        self.upc5 = nn.Sequential(
            vgg_layer(64 * 2, 64), nn.ConvTranspose2d(64, nc, 3, 1, 1), nn.Sigmoid()
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        if self.use_skip:
            vec, skip = input
            d1 = self.upc1(vec.view(-1, self.dim, 1, 1))  # 1 -> 4
            up1 = self.up(d1)  # 4 -> 8
            d2 = self.upc2(torch.cat([up1, skip[3]], 1))  # 8 x 8
            up2 = self.up(d2)  # 8 -> 16
            d3 = self.upc3(torch.cat([up2, skip[2]], 1))  # 16 x 16
            up3 = self.up(d3)  # 8 -> 32
            d4 = self.upc4(torch.cat([up3, skip[1]], 1))  # 32 x 32
            up4 = self.up(d4)  # 32 -> 64
            output = self.upc5(torch.cat([up4, skip[0]], 1))  # 64 x 64
        else:
            vec = input
            d1 = self.upc1(vec.view(-1, self.dim, 1, 1))  # 1 -> 4
            up1 = self.up(d1)  # 4 -> 8
            d2 = self.upc2(up1)  # 8 x 8
            up2 = self.up(d2)  # 8 -> 16
            d3 = self.upc3(up2)  # 16 x 16
            up3 = self.up(d3)  # 8 -> 32
            d4 = self.upc4(up3)  # 32 x 32
            up4 = self.up(d4)  # 32 -> 64
            output = self.upc5(up4)  # 64 x 64
        return output


class ConvDecoder(nn.Module):
    name = "conv_decoder_64"
    def __init__(self, dim, nc=1,):
        """Input is C x 8 x 8 feature map from LSTM output

        Args:
            dim ([type]): spatial size of the input feature map
            nc (int, optional): number of output channels
        """
        super(ConvDecoder, self).__init__()
        self.dim = dim
        # dim x 8 x 8 -> 256 x 16 x 16
        self.upc2 = nn.Sequential(
            vgg_layer(dim, 512), vgg_layer(512, 512), vgg_layer(512, 256)
        )
        # 256 x 16 x 16 -> 128 x 32 x 32
        self.upc3 = nn.Sequential(
            vgg_layer(256 * 2, 256), vgg_layer(256, 256), vgg_layer(256, 128)
        )
        # 128 x 32 x 32 -> 64 x 64 x 64
        self.upc4 = nn.Sequential(vgg_layer(128 * 2, 128), vgg_layer(128, 64))
        # 64 x 64 x 64 -> nc x 64 x 64
        self.upc5 = nn.Sequential(
            vgg_layer(64 * 2, 64), nn.ConvTranspose2d(64, nc, 3, 1, 1), nn.Sigmoid()
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        """
        Expects input to be dim x 8 x 8

        Args:
            input ([type]): Tuple of (latent, skip)

        Returns:
            [type]: [description]
        """
        vec, skip = input
        d2 = self.upc2(vec)
        up2 = self.up(d2) # 256 x 16 x 16
        d3 = self.upc3(torch.cat([up2, skip[2]], 1))  # 16 x 16
        up3 = self.up(d3)  # 8 -> 32
        d4 = self.upc4(torch.cat([up3, skip[1]], 1))  # 32 x 32
        up4 = self.up(d4)  # 32 -> 64
        output = self.upc5(torch.cat([up4, skip[0]], 1))  # 64 x 64
        return output


""" CDNA MODULES """
class MaskDecoder(nn.Module):
    name = "mask_decoder_64"
    def __init__(self, dim, nc=1):
        """Input is C x 8 x 8 feature map from LSTM output

        Args:
            dim ([type]): channels of the input feature map
            nc (int, optional): number of output masks
        """
        super(MaskDecoder, self).__init__()
        self.dim = dim
        # dim x 8 x 8 -> 256 x 16 x 16
        self.upc2 = nn.Sequential(
            vgg_layer(dim, 512), vgg_layer(512, 512), vgg_layer(512, 256)
        )
        # 256 x 16 x 16 -> 128 x 32 x 32
        self.upc3 = nn.Sequential(
            vgg_layer(256, 256), vgg_layer(256, 256), vgg_layer(256, 128)
        )
        # 128 x 32 x 32 -> 64 x 64 x 64
        self.upc4 = nn.Sequential(vgg_layer(128, 128), vgg_layer(128, 64))
        # 64 x 64 x 64 -> nc x 64 x 64
        self.upc5 = nn.Sequential(
            vgg_layer(64, 64), nn.ConvTranspose2d(64, nc, 3, 1, 1), nn.InstanceNorm2d(nc), nn.LeakyReLU(0.2, inplace=True)
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        """
        Expects input to be dim x 8 x 8 convlstm latent, and skip connection from encoder,
        Args:
            input ([type]): Tuple of (latent, skip)

        Returns: kernel conv output, mask conv output
            [type]: [description]
        """
        vec = input
        d2 = self.upc2(vec)
        up2 = self.up(d2) # 256 x 16 x 16
        d3 = self.upc3(up2)  # 16 x 16
        up3 = self.up(d3)  # 8 -> 32
        d4 = self.upc4(up3)  # 32 x 32
        up4 = self.up(d4)  # 32 -> 64
        output = self.upc5(up4)  # 64 x 64
        # outputs 26 channels, 13 x 64 x 64 output for predicting CDNA kernel,
        # 13 x 64 x 64 output for predicting mask
        kernel_conv, mask_conv = torch.chunk(output, 2, dim=1)
        kernel_shape = kernel_conv.shape
        # softmax mask weights so they add up to 1
        mask_conv = F.softmax(mask_conv.view(vec.shape[0], -1), dim=1).view(kernel_shape)
        return kernel_conv, mask_conv


class CDNADecoder(nn.Module):
    name = "cdna_decoder_64"
    def __init__(self, channels, cdna_kernel_size):
        """Makes image using the CDNA compositing strategy from the ConvLSTM latent

        Args:
            channels ([type]): spatial size of the input feature map
            nc (int, optional): number of output channels
        """
        super(CDNADecoder, self).__init__()
        self.channels = channels
        image_width = 64
        self.num_flows = num_flows = 13
        self.cdna_kernel_size = cdna_kernel_size
        # outputs 26 channels, 13 x 64 x 64 output for predicting CDNA kernel,
        # 13 x 64 x 64 output for predicting mask
        self.decoder = MaskDecoder(channels, num_flows * 2)

        # takes in a B x 13 x 64 x 64 prediction from the decoder
        # and maps to 13 x flattend kernel size
        self.kernel_mlp = nn.Sequential(
            nn.Linear(image_width ** 2, cdna_kernel_size ** 2)
        )


    def forward(self, prev_image, pred_image_latent, context_image):
        """Applies learned pixel flow to image to get next image

        Args:
            prev_image ([B,3,64,64] image): image to predict next image from
            prev_image_latent ([B,128,8,8] tensor): ConvLSTM latent of predicted image
            prev_image_skip (List[Tensor]): encoder activations of prev_image
            context_image ([B, 3, 64, 64]): last ground truth image

        Returns:
            Tensor: [B, 3, 64, 64] the predicted images
        """
        mask_conv, kernel_conv = self.decoder(pred_image_latent)
        # mask_conv, kernel_conv is B x num_flows x 64 x 64
        # reshape kernel conv to B x num_flows x 4096 for kernel prediction
        if len(kernel_conv.shape) != 4:
            import ipdb; ipdb.set_trace()
        B, num_flows, height, width = kernel_conv.shape
        kernel_conv = kernel_conv.view(B, num_flows, -1)
        # B x  num_flows x 4096 -> B x num_flows x (kernel size)**2
        kernels = F.relu(self.kernel_mlp(kernel_conv - RELU_SHIFT)) + RELU_SHIFT
        kernels = kernels.permute([0, 2, 1]) # make num_flows last dim
        # normalize the kernels so it adds up to 1.
        # the logic is that we can only move at most all the existing pixels.
        kernels /= kernels.sum(dim=1, keepdim=True)
        # Now make it B x kernel_size x kernel_size x num_flows
        kernels = kernels.view(B, self.cdna_kernel_size, self.cdna_kernel_size, num_flows)

        # first get warped images from the context images using the first kernel
        # must convert images to B H W C for tensorflow logic
        if len(context_image.shape) != 4:
            import ipdb; ipdb.set_trace()
        context_image = context_image.permute(0,2,3,1)
        # B x 64 x 64 x 1 x 3
        warped_context_img = apply_cdna_kernels_torch(context_image, kernels[:, :, :, :1])

        # then get warped images from the previous image using remaining kernels
        # must convert images to BHWC for tensorflow logic
        prev_image = prev_image.permute(0,2,3,1)
        # B x 64 x 64 x 11 x 3
        warped_prev_img = apply_cdna_kernels_torch(prev_image, kernels[:, :, :, 1:])
        # B x 64 x 64 x 13 x 3
        warped_imgs = torch.cat([warped_context_img, warped_prev_img], -2)
        # B x 13 x 64 x 64 x 3
        warped_imgs = warped_imgs.permute(0, 3, 1, 2, 4)

        # Combine with the B x 13 x 64 x 64 x 1 masks
        masks = mask_conv.unsqueeze(-1)
        # B x 13 x 64 x 64 x 3
        weighted_imgs = masks * warped_imgs
        # B x  64 x 64 x 3
        composite_imgs = weighted_imgs.sum(1)
        # B x 3 x 64 x 64
        return composite_imgs.permute(0, 3, 1, 2)


def encoder_test():
    import numpy as np
    from torchvision.transforms import ToTensor

    print("Encoder Tests")
    # Singleview Test
    print("Single view Test:")
    img = np.random.normal(size=(2, 64, 64, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    print(tensor.shape)
    enc = Encoder(dim=128, nc=3, multiview=False)
    out = enc(tensor)
    print(out[0].shape)

    # Multiview Test
    print("Multiview Test:")
    img = np.random.normal(size=(2, 128, 64, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    print(tensor.shape)
    enc = Encoder(dim=128, nc=3, multiview=True)
    out = enc(tensor)
    print(out[0].shape)



def conv_encoder_test():
    import numpy as np
    from torchvision.transforms import ToTensor

    print("Conv Encoder Tests")
    # Singleview Test
    print("Single view Test:")
    img = np.random.normal(size=(2, 64, 64, 4)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    print(tensor.shape)
    enc = ConvEncoder(dim=128, nc=4)
    out = enc(tensor)
    print(out[0].shape)


def decoder_test():
    import numpy as np
    from torchvision.transforms import ToTensor

    print("Decoder Tests")
    print("Single view Test:")
    img = np.random.normal(size=(2, 64, 64, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    enc = Encoder(dim=128, nc=3, multiview=False)
    out = enc(tensor)
    dec = Decoder(dim=128, nc=3, multiview=False)
    out = dec(out)
    print(out.shape)

    print("Multiview Test:")
    img = np.random.normal(size=(2, 128, 64, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    enc = Encoder(dim=128, nc=3, multiview=True)
    out = enc(tensor)
    dec = Decoder(dim=128, nc=3, multiview=True)
    out = dec(out)
    print(out.shape)


def conv_decoder_test():
    import numpy as np
    from torchvision.transforms import ToTensor

    print("Conv Decoder Tests")
    print("Single view Test:")
    img = np.random.normal(size=(2, 64, 64, 3)).astype(np.float32)
    tensor = torch.stack([ToTensor()(i) for i in img])
    enc = ConvEncoder(dim=128, nc=3)
    out = enc(tensor)
    ipdb.set_trace()
    dec = ConvDecoder(dim=128, nc=3)
    out = dec(out)
    print(out.shape)

def cdna_decoder_test():
    import numpy as np
    from torchvision.transforms import ToTensor

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("CDNA Tests")
    print("Single view Test:")
    # img = np.random.normal(size=(2, 64, 64, 3)).astype(np.float32)
    # tensor = torch.stack([ToTensor()(i) for i in img])
    batch_size = 7
    prev_image = torch.from_numpy(np.random.normal(size=(batch_size, 3, 64, 64)).astype(np.float32)).to(device)
    curr_image_latent = torch.from_numpy(np.random.normal(size=(batch_size, 128, 8, 8)).astype(np.float32)).to(device)
    prev_image_skip = [
        torch.from_numpy(np.random.normal(size=(batch_size, 64, 64, 64)).astype(np.float32)).to(device),
        torch.from_numpy(np.random.normal(size=(batch_size, 128, 32, 32)).astype(np.float32)).to(device),
        torch.from_numpy(np.random.normal(size=(batch_size, 256, 16, 16)).astype(np.float32)).to(device),
        torch.from_numpy(np.random.normal(size=(batch_size, 128, 8, 8)).astype(np.float32)).to(device)
    ]
    context_image =torch.from_numpy(np.random.normal(size=(batch_size, 3, 64, 64)).astype(np.float32)).to(device)
    dec = CDNADecoder(dim=128).to(device)
    out = dec(prev_image, curr_image_latent, prev_image_skip, context_image)
    print(out.shape)

if __name__ == "__main__":
    import ipdb
    # encoder_test()
    # decoder_test()
    # conv_encoder_test()
    # conv_decoder_test()
    cdna_decoder_test()