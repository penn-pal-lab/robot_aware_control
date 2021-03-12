
import numpy as np
import torch

def apply_cdna_kernels_torch(image, kernels):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns: A 5-D tensor of shape
            `[batch, in_height, in_width, num_transformed_images, in_channels]`.
    """
    batch_size, height, width, color_channels = image.shape
    batch_size, kernel_height, kernel_width, num_transformed_images = kernels.shape
    kernel_size = [kernel_height, kernel_width]

    # _, image_padded = pad_image(kernels, image)
    # image_padded = torch.from_numpy(image_padded)
    image_padded = torch.from_numpy(image)
    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    kernels = torch.from_numpy(kernels)
    kernels = kernels.permute([1, 2, 0, 3])
    kernels = kernels.reshape([kernel_size[0], kernel_size[1], batch_size, num_transformed_images])

    # convert BCHW to CHWB tensor
    # Swap the batch and channel dimensions.
    image_transposed = image_padded.permute([3,1,2,0])
    # Transform image.
    outputs = cnn2d_depthwise_torch(image_transposed, kernels)
    # outputs = tf.nn.depthwise_conv2d(image_transposed, kernels, [1, 1, 1, 1], padding='VALID', rate=dilation_rate)
    # Transpose the dimensions to where they belong.
    outputs = outputs.reshape([color_channels, height, width, batch_size, num_transformed_images])
    outputs = outputs.permute([3, 1, 2, 4, 0])
    return outputs


def apply_cdna_kernels_tf(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    tf.enable_eager_execution()
    image = tf.convert_to_tensor(image, np.float32)
    kernels = tf.convert_to_tensor(kernels, np.float32)


    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='CONSTANT')
    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    kernels = tf.transpose(kernels, [1, 2, 0, 3])
    kernels = tf.reshape(kernels, [kernel_size[0], kernel_size[1], batch_size, num_transformed_images])
    # Swap the batch and channel dimensions.
    image_transposed = tf.transpose(image_padded, [3, 1, 2, 0])
    # Transform image.
    outputs = tf.nn.depthwise_conv2d(image_transposed, kernels, [1, 1, 1, 1], padding='VALID', rate=dilation_rate)
    # Transpose the dimensions to where they belong.
    outputs = tf.reshape(outputs, [color_channels, height, width, batch_size, num_transformed_images])
    outputs = tf.transpose(outputs, [3, 1, 2, 4, 0])

    return outputs


def cnn2d_depthwise_torch(image: np.ndarray,
                          filters: np.ndarray):

    from torch.nn import functional as F

    image_torch, filters_torch = convert_to_torch(image, filters)

    df, _, cin, cmul = filters.shape
    filters_torch = filters_torch.transpose(0, 1).contiguous()
    filters_torch = filters_torch.view(cin * cmul, 1, df, df)

    features_torch = F.conv2d(image_torch, filters_torch, padding=df // 2, groups=cin)
    features_torch = features_torch.permute([0, 3, 2, 1])
    return features_torch


def cnn2d_depthwise_tf(image: np.ndarray,
                       filters: np.ndarray):

    import tensorflow as tf
    tf.enable_eager_execution()

    features_tf = tf.nn.depthwise_conv2d(image, filters, strides=[1, 1, 1, 1], padding='SAME')

    return features_tf

def pad2d(inputs, size, strides=(1, 1), rate=(1, 1), padding='SAME', mode='CONSTANT'):
    """
    Pads a 4-D tensor according to the convolution padding algorithm.

    Convolution with a padding scheme
        conv2d(..., padding=padding)
    is equivalent to zero-padding of the input with such scheme, followed by
    convolution with 'VALID' padding
        padded = pad2d(..., padding=padding, mode='CONSTANT')
        conv2d(padded, ..., padding='VALID')

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        padding: A string, either 'VALID', 'SAME', or 'FULL'. The padding algorithm.
        mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).

    Returns:
        A 4-D tensor.

    Reference:
        https://www.tensorflow.org/api_guides/python/nn#convolution
    """
    paddings = pad2d_paddings(inputs, size, strides=strides, rate=rate, padding=padding)
    if paddings == [[0, 0]] * 4:
        outputs = inputs
    else:
        outputs = tf.pad(inputs, paddings, mode=mode)
    return outputs


def pad2d_paddings(inputs, size, strides=(1, 1), rate=(1, 1), padding='SAME'):
    """
    Computes the paddings for a 4-D tensor according to the convolution padding algorithm.

    See pad2d.

    Reference:
        https://www.tensorflow.org/api_guides/python/nn#convolution
        https://www.tensorflow.org/api_docs/python/tf/nn/with_space_to_batch
    """
    size = np.array(size) if isinstance(size, (tuple, list)) else np.array([size] * 2)
    strides = np.array(strides) if isinstance(strides, (tuple, list)) else np.array([strides] * 2)
    rate = np.array(rate) if isinstance(rate, (tuple, list)) else np.array([rate] * 2)
    if np.any(strides > 1) and np.any(rate > 1):
        raise ValueError("strides > 1 not supported in conjunction with rate > 1")
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 4
    input_size = np.array(input_shape[1:3])
    if padding in ('SAME', 'FULL'):
        if np.any(rate > 1):
            # We have two padding contributions. The first is used for converting "SAME"
            # to "VALID". The second is required so that the height and width of the
            # zero-padded value tensor are multiples of rate.

            # Spatial dimensions of the filters and the upsampled filters in which we
            # introduce (rate - 1) zeros between consecutive filter values.
            dilated_size = size + (size - 1) * (rate - 1)
            pad = dilated_size - 1
        else:
            pad = np.where(input_size % strides == 0,
                           np.maximum(size - strides, 0),
                           np.maximum(size - (input_size % strides), 0))
        if padding == 'SAME':
            # When full_padding_shape is odd, we pad more at end, following the same
            # convention as conv2d.
            pad_start = pad // 2
            pad_end = pad - pad_start
        else:
            pad_start = pad
            pad_end = pad
        if np.any(rate > 1):
            # More padding so that rate divides the height and width of the input.
            # TODO: not sure if this is correct when padding == 'FULL'
            orig_pad_end = pad_end
            full_input_size = input_size + pad_start + orig_pad_end
            pad_end_extra = (rate - full_input_size % rate) % rate
            pad_end = orig_pad_end + pad_end_extra
        paddings = [[0, 0],
                    [pad_start[0], pad_end[0]],
                    [pad_start[1], pad_end[1]],
                    [0, 0]]
    elif padding == 'VALID':
        paddings = [[0, 0]] * 4
    else:
        raise ValueError("Invalid padding scheme %s" % padding)
    return paddings


def convert_to_torch(image, filters):
    # BHWC -> BCWH
    image_torch = image.permute([0, 3, 2, 1])
    filters_torch = filters.permute([3, 2, 1, 0])

    return image_torch, filters_torch


def pad_image(filters, image):

    assert filters.shape[1] == filters.shape[2]
    assert filters.shape[1] % 2

    filter_len = filters.shape[1]
    pad = filter_len // 2

    image_padded = np.pad(image, [(0, ), (pad,), (pad,), (0,)], 'constant')

    return filter_len, image_padded


def cnn2d_depthwise(image: np.ndarray,
                    filters: np.ndarray):
    """
    Depthwise convolutions.
    Args:
        image: (N, hi, wi, cin).
        filters: (hf, wf, cin, cmul).
    Returns:
        (hi, wi, cin * cmul)
    """

    filter_len, image_padded = pad_image(filters, image)

    n_cout_cin = filters.shape[-1]

    out = np.zeros([image.shape[0], image.shape[1], image.shape[2], image.shape[3] * n_cout_cin])
    for idx_batch in range(image.shape[0]):
        for idx_cin in range(image.shape[3]):
            for idx_row in range(image.shape[1]):
                for idx_col in range(image.shape[2]):
                    patch = image_padded[idx_batch, idx_row: idx_row + filter_len, idx_col: idx_col + filter_len, idx_cin]
                    for idx_cmul in range(n_cout_cin):
                        filter = filters[:, :, idx_cin, idx_cmul]
                        out[idx_batch, idx_row, idx_col, idx_cin * n_cout_cin + idx_cmul] = (filter * patch).sum()

    return out
if __name__ == '__main__':
    import ipdb

    H_IMG = 64  # Image height
    W_IMG = 64  # Image width
    CIN = 3  # Number input channels
    FILTER_LEN = 5  # Length of one side of filter
    CMUL = 13  # Channel multiplier (depthwise cnns)
    BATCH = 8
    # Create inputs
    image = np.random.rand(BATCH, H_IMG, W_IMG, CIN)
    filters = np.random.rand(BATCH, FILTER_LEN, FILTER_LEN, CMUL)
    # features_np = cnn2d_depthwise(image, filters)
    torch_cdna = apply_cdna_kernels_torch(image, filters).numpy()
    # ipdb.set_trace()
    import tensorflow as tf
    tf_cdna = apply_cdna_kernels_tf(image, filters).numpy()
    print(np.allclose(tf_cdna, torch_cdna))