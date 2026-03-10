import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.

    x : input tensor of shape (N, C_in, H, W)
    W : weights of shape (C_out, C_in, KH, KW)
    b : bias of shape (C_out,)
    
    Returns:
    y : output tensor of shape (N, C_out, H_out, W_out)
    """

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape

    # Output dimensions
    H_out = H - KH + 1
    W_out = W_in - KW + 1

    # Initialize output
    y = np.zeros((N, C_out, H_out, W_out))

    # Convolution operation
    for i in range(H_out):
        for j in range(W_out):

            # Extract sliding window
            x_slice = x[:, :, i:i+KH, j:j+KW]   # (N, C_in, KH, KW)

            # Multiply and sum across channels and kernel dims
            y[:, :, i, j] = np.tensordot(
                x_slice, W, axes=([1,2,3], [1,2,3])
            ) + b

    return y