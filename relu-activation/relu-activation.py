import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    
    ReLU(x) = max(0, x)
    Works element-wise for scalars, lists, and NumPy arrays.
    """

    # Convert input to NumPy array
    x = np.array(x)

    # Apply ReLU
    result = np.maximum(0, x)

    return result


# Example tests
print(relu([-2, -1, 0, 3]))
print(relu(5.0))
print(relu([[-1, 2], [3, -4]]))