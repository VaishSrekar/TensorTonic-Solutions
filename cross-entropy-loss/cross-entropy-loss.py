import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    
    Parameters:
    y_true : array-like of shape (N,)
        True class labels.
    y_pred : array-like of shape (N, K)
        Predicted probabilities for each class.
        
    Returns:
    float
        Average cross-entropy loss.
    """
    
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Number of samples
    N = y_true.shape[0]
    
    # Extract predicted probability of the correct class
    correct_class_probs = y_pred[np.arange(N), y_true]
    
    # Compute cross-entropy loss
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss


# Example tests
y_true = [0, 1]
y_pred = [[0.9, 0.1], [0.3, 0.7]]
print(cross_entropy_loss(y_true, y_pred))  # Output: 0.231018

y_true = [2]
y_pred = [[0.1, 0.1, 0.8]]
print(cross_entropy_loss(y_true, y_pred))  # Output: 0.223144

y_true = [1, 0, 1]
y_pred = [[0.2, 0.8], [0.6, 0.4], [0.49, 0.51]]
print(cross_entropy_loss(y_true, y_pred))  # Output: 0.469105