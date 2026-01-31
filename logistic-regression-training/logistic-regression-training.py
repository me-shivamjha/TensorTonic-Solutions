import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N,D = X.shape

    w = np.zeros(D)
    b = 0.0

    for i in range(steps):
        # 2. Calculate linear prediction z
        z = np.dot(X, w) + b
        
        # 3. Calculate probability p using sigmoid
        p = _sigmoid(z)
        
        # 4. Calculate gradients (dw, db)
        dw = (1/N) * np.dot(X.T, (p - y))
        db = (1/N) * np.sum(p - y)
        
        # 5. Update weights (w, b)
        w = w - lr * dw
        b = b - lr * db
        
    return w, b

  


    
    pass