# Linear Regression Using Normal Equation
import torch
def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)

    XtX   = X_t.T @ X_t # (n, n)
    Xty   = X_t.T @ y_t # (n, 1)
    theta = torch.linalg.inv(XtX) @ Xty # (n, 1)
    
    theta = theta.flatten()

    return torch.round(theta * 1e4) / 1e4
