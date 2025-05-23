"""
Optimizer utilities for training neural networks.

This module provides custom optimization algorithms and utilities, including:
- Proximal Gradient Descent (PGD) optimization steps
- Proximal operators for different regularization terms
"""

import torch

def proximal_gradient_step(parameters, gradients, learning_rate, lambda_reg=0.1):
    """
    Perform one step of Proximal Gradient Descent.
    
    This implements the update rule:
    x_{k+1} = prox_{λg}(x_k - λ∇f(x_k))
    where g is the regularization term (e.g., L1 norm)
    
    Args:
        parameters (list[torch.Tensor]): Model parameters to update
        gradients (list[torch.Tensor]): Corresponding gradients
        learning_rate (float): Learning rate for the step
        lambda_reg (float): Regularization strength
        
    Returns:
        list[torch.Tensor]: Updated parameters after proximal step
    """
    # First perform regular gradient step
    updated_params = []
    for param, grad in zip(parameters, gradients):
        if grad is None:
            updated_params.append(param)
            continue
        # Regular gradient descent step
        updated = param - learning_rate * grad
        # Apply proximal operator (soft thresholding for L1)
        threshold = learning_rate * lambda_reg
        updated = torch.sign(updated) * torch.max(torch.abs(updated) - threshold, torch.zeros_like(updated))
        updated_params.append(updated)
    
    return updated_params

class ProximalOptimizer:
    """
    Optimizer implementing Proximal Gradient Descent.
    
    This optimizer handles both smooth and non-smooth terms in the objective function,
    using proximal operators for the non-smooth components.
    """
    def __init__(self, parameters, lr=0.001, lambda_reg=0.1):
        """
        Initialize the ProximalOptimizer.
        
        Args:
            parameters (iterable): Iterable of parameters to optimize
            lr (float): Learning rate
            lambda_reg (float): Regularization strength
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.state = {}
        
    def zero_grad(self):
        """Clear gradients of all parameters."""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    
    def step(self):
        """
        Perform one optimization step using Proximal Gradient Descent.
        
        This method:
        1. Collects current parameters and their gradients
        2. Performs proximal gradient step
        3. Updates parameters in-place
        """
        # Collect current parameters and gradients
        params = []
        grads = []
        for p in self.parameters:
            params.append(p.data)
            grads.append(p.grad.data if p.grad is not None else None)
        
        # Perform proximal gradient step
        updated_params = proximal_gradient_step(
            params, grads, self.lr, self.lambda_reg
        )
        
        # Update parameters in-place
        for p, new_p in zip(self.parameters, updated_params):
            p.data.copy_(new_p) 