import torch
from torch.optim import Optimizer
import math

class Muon(Optimizer):
    """Muon optimizer - orthogonalized momentum.
    
    Uses Newton-Schulz iteration to orthogonalize gradients before applying momentum.
    Only works for 2D+ parameters (matrices). Use AdamW for 1D params (norms, biases).
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                
                # Only orthogonalize 2D+ params
                if g.ndim >= 2:
                    g = self._newton_schulz(g, ns_steps)
                
                # Momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                # Update
                p.add_(g, alpha=-lr)
    
    def _newton_schulz(self, G, steps=5, eps=1e-7):
        """Approximate G with U @ V.T where U, S, V = G.svd()
        
        This gives us the "direction" of G without magnitude distortion.
        """
        # Newton-Schulz coefficients (optimized for fast convergence)
        a, b, c = (3.4445, -4.7750, 2.0315)
        
        X = G.bfloat16() / (G.norm() + eps)
        
        # Handle non-square matrices
        transposed = False
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True
        
        # Newton-Schulz iterations
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if transposed:
            X = X.T
        
        return X.to(G.dtype)


def get_optimizer(model, config):
    """Split params: Muon for matrices, AdamW for vectors (norms, biases)."""
    
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 1D params (norms, biases) -> AdamW
        # 2D+ params (matrices) -> Muon
        if param.ndim < 2:
            adamw_params.append(param)
        else:
            muon_params.append(param)
    
    optimizers = []
    
    if muon_params:
        optimizers.append(Muon(muon_params, lr=config.learning_rate * 0.1, momentum=0.95))
    
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=config.learning_rate, weight_decay=0.0))
    
    return optimizers