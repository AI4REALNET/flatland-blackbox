import numpy as np
import torch


class EdgeWeightParam(torch.nn.Module):
    """
    Holds a 1D parameter vector for edge weights, one float per edge.
    """

    def __init__(self, num_edges):
        super().__init__()
        # Initialize all edge weights as learnable parameters
        self.edge_weights = torch.nn.Parameter(torch.ones(num_edges))

    def forward(self):
        return self.edge_weights


class DifferentiableSolver(torch.autograd.Function):
    """
    forward => solver(w)
    backward => solver(w + lambda * grad).
    """

    @staticmethod
    def forward(ctx, w_tensor, solver_fn, lambda_val):
        with torch.no_grad():
            w_np = w_tensor.detach().cpu().numpy()
            plan_np = solver_fn(w_np)  # shape e.g. [num_edges] with 0/1 usage
            ctx.solver_fn = solver_fn
            ctx.lambda_val = lambda_val
            ctx.save_for_backward(w_tensor, torch.from_numpy(plan_np).float())
        return torch.from_numpy(plan_np).float().to(w_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        (w_tensor, plan_tensor) = ctx.saved_tensors
        solver_fn = ctx.solver_fn
        lambda_val = ctx.lambda_val

        w_np = w_tensor.detach().cpu().numpy()
        plan_np = plan_tensor.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()

        # w_perturbed = w + lambda * dL/d(plan)
        w_perturbed = np.maximum(w_np + lambda_val * grad_output_np, 1e-6)
        plan_perturbed_np = solver_fn(w_perturbed)
        # finite diff
        gradient_np = -(plan_np - plan_perturbed_np) / lambda_val
        grad_w = torch.from_numpy(gradient_np).to(w_tensor.device).float()

        return grad_w, None, None
