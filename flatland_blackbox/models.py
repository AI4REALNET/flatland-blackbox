import numpy as np
import torch


class EdgeWeightParam(torch.nn.Module):
    """Module that holds a learnable 1D parameter vector for edge weights.

    Each element corresponds to a multiplicative scalar for an edge.
    """

    def __init__(self, num_edges):
        """Initializes the EdgeWeightParam module.

        Args:
            num_edges (int): The number of edges (i.e. the length of the parameter vector).
        """
        super().__init__()
        # Initialize all edge weights as learnable parameters
        self.edge_weights = torch.nn.Parameter(torch.ones(num_edges))

    def forward(self):
        """Returns the current edge weight parameters.

        Returns:
            torch.Tensor: The 1D tensor of edge weights.
        """
        return self.edge_weights


class DifferentiableSolver(torch.autograd.Function):
    """A differentiable wrapper for a solver function.

    This autograd function runs a solver in the forward pass and then uses a finite difference
    approximation in the backward pass to compute gradients with respect to the solver output.
    """

    @staticmethod
    def forward(ctx, w_tensor, solver_fn, lambda_val):
        """Performs the forward pass of the differentiable solver.

        The solver function is called with the numpy array of weights, and its output is stored
        for the backward pass.

        Args:
            w_tensor (torch.Tensor): The input weight tensor.
            solver_fn (callable): A function that takes a numpy array and returns a numpy array
                                  representing the plan usage.
            lambda_val (float): A scaling parameter used in the backward pass for finite differences.

        Returns:
            torch.Tensor: A tensor representing the plan usage computed by the solver.
        """
        with torch.no_grad():
            w_np = w_tensor.detach().cpu().numpy()
            plan_np = solver_fn(w_np)  # shape e.g. [num_edges] with 0/1 usage
            ctx.solver_fn = solver_fn
            ctx.lambda_val = lambda_val
            ctx.save_for_backward(w_tensor, torch.from_numpy(plan_np).float())
        return torch.from_numpy(plan_np).float().to(w_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        """Performs the backward pass to compute gradients with respect to w_tensor.

        It perturbs the weight tensor by a small amount (lambda_val * grad_output) and
        approximates the gradient using finite differences.

        Args:
            grad_output (torch.Tensor): The gradient of the loss with respect to the output of the forward pass.

        Returns:
            tuple: A tuple (grad_w, None, None) where grad_w is the gradient with respect to w_tensor.
                   The gradients for solver_fn and lambda_val are set to None.
        """
        w_tensor, plan_tensor = ctx.saved_tensors
        solver_fn = ctx.solver_fn
        lambda_val = ctx.lambda_val

        w_np = w_tensor.detach().cpu().numpy()
        plan_np = plan_tensor.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()

        # Perturb the weights.
        w_perturbed = np.maximum(w_np + lambda_val * grad_output_np, 0)

        # Debug prints: print original and perturbed weights.
        # print("DEBUG: Original weights:", w_np)
        # print("DEBUG: grad_output_np:", grad_output_np)
        # print("DEBUG: Perturbed weights:", w_perturbed)

        plan_perturbed_np = solver_fn(w_perturbed)

        # Compute finite difference gradient.
        gradient_np = -(plan_np - plan_perturbed_np) / lambda_val
        grad_w = torch.from_numpy(gradient_np).to(w_tensor.device).float()

        return grad_w, None, None
