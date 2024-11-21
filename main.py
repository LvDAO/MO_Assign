from optimizer import *
from manifold import *

import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)


# Initialize parameters
rdim = 1000
kdim = 10
max_iter = 200

eye_mat = torch.eye(rdim)[:, :kdim]


# Create a simple test function
def test_function(x):
    return torch.norm(x - eye_mat, p=4) ** 4


# Create Stiefel manifold
manifold = StiefelManifold(rdim, kdim)

# Generate random initial point and orthogonalize
start_point = torch.nn.init.orthogonal_(torch.empty(rdim, kdim))

# Test gradient descent
print("\nTesting Gradient Descent:")
gd = GradientDescent(manifold)
result_gd = gd.optimize(start_point, test_function, max_iter)
print(f"Final function value: {test_function(result_gd):.10f}")

# Test BB method
print("\nTesting BB Method:")
bb = BBMethod(manifold)
result_bb = bb.optimize(start_point, test_function, max_iter)
print(f"Final function value: {test_function(result_bb):.10f}")

# Test regularized Newton method
print("\nTesting Regularized Newton Method:")
rn = RegularizedNewton(manifold)
result_rn = rn.optimize(start_point, test_function, max_iter)
print(f"Final function value: {test_function(result_rn):.10f}")

# Plot convergence curves
plt.figure(figsize=(10, 6))
plt.semilogy(
    gd.log_data["iteration"], gd.log_data["function_value"], label="Gradient Descent"
)
plt.semilogy(bb.log_data["iteration"], bb.log_data["function_value"], label="BB Method")
plt.semilogy(
    rn.log_data["iteration"], rn.log_data["function_value"], label="Regularized Newton"
)
plt.xlabel("Iterations")
plt.ylabel("Function Value (log scale)")
plt.title("Optimization Methods Convergence Comparison")
plt.legend()
plt.grid(True)
plt.savefig('optimization_comparison.png')
plt.close()
