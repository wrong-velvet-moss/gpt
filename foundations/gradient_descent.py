class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        """
        Every neutral network from a simple linear model learns by gradient descent.

        Let's build the core optimisation algorithm from scratch.

        Minimise the function f(x) = x^2 using gradient descent. The
        minimum is obviously x = 0, but you must arrive at it 
        iteratively, the same way a neural network adjusts
        millions of parameters during training.

        The derivative of f(x) = x^2 is f'(x) = 2x. At each step,
        update x using the gradient descent rule.
        x_new = x_old - alpha f'(x_old),
        where a is the learning rate, a small constant
        that controls the step size.
        """
        x = init
        for _ in range(iterations):
            derivative = 2 * x
            x = x - learning_rate * derivative
        return round(x,5)
