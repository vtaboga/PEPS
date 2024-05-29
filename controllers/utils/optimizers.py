import jax
import jax.numpy as jnp
import time

from jax import jit


def projected_gradient_descent(
    loss_func,
    x0,
    lower_bound,
    upper_bound,
    iterations_number,
    line_search_alpha,
    line_search_beta,
    line_search_maximum_iterations,
    eps_grad=0.001,
    logs_file: str = None,
) -> (float, float):

    grad_loss_func = jax.grad(loss_func)

    logs = []

    def log_to_memory(message: str):
        logs.append(message)

    def save_logs_to_file():
        if logs_file is not None:
            with open(logs_file, 'a') as f:
                for log in logs:
                    f.write(log + '\n')

    def project(x):
        return jnp.clip(x, lower_bound, upper_bound)

    @jit
    def backtracking_line_search(x, grad_x) -> jnp.array:
        t = jnp.array(1.0)

        def cond_fun(loop_vars):
            i, t = loop_vars
            x_new = x - t * grad_x
            fx_new = loss_func(x_new)
            fx = loss_func(x)
            max_iter_condition = (i < line_search_maximum_iterations)
            armijo_goldstein_condition = (fx_new > fx - line_search_alpha * t * jnp.dot(grad_x, grad_x))
            return max_iter_condition & armijo_goldstein_condition

        def body_fun(loop_vars):
            i, t = loop_vars
            t = t * line_search_beta
            return i + 1, t

        _, t_final = jax.lax.while_loop(cond_fun, body_fun, (0, t))
        return t_final

    @jit
    def grad_update(x):
        grad_x = grad_loss_func(x)
        step_size = backtracking_line_search(
            x,
            grad_x,
        )
        x = x - step_size * grad_x
        x = project(x)
        return x

    x = x0
    grad_norm = jnp.inf
    grad_diff = jnp.inf
    iteration_counter = 0

    log_to_memory('---- New Call ----')
    start_time = time.time()

    while (iteration_counter < iterations_number) and (grad_diff > eps_grad):
        log_message = f'--- iteration {iteration_counter} action {x} grad norm {jnp.round(grad_norm, 5)}'
        log_to_memory(log_message)
        x_new = grad_update(x)
        new_grad_norm = jnp.linalg.norm(grad_loss_func(x_new), ord=2)
        iteration_counter += 1
        x, grad_norm, prev_grad_norm = x_new, new_grad_norm, grad_norm
        grad_diff = jnp.abs(prev_grad_norm - grad_norm)

    elapsed_time = time.time() - start_time
    log_to_memory(f'Execution Time: {elapsed_time:.2f} seconds')
    save_logs_to_file()

    return x, grad_diff