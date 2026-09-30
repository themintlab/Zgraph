import jax
import jax.numpy as jnp

def marg_naive(x, beta):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    safe_beta = jnp.where(beta == 0.0, 1e-10, beta)
    c = (x - x_max) / safe_beta
    return jnp.squeeze(x_max + beta * jnp.log(jnp.sum(jnp.exp(c), axis=-1, keepdims=True)), axis=-1)

# Let's test 1st, 2nd, and 3rd derivatives with beta=0.0
x = jnp.array([1.0, 2.0, 3.0])

print("Naive 1st:", jax.grad(marg_naive)(x, 0.0))
print("Naive 2nd:", jax.jacobian(jax.grad(marg_naive))(x, 0.0))

