import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

def builtin(x, beta):
    return beta * logsumexp(x / beta, axis=-1)

def builtin_with_bypass(x, beta):
    if x.shape[-1] == 1:
        return jnp.squeeze(x, axis=-1)
    return beta * logsumexp(x / beta, axis=-1)

x = jnp.array([1.0])
beta = 1e-4

g1 = jax.grad(lambda x: jnp.sum(builtin(x, beta)))(x)
print("G1 builtin:", g1)

# we need to test up to 3rd derivative
try:
    g3 = jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(builtin(x, beta)))(x)))(x)))(x)
    print("G3 builtin:", g3)
except Exception as e:
    print("G3 builtin error:", e)

