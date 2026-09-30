import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

def marginalize_builtin(x, beta):
    return beta * logsumexp(x / beta, axis=-1)

def marginalize_custom(x, beta):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    safe_beta = jnp.where(beta == 0.0, 1.0, beta)
    z = (x - x_max) / safe_beta
    lse = jnp.log(jnp.sum(jnp.exp(z), axis=-1, keepdims=True))
    res = jnp.where(beta == 0.0,
                    x_max,
                    x_max + safe_beta * lse)
    return jnp.squeeze(res, axis=-1)

x = jnp.array([1.0, 2.0, 3.0])
beta = 1e-4

print("Builtin:", marginalize_builtin(x, beta))
print("Custom:", marginalize_custom(x, beta))

print("\n--- Grad 1 w.r.t x ---")
print("Builtin:", jax.grad(marginalize_builtin, argnums=0)(x, beta))
print("Custom:", jax.grad(marginalize_custom, argnums=0)(x, beta))

print("\n--- Grad 3 w.r.t x ---")
def grad3(f):
    return jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(jax.grad(f, argnums=0)(x, beta)))(x)))(x)

# Let's add try-except to catch NaN or errors
try:
    print("Builtin:", grad3(marginalize_builtin))
except Exception as e:
    print("Builtin Grad3 Error:", e)

try:
    print("Custom:", grad3(marginalize_custom))
except Exception as e:
    print("Custom Grad3 Error:", e)
    
print("\n--- Testing beta=0.0 ---")
try:
    print("Builtin beta=0:", marginalize_builtin(x, 0.0))
except Exception as e:
    print("Builtin beta=0 Error:", e)
    
try:
    print("Custom beta=0:", marginalize_custom(x, 0.0))
except Exception as e:
    print("Custom beta=0 Error:", e)

print("\n--- Grad w.r.t beta (beta=1.0) ---")
b_test = 1.0
print("Builtin:", jax.grad(marginalize_builtin, argnums=1)(x, b_test))
print("Custom:", jax.grad(marginalize_custom, argnums=1)(x, b_test))

print("\n--- Grad w.r.t beta (beta=1e-4) ---")
b_test = 1e-4
print("Builtin:", jax.grad(marginalize_builtin, argnums=1)(x, b_test))
print("Custom:", jax.grad(marginalize_custom, argnums=1)(x, b_test))
