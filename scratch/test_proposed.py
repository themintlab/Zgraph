import jax
import jax.numpy as jnp

def proposed_marginalize(energy_landscape, beta=1.0):
    energy_max = jnp.max(energy_landscape, axis=-1, keepdims=True)
    safe_beta = jnp.where(beta == 0.0, 1e-10, beta)
    centered = (energy_landscape - energy_max) / safe_beta
    sum_exp = jnp.sum(jnp.exp(centered), axis=-1, keepdims=True)
    res = energy_max + beta * jnp.log(sum_exp)
    return jnp.squeeze(res, axis=-1)

# Tests
x1 = jnp.array([1.0])
x3 = jnp.array([1.0, 2.0, 3.0])
x_maxes = jnp.array([3.0, 3.0, 1.0])

print("Singleton beta=1:", proposed_marginalize(x1, 1.0))
print("Singleton beta=0:", proposed_marginalize(x1, 0.0))
print("Singleton beta=1e-4:", proposed_marginalize(x1, 1e-4))

print("Multistate beta=1:", proposed_marginalize(x3, 1.0))
print("Multistate beta=0:", proposed_marginalize(x3, 0.0))
print("Multistate beta=1e-4:", proposed_marginalize(x3, 1e-4))

print("Multi-max beta=0:", proposed_marginalize(x_maxes, 0.0))

print("\nGradients w.r.t x (beta=1.0):")
print(jax.grad(proposed_marginalize, argnums=0)(x3, 1.0))

print("\nGradients w.r.t x (beta=0.0):")
print(jax.grad(proposed_marginalize, argnums=0)(x3, 0.0))

print("\nGradients w.r.t x (beta=1e-4):")
print(jax.grad(proposed_marginalize, argnums=0)(x3, 1e-4))

# 3rd order
grad_3 = jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(jax.grad(proposed_marginalize, argnums=0)(x, 1e-4)))(x)))(x3)
print("\nGrad3 w.r.t x (beta=1e-4):", grad_3)

# 3rd order singleton
grad_3_1 = jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(jax.grad(proposed_marginalize, argnums=0)(x, 1e-4)))(x)))(x1)
print("Grad3 w.r.t x singleton (beta=1e-4):", grad_3_1)

