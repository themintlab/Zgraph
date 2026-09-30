import jax
import jax.numpy as jnp

def marginalize_cond(energy_landscape, beta):
    energy_max = jnp.max(energy_landscape, axis=-1, keepdims=True)
    def true_fn(_):
        return jnp.squeeze(energy_max, axis=-1)
    def false_fn(_):
        centered = (energy_landscape - energy_max) / beta
        sum_exp = jnp.sum(jnp.exp(centered), axis=-1, keepdims=True)
        return jnp.squeeze(energy_max + beta * jnp.log(sum_exp), axis=-1)
    return jax.lax.cond(beta == 0.0, true_fn, false_fn, operand=None)

x = jnp.array([1.0, 2.0, 3.0])
print("Grad x at beta=0:", jax.grad(marginalize_cond, argnums=0)(x, 0.0))
print("Grad beta at beta=0:", jax.grad(marginalize_cond, argnums=1)(x, 0.0))

