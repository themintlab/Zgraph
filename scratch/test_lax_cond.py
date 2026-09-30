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
        
    # lax.cond requires operands to be identical shapes/types
    return jax.lax.cond(beta == 0.0, true_fn, false_fn, operand=None)

# Does this work well?
x = jnp.array([1.0, 2.0, 3.0])
print(marginalize_cond(x, 0.0))

# Try vmap with different betas
try:
    v_cond = jax.vmap(marginalize_cond, in_axes=(None, 0))(x, jnp.array([0.0, 1.0]))
    print("Vmap cond:", v_cond)
except Exception as e:
    print("Vmap error:", e)

