import jax
import jax.numpy as jnp

boundary_idx = jnp.array(50) # Scalar
k_needed = 2
boundary_idx_expanded = jnp.broadcast_to(jnp.expand_dims(boundary_idx, (-1, -2)), (*boundary_idx.shape, 1, k_needed))
print("Expanded shape:", boundary_idx_expanded.shape)

top_indices = jnp.zeros((100, 2), dtype=jnp.int32)
res = jnp.take_along_axis(top_indices, boundary_idx_expanded, axis=-2)
print("Res shape:", res.shape)
try:
    print("Squeeze shape:", res.squeeze(-2).shape)
except Exception as e:
    print("Error:", e)

