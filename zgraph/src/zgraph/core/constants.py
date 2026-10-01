"""
Centralized numerical constants and limits for the ZGraph engine.
"""

# A sufficiently high energy value representing an infinite potential barrier (Tropical Zero).
# Used to dynamically mask out forbidden microstates or unreachable phases.
TROPICAL_ZERO_BARRIER = 1e9

# The absolute minimum allowable beta parameter to prevent division by zero
# during explicit logsumexp formulations (T -> 0 hardmax limit).
MIN_SAFE_BETA = 1e-10

# The default minimum beta applied as a floor in Soft-Min-Sum tensor contractions
# (TropicalMatMulNode) to guarantee gradient stability.
DEFAULT_MATMUL_MIN_BETA = 1e-4
