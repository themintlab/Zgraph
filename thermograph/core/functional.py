import torch

def SGTE_polynomial(T, a, b, c, d, e, f):
    """Evaluates a temperature-dependent SGTE polynomial."""
    T_safe = torch.clamp(T, min=1e-6)
    # Polynomial part - Horner's Method
    # Reduces multiplications and prevents catastrophic cancellation at high T
    # P(T) = a + T*(b + T*(d + T*e))
    poly_term = a + T_safe * (b + T_safe * (d + T_safe * e))

    # Transcendental Part
    # c*T*ln(T) + f/T
    log_term = c * T_safe * torch.log(T_safe)
    inv_term = f / T_safe
    return poly_term + log_term + inv_term
