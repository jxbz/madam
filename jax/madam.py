import jax.numpy as np
from jax.experimental import optimizers

@optimizers.optimizer
def madam(step_size=0.01, b2=0.999, g_bound=10):
  step_size = optimizers.make_schedule(step_size)
  def init(x0):
    s0 = np.sqrt(np.mean(x0*x0))                        # Initial scale.
    v0 = np.zeros_like(x0)                              # 2nd moment.
    return x0, s0, v0
  def update(i, g, state):
    x, s, v = state
    v = (1 - b2) * np.square(g) + b2 * v                # Update 2nd moment.
    vhat = v / (1 - b2 ** (i + 1))                      # Bias correction.
    g_norm = np.nan_to_num( g / np.sqrt(vhat) )         # Normalise gradient.
    g_norm = np.clip( g_norm, -g_bound, g_bound )       # Bound g.
    x *= np.exp( -step_size(i) * g_norm * np.sign(x) )  # Multiplicative update.
    x = np.clip( x, -s, s)                              # Bound parameters.
    return x, s, v
  def get_params(state):
    x, s, v = state
    return x
  return init, update, get_params