import numpy as np
from statsmodels.tsa.statespace import UnobservedComponents

# The smallest thing that is still a state space model: local level.
# A = [[1]], C = [[1]], and one variance per noise source.
mod = UnobservedComponents(y, level=True, stochastic_level=True)
res = mod.fit(disp=False)   # the optimiser runs the filter in its objective

# You can read the model straight off the representation:
A = mod.ssm["transition"]   # (n, n)  the memory
B = mod.ssm["selection"]    # (n, m)  how inputs enter
C = mod.ssm["design"]       # (p, n)  how the state is observed
Q = mod.ssm["state_cov"]    # process noise
R = mod.ssm["obs_cov"]      # measurement noise

# And the filter's own output, at the level of the lowering:
res.filter_results.predicted_state[:, :5]     # x̂⁻ - the value L1 carries
res.filter_results.standardized_forecasts_error  # must be white, else state is wrong
res.llf
