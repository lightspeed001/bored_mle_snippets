import numpy as np
from scipy.linalg import solve_discrete_are

n, T = 3, 40
A, C = np.diag([.92, .80, .65]), np.ones((1, n))
Q, R = np.eye(n) * 1e-2, np.eye(1) * 1e-1

# The DARE fixed point: the covariance that stops moving.
P_inf = solve_discrete_are(A.T, C.T, Q, R)   # note B -> C.T in this form
K_inf = P_inf @ C.T @ np.linalg.inv(C @ P_inf @ C.T + R)

# L1: run the loop. L3: convolve with the frozen kernel. Same answer.
h = np.array([float(C @ np.linalg.matrix_power(A, k - 1) @ K_inf)
              for k in range(1, T + 1)])

y_conv = np.convolve(innovations, h)[:T]
assert np.allclose(y_conv, y_sequential, atol=1e-10)

# T sequential steps became one convolution. That is the whole of L3, and it is
# the first thing Mamba takes away.
