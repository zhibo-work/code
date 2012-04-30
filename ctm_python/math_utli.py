import numpy as np

def log_sum(log_a, log_b):
	v = 0
	if log_a == -1:
		return log_b
	if log_a < log_b:
		v = log_b + np.log(1 + np.exp(log_a - log_b))
	else:
		v = log_a + np.log(1 + np.exp(log_b - log_a))
	return v


def safe_log(x):
	if x == 0.0:
		return -1000
	else:
		return np.log(x)