import numpy as np
import collections


class MackeyGlass:
	"""
	Generate the Mackey Glass time-series. Parameters are:
	(adapted from Oger toolbox)
	"""
	def __init__(self, tau, dt, n_samples):
		self.tau, self.dt, self.n_samples = tau, dt, n_samples
		self.rng = np.random.default_rng()

	def generate_timeseries(self, T):
		history_len = int(self.tau * self.dt)

		# Initial conditions for the history of the system
		timeseries = 1.2
		samples = []
		for _ in range(self.n_samples):
			history = collections.deque(1.2 * np.ones(history_len) + 0.2 * (self.rng.random(history_len) - 0.5))
			# Preallocate the array for the time-series
			inp = np.zeros((T, 1))

			for timestep in range(T):
				for _ in range(int(self.dt)):
					xtau = history.popleft()
					history.append(timeseries)
					timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / self.dt
					inp[timestep] = timeseries
			# squash time series...
			samples.append(inp.T)

		return samples


def build_input_and_targets(input_params, key):
	"""
	Build MG input and multiple targets (memory and prediction)
	Returns:
	"""
	mg_tau, task_tau, T, n_steps = input_params
	dt = T / n_steps

	input_ts = MackeyGlass(mg_tau, dt, n_samples=1).generate_timeseries(T)[0]

	tget_indices = np.concatenate((np.arange(-task_tau, 0), np.arange(0, task_tau+1)), axis=0)
	outputs = np.empty((tget_indices.shape[0], T))
	for idx, ii in enumerate(tget_indices):
		if ii < 0:
			outputs[idx, ii:] = input_ts[0, ii:]
		else:
			outputs[idx, :-ii] = input_ts[0, :-ii]

	#
	#
	#
	# outputs[0, :] = input_ts[0, :]
	# for ii in range(task_tau):
	# 	outputs[ii+1, ii+1:] = input_ts[0, ii+1:]