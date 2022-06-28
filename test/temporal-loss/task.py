import numpy as np
import jax.numpy as jnp
import collections
import matplotlib.pyplot as plt


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


def build_input_and_targets(input_params):
	"""
	Build MG input and multiple targets (memory and prediction)
	Returns:
	"""
	mg_tau, task_tau, T, n_steps, batch_size = input_params
	dt = T / n_steps

	input_ts = MackeyGlass(mg_tau, dt, n_samples=batch_size).generate_timeseries(T)
	tget_indices = np.concatenate((np.arange(-task_tau, 0), np.arange(0, task_tau+1)), axis=0)
	outputs = np.zeros((len(input_ts), T, tget_indices.shape[0]))


	# outputs = np.empty((tget_indices.shape[0], T))
	# outputs[:] = np.NaN

	for batch, batch_input in enumerate(input_ts):
		for idx, ii in enumerate(tget_indices):
			if ii < 0:
				outputs[batch, :ii, idx] = batch_input[0, :ii]
			else:
				outputs[batch, ii:, idx] = batch_input[0, ii:]
	inputs = np.swapaxes(np.array(input_ts), 1, 2)

	# inputs_tx1 = jnp.expand_dims(inputs, axis=1)
	# targets_tx1 = jnp.expand_dims(outputs, axis=1)

	return inputs, outputs

	#
	#
	#
	# outputs[0, :] = input_ts[0, :]
	# for ii in range(task_tau):
	# 	outputs[ii+1, ii+1:] = input_ts[0, ii+1:]





def plot_batch(inputs, targets):
	plt.figure(figsize=(10, 7))
	plt.subplot(111)
