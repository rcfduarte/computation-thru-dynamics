from __future__ import print_function, division, absolute_import
import h5py
import jax.numpy as np
from jax import random
import numpy as onp  # original CPU-backed NumPy
import matplotlib.pyplot as plt
import os


def ensure_dir(file_path):
	"""Make sure the directory exists, create if it does not."""
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)


def write_file(data_fname, data_dict):
	"""Write a simple dictionary using h5py."""
	try:
		ensure_dir(data_fname)
		with h5py.File(data_fname, 'w') as hf:
			for k in data_dict:
				hf.create_dataset(k, data=data_dict[k])
			# add attributes
	except IOError:
		print("Cannot write % for writing." % data_fname)
		raise


def read_file(data_fname):
	"""Read a simple dictionary of np arrays using h5py."""
	try:
		with h5py.File(data_fname, 'r') as hf:
			data_dict = {k: onp.array(v) for k, v in hf.items()}
			return data_dict
	except IOError:
		print("Cannot open %s for reading." % data_fname)
		raise


def keygen(key, nkeys):
	"""
	Generate randomness that JAX can use by splitting the JAX keys.
	Args:
	  key : the random.PRNGKey for JAX
	  nkeys : how many keys in key generator
	Returns:
	  2-tuple (new key for further generators, key generator)
	"""
	keys = random.split(key, nkeys + 1)
	return keys[0], (k for k in keys[1:])


# ##################################################
# Plot
def plot_params(params):
	""" Plot the parameters of the vanilla RNN. """
	plt.figure(figsize=(16,8))
	plt.subplot(231)
	plt.stem(params['wO'][0,:])
	plt.title('wO - output weights')

	plt.subplot(232)
	plt.stem(params['h0'])
	plt.title('h0 - initial hidden state')

	plt.subplot(233)
	plt.imshow(params['wR'], interpolation=None)
	plt.title('wR - recurrent weights')
	plt.colorbar()

	plt.subplot(234)
	plt.stem(params['wI'])
	plt.title('wI - input weights')

	plt.subplot(235)
	plt.stem(params['bR'])
	plt.title('bR - recurrent biases')

	plt.subplot(236)
	evals, _ = onp.linalg.eig(params['wR'])
	x = onp.linspace(-1, 1, 1000)
	plt.plot(x, onp.sqrt(1-x**2), 'k')
	plt.plot(x, -onp.sqrt(1-x**2), 'k')
	plt.plot(onp.real(evals), onp.imag(evals), '.')
	plt.axis('equal')
	plt.title('Eigenvalues of wR')


def plot_examples(ntimesteps, rnn_internals, nexamples=1):
	"""Plot some input/hidden/output triplets."""
	plt.figure(figsize=(nexamples*5, 12))
	for bidx in range(nexamples):
		plt.subplot(3, nexamples, bidx+1)
		plt.plot(rnn_internals['inputs'][bidx,:], 'k')
		plt.xlim([0, ntimesteps])
		plt.title('Example %d' % (bidx))
		if bidx == 0:
			plt.ylabel('Input')

	ntoplot = 10
	closeness = 0.25
	for bidx in range(nexamples):
		plt.subplot(3, nexamples, nexamples+bidx+1)
		plt.plot(rnn_internals['hiddens'][bidx, :, 0:ntoplot] +
		         closeness * onp.arange(ntoplot), 'b')
		plt.xlim([0, ntimesteps])
		if bidx == 0:
			plt.ylabel('Hidden Units')

	for bidx in range(nexamples):
		plt.subplot(3, nexamples, 2*nexamples+bidx+1)
		plt.plot(rnn_internals['outputs'][bidx,:,:], 'r')
		plt.plot(rnn_internals['targets'][bidx,:,:], 'k')
		plt.xlim([0, ntimesteps])
		plt.xlabel('Timesteps')
		if bidx == 0:
			plt.ylabel('Output')
