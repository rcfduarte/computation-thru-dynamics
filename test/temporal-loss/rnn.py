from __future__ import print_function, division, absolute_import
import datetime
import h5py
from functools import partial

import jax.numpy as np
from jax import grad, jit, lax, random, vmap
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import os
import time

import integrator_rnn_tutorial.utils as utils

MAX_SEED_INT = 10000000


def random_vrnn_params(key, u, n, o, g=1.0):
	"""
	Generate random RNN parameters
	"""
	key, skeys = utils.keygen(key, 4)
	hscale = 0.1
	ifactor = 1.0 / np.sqrt(u)
	hfactor = g / np.sqrt(n)
	pfactor = 1.0 / np.sqrt(n)
	return {'h0': random.normal(next(skeys), (n,)) * hscale,
	        'wI': random.normal(next(skeys), (n, u)) * ifactor,
	        'wR': random.normal(next(skeys), (n, n)) * hfactor,
	        'wO': random.normal(next(skeys), (o, n)) * pfactor,
	        'bR': np.zeros([n]),
	        'bO': np.zeros([o])}


def affine(params, x):
	"""
	Implement y = w x + b
	"""
	return np.dot(params['wO'], x) + params['bO']


# Affine expects n_W_m m_x_1, but passing in t_x_m (has txm dims) So
# map over first dimension to hand t_x_m.  I.e. if affine yields
# n_y_1 = dot(n_W_m, m_x_1), then batch_affine yields t_y_n.
batch_affine = vmap(affine, in_axes=(None, 0))


def vrnn(params, h, x):
	"""
	Run the Vanilla RNN one step
	"""
	a = np.dot(params['wI'], x) + params['bR'] + np.dot(params['wR'], h)
	return np.tanh(a)


def vrnn_scan(params, h, x):
	"""
	Run the Vanilla RNN one step, returning (h ,h).
	"""
	h = vrnn(params, h, x)
	return h, h


def vrnn_run_with_h0(params, x_t, h0):
	"""Run the Vanilla RNN T steps, where T is shape[0] of input."""
	h = h0
	f = partial(vrnn_scan, params)
	_, h_t = lax.scan(f, h, x_t)
	o_t = batch_affine(params, h_t)
	return h_t, o_t


def vrnn_run(params, x_t):
	"""Run the Vanilla RNN T steps, where T is shape[0] of input."""
	return vrnn_run_with_h0(params, x_t, params['h0'])


# Let's upgrade it to handle batches using `vmap`
# Make a batched version of the `predict` function
batched_rnn_run = vmap(vrnn_run, in_axes=(None, 0))
batched_rnn_run_w_h0 = vmap(vrnn_run_with_h0, in_axes=(None, 0, 0))


def loss(params, inputs_bxtxu, targets_bxtxo, l2reg):
	"""Compute the least squares loss of the output, plus L2 regularization."""
	_, outs_bxtxo = batched_rnn_run(params, inputs_bxtxu)
	l2_loss = l2reg * optimizers.l2_norm(params) ** 2
	lms_loss = np.mean((outs_bxtxo - targets_bxtxo) ** 2)
	total_loss = lms_loss + l2_loss
	return {'total': total_loss, 'lms': lms_loss, 'l2': l2_loss}


def update_w_gc(i, opt_state, opt_update, get_params, x_bxt, f_bxt, max_grad_norm, l2reg):
	"""
	Update the parameters w/ gradient clipped, gradient descent updates.
	"""
	params = get_params(opt_state)

	def training_loss(params, x_bxt, f_bxt, l2reg):
		return loss(params, x_bxt, f_bxt, l2reg)['total']

	grads = grad(training_loss)(params, x_bxt, f_bxt, l2reg)
	clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
	return opt_update(i, clipped_grads, opt_state)


loss_jit = jit(loss)
update_w_gc_jit = jit(update_w_gc, static_argnums=(2,3))


def run_trials(batched_run_fun, inputs_targets_h0s_fun, nbatches, batch_size):
	"""Run a bunch of trials and save everything in a dictionary."""
	inputs = []
	hiddens = []
	outputs = []
	targets = []
	h0s = []
	for n in range(nbatches):
		data_seeds = onp.random.randint(0, MAX_SEED_INT, size=batch_size)
		keys = np.array([random.PRNGKey(ds) for ds in data_seeds])
		input_b, target_b, h0s_b = inputs_targets_h0s_fun(keys)
		if h0s_b is None:
			h_b, o_b = batched_run_fun(input_b)
		else:
			h_b, o_b = batched_run_fun(input_b, h0s_b)
			h0s.append(h0s_b)

		inputs.append(input_b)
		hiddens.append(h_b)
		outputs.append(o_b)
		targets.append(target_b)

	trial_dict = {'inputs' : onp.vstack(inputs), 'hiddens' : onp.vstack(hiddens),
	              'outputs' : onp.vstack(outputs), 'targets' : onp.vstack(targets)}
	if h0s_b is not None:
		trial_dict['h0s'] = onp.vstack(h0s)
	else:
		trial_dict['h0s'] = None
	return trial_dict