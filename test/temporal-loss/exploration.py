# Numpy, JAX, Matplotlib and h5py should all be correctly installed and on the python path.
from __future__ import print_function, division, absolute_import
import datetime
import h5py
import jax.numpy as np
from jax import random
from jax.example_libraries import optimizers
from jax import jit, vmap
import matplotlib.pyplot as plt
import numpy as onp             # original CPU-backed NumPy
import os
import sys
import time
from importlib import reload

sys.path.append('/media/neuro/Research/_dev/tests/test/temporal-loss')
import task
import rnn
import utils

# #######################################################
# Parameters
# Integration parameters
T = 1.0          # Arbitrary amount time, roughly physiological.
ntimesteps = 25  # Divide T into this many bins
bval = 0.01      # bias value limit
sval = 0.025     # standard deviation (before dividing by sqrt(dt))
input_params = (bval, sval, T, ntimesteps)

# Integrator RNN hyperparameters
# u = 1         # Number of inputs to the RNN (determined by the task)
n = 100       # Number of units in the RNN
# o = 1         # Number of outputs in the RNN

param_scale = 0.85 # Scaling of the recurrent weight matrix

# Optimization hyperparameters
num_batchs = 100000         # Total number of batches to train on.
batch_size = 128          # How many examples in each batch
eval_batch_size = 1024    # How large a batch for evaluating the RNN
step_size = 0.025          # initial learning rate
decay_factor = 0.99975     # decay the learning rate this much
# Gradient clipping is HUGELY important for training RNNs
max_grad_norm = 10.0      # max gradient norm before clipping, clip to this value.
l2reg = 0.0002           # amount of L2 regularization on the weights
adam_b1 = 0.9             # Adam parameters
adam_b2 = 0.999
adam_eps = 1e-1
print_every = 100          # Print training informatino every so often

# ###############################################################
# JAX handles randomness differently than numpy or matlab.
# one threads the randomness through to each function.
#  It's a bit tedious, but very easy to understand and with
# reliable effect.
seed = onp.random.randint(0, 1000000) # get randomness from CPU level numpy
print("Seed: %d" % seed)
key = random.PRNGKey(seed) # create a random key for jax for use on device.

input_pars = {
	'mg_tau': 17,
	'task_tau': 5,
	'T': 1000,
	'n_steps': 1000,
	'batch_size': 500
}
u = 1
o = (input_pars['task_tau']*2)+1

input_params = (input_pars['mg_tau'], input_pars['task_tau'], input_pars['T'], input_pars['n_steps'], input_pars[
	'batch_size'])

inputs, targets = task.build_input_and_targets(input_params)

# fig, ax = plt.subplots()
plt.plot(inputs[0, :])
plt.show()


# Now batch it and jit.
build_input_and_target = task.build_input_and_targets
# build_inputs_and_targets = vmap(build_input_and_target, in_axes=(None, 0))
build_inputs_and_targets_jit = jit(build_input_and_target,
                                   static_argnums=(0,))





# ################################################################
# Init some parameters for training.
key, subkey = random.split(key, 2)
init_params = rnn.random_vrnn_params(subkey, u, n, o, g=param_scale)

utils.plot_params(init_params)

# Create a decay function for the learning rate
decay_fun = optimizers.exponential_decay(step_size, decay_steps=1,
                                         decay_rate=decay_factor)

batch_idxs = onp.linspace(1, num_batchs)
plt.plot(batch_idxs, [decay_fun(b) for b in batch_idxs])
plt.axis('tight')
plt.xlabel('Batch number')
plt.ylabel('Learning rate')
plt.show()

reload(rnn)
# Initialize the optimizer.  Please see jax/experimental/optimizers.py
opt_init, opt_update, get_params = optimizers.adam(decay_fun, adam_b1, adam_b2, adam_eps)
opt_state = opt_init(init_params)

# Run the optimization loop, first jit'd call will take a minute.
start_time = time.time()
all_train_losses = []
for batch in range(num_batchs):
	key, subkey = random.split(key, 2)
	skeys = random.split(subkey, batch_size)

	inputs, targets = build_inputs_and_targets_jit(input_params)
	opt_state = rnn.update_w_gc_jit(batch, opt_state, opt_update, get_params, inputs,
	                                targets, max_grad_norm, l2reg)
	if batch % print_every == 0:
		params = get_params(opt_state)
		all_train_losses.append(rnn.loss_jit(params, inputs, targets, l2reg))
		train_loss = all_train_losses[-1]['total']
		batch_time = time.time() - start_time
		step_size = decay_fun(batch)
		s = "Batch {} in {:0.2f} sec, step size: {:0.5f}, training loss {:0.4f}"
		print(s.format(batch, batch_time, step_size, train_loss))
		start_time = time.time()

# List of dicts to dict of lists
all_train_losses = {k: [dic[k] for dic in all_train_losses] for k in all_train_losses[0]}

# Show the loss through training.
xlims = [2, 50]
plt.figure(figsize=(16,4))
plt.subplot(141)
plt.plot(all_train_losses['total'][xlims[0]:xlims[1]], 'k')
plt.title('Total')

plt.subplot(142)
plt.plot(all_train_losses['lms'][xlims[0]:xlims[1]], 'r')
plt.title('Least mean square')

plt.subplot(143)
plt.plot(all_train_losses['l2'][xlims[0]:xlims[1]], 'g');
plt.title('L2')

plt.subplot(144)
plt.plot(all_train_losses['total'][xlims[0]:xlims[1]], 'k')
plt.plot(all_train_losses['lms'][xlims[0]:xlims[1]], 'r')
plt.plot(all_train_losses['l2'][xlims[0]:xlims[1]], 'g')
plt.title('All losses')
plt.show()

# #########################################################
# Take a batch for an evalulation loss, notice the L2 penalty is 0
# for the evaluation.
params = get_params(opt_state)

key, subkey = random.split(key, 2)
skeys = random.split(subkey, batch_size)
inputs, targets = build_inputs_and_targets_jit(input_params)
eval_loss = rnn.loss_jit(params, inputs, targets, l2reg=0.0)['total']
eval_loss_str = "{:.5f}".format(eval_loss)
print("Loss on a new large batch: %s" % (eval_loss_str))


reload(rnn)


# Visualize how good this trained integrator is
def inputs_targets_no_h0s():
	inputs_b, targets_b = \
		build_inputs_and_targets_jit(input_params)
	h0s_b = None # Use trained h0
	return inputs_b, targets_b, h0s_b


rnn_run = lambda inputs: rnn.batched_rnn_run(params, inputs)

give_trained_h0 = lambda batch_size : np.array([params['h0']] * batch_size)

rnn_internals = rnn.run_trials(rnn_run, inputs_targets_no_h0s, 1, 16)

# integrator.plot_batch(ntimesteps, rnn_internals['inputs'],
#                       rnn_internals['targets'], rnn_internals['outputs'],
#                       onp.abs(rnn_internals['targets'] - rnn_internals['outputs']))


## Plot MSE

## Plot capacity
def compute_capacity(z_hat, z):
	"""
	Compute capacity to reconstruct z based on linearly combining x
	:param X: state matrix (NxT)
	:param z: target output (1xT)
	:return: capacity
	"""
	covs = (onp.cov(z_hat, z)[0, 1] ** 2.)
	vars = (onp.var(z) * onp.var(z_hat))
	capacity = covs / vars
	error = onp.mean((z - z_hat) ** 2)
	return capacity, error


tget_indices = onp.concatenate((onp.arange(-input_pars['task_tau'], 0),
                               onp.arange(0, input_pars['task_tau']+1)), axis=0)

capacities = onp.zeros(shape=(input_pars['batch_size'], len(tget_indices)))
errors = onp.zeros(shape=(input_pars['batch_size'], len(tget_indices)))
for batch in range(input_pars['batch_size']):
	for idx, k in enumerate(tget_indices):
		z_hat = rnn_internals['outputs'][batch, :, k]
		z = rnn_internals['targets'][batch, :, k]
		cap, mse = compute_capacity(z_hat, z)
		capacities[batch, idx] = cap
		errors[batch, idx] = mse


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].errorbar(tget_indices, capacities.mean(0), capacities.std(0))
ax[0].set_title("Capacity")
ax[1].errorbar(tget_indices, errors.mean(0), errors.std(0))
ax[1].set_title("MSE")
plt.show()

# Take a look at the trained parameters.
utils.plot_params(params)
plt.show()