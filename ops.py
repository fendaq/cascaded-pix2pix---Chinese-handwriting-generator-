import tensorflow as tf


import tensorflow as tf
import warnings


NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
	return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
	# Usually num_iters = 1 will be enough
	W_shape = W.shape.as_list()
	W_reshaped = tf.reshape(W, [-1, W_shape[-1]])

	if u is None:
		u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

	def power_iteration(i, u_i, v_i):
		v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
		u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
		return i + 1, u_ip1, v_ip1

	_, u_final, v_final = tf.while_loop(
	cond=lambda i, _1, _2: i < num_iters,
	body=power_iteration,
	loop_vars=(tf.constant(0, dtype=tf.int32),
		u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]])))
	if update_collection is None:
		warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
		sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
		W_bar = W_reshaped / sigma
		with tf.control_dependencies([u.assign(u_final)]):
			W_bar = tf.reshape(W_bar, W_shape)
	else:
		sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
		W_bar = W_reshaped / sigma
		W_bar = tf.reshape(W_bar, W_shape)
		if update_collection != NO_OPS:
			tf.add_to_collection(update_collection, u.assign(u_final))

	if with_sigma:
		return W_bar, sigma
	else:
		return W_bar




def leaky_relu(x, leak=0.2, name="leaky_relu"):
	     with tf.variable_scope(name):
	         f1 = 0.5 * (1 + leak)
	         f2 = 0.5 * (1 - leak)
	         return f1 * x + f2 * abs(x)

def instance_norm(x, name="instance_norm"):
	with tf.variable_scope(name):
	    epsilon = 1e-5
	    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
	    scale = tf.get_variable('scale',[x.get_shape()[-1]], 
	        initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
	    offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
	    out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
	    return out

def dropout(x, rate=0.5, training=True):
	return tf.layers.dropout(x, rate=0.5, training=training)

def conv2d(x, output_dim, kernel_size=4, strides=2, padding = "SAME", spectral_normed=False,
			update_collection=None, stddev = 0.02, name="conv_2d"):
	
	with tf.variable_scope(name) as scope:
		w = tf.get_variable("w", [kernel_size, kernel_size, x.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
		if spectral_normed:
			conv = tf.nn.conv2d(x, spectral_normed_weight(w, update_collection=update_collection),
								strides=[1, strides, strides, 1], padding=padding)
		else:
			conv = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
		
		conv = tf.nn.bias_add(conv, biases)

		return conv

"""def conv2d(x, num_outputs, kernel_size=4, strides=2, padding = "SAME", stddev = 0.02, name="conv_2d"):
	return tf.contrib.layers.conv2d(x, num_outputs, kernel_size, strides, padding,\
	  activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
"""	
def deconv2d(x, num_outputs, kernel_size=4, strides=2, padding = "SAME", stddev = 0.02, name="conv_2d"):
	return tf.contrib.layers.conv2d_transpose(x, num_outputs, kernel_size, strides, padding,\
	  activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))


def dense(x, output_dim, spectral_normed=False, update_collection=None, name="dense"):
	with tf.variable_scope(name) as scope:
		w = tf.get_variable("w", [x.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
		if spectral_normed:
			mul = tf.matmul(x, spectral_normed_weight(w, update_collection=update_collection))
		else:
			mul =  tf.matmul(x, w)
	return mul + b



def residule_block(x, dim, name="residule_block"):
	with tf.variable_scope(name) as scope:
		y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
		y = conv2d(y, dim, 3, strides=1, padding='VALID', name='r_conv_1')
		y = tf.nn.relu(instance_norm(y,'r_bn_1'))

		y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
		y = conv2d(y, dim, 3, strides=1, padding='VALID', name='r_conv_2')
		y = instance_norm(y,'r_bn_2')
		return y+x

