from glob import glob
from PIL import Image
import tensorflow as tf
import numpy as np
import scipy.io
import time
import os
import collections
import argparse
from ops import *

def load_image(im_path):
	im = Image.open(im_path)
	im = np.expand_dims(np.array(im).astype(np.float32), axis=0)
	im = im/127.5 - 1.0
	return im

def save_image(im, im_path, phase, ptype, exp_name):
	im = np.uint8((im+1.)*127.5)
	im = Image.fromarray(np.squeeze(im))
	data_name = im_path.split(os.sep)[3]
	if phase=="train":
		im.save(os.path.join('checkpoints'+exp_name, ptype+data_name))
	else:
		im.save(os.path.join('result'+exp_name,  ptype+data_name))


def discriminator(inputs, targets, opt, name, update_collection=None, reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		#CNN
		ndf = opt.ndf
		x = tf.concat([inputs, targets], axis=3)

		x = conv2d(inputs, ndf, 4, strides=2, spectral_normed=True, update_collection=update_collection, name='d_conv_1')
		x = leaky_relu(instance_norm(x,'d_bn_1'))

		x = conv2d(x, ndf*2, 4, strides=2, spectral_normed=True, update_collection=update_collection, name='d_conv_2')
		x = leaky_relu(instance_norm(x,'d_bn_2'))

		x = conv2d(x, ndf*4, 4, strides=2, spectral_normed=True, update_collection=update_collection, name='d_conv_3')
		x = leaky_relu(instance_norm(x,'d_bn_3'))

		x = conv2d(x, ndf*8, 4, strides=1, spectral_normed=True, update_collection=update_collection, name='d_conv_4')
		x = leaky_relu(instance_norm(x,'d_bn_4'))

		x = conv2d(x, 1, 4, strides=1,  spectral_normed=True, update_collection=update_collection, padding='SAME', name = "d_pred_1")
		out = tf.sigmoid(x)
		return out

def generator(inputs, is_training, opt):
	
	with tf.variable_scope("encoder") as scope:
		
		ndf = opt.ndf
		#encoder
		ndf_spec = [ndf, ndf*2, ndf*4, ndf*8, ndf*8, ndf*8, ndf*8]
		# rate = [1, 2, 2, 1, 1, 1, 1]
		encode_layers = []
		x = conv2d(inputs, ndf_spec[0], 4, strides=2, padding='SAME', name='g_conv_1')
		encode_layers.append(x)

		for i in range(1,5): 
			x = leaky_relu(x)
			# x = conv2d(x, 1, 4, strides=1, rate=rate[i],padding='SAME', name='g_dia_%d'%(i+1))
			x = conv2d(x, ndf_spec[i], 4, strides=2 ,padding='SAME', name='g_conv_%d'%(i+1))
			x = instance_norm(x, 'g_bn_%d'%(i+1))
			encode_layers.append(x)

	with tf.variable_scope("decoder") as scope:
		
		#encoder
		ngf = opt.ngf
		ngf_spec = [ngf*8, ngf*8, ngf*8, ngf*4, ngf*2, ngf]
		

		for i in range(0,4):
			if i != 0:
				x = tf.concat([x, encode_layers[4-i]], axis=3)
			x = tf.nn.relu(x)
			x = deconv2d(x, ngf_spec[i], 4, strides=2, padding='SAME', name='g_deconv_%d'%(i+1))
			x = instance_norm(x,'g_bn_%d'%(i+1))
			if i < 3:
				x = dropout(x, rate=0.5, training=is_training)

		x = tf.concat([x, encode_layers[0]], axis=3)
		x = tf.nn.relu(x)
		x = deconv2d(x, 3, strides=2, padding='SAME', name='g_deconv_7')
		output = tf.tanh(x)

		return output

def transform(A, B, C, scale_size, crop_size):
    r_A = tf.image.resize_images(A, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
    r_B = tf.image.resize_images(B, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
    r_C = tf.image.resize_images(C, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - crop_size + 1)), dtype=tf.int32)
    if scale_size > crop_size:
        r_A = tf.image.crop_to_bounding_box(r_A, offset[0], offset[1], crop_size, crop_size)
        r_B = tf.image.crop_to_bounding_box(r_B, offset[0], offset[1], crop_size, crop_size)
        r_C = tf.image.crop_to_bounding_box(r_C, offset[0], offset[1], crop_size, crop_size)
    elif scale_size < crop_size:
        raise Exception("scale size cannot be less than crop size")
    return r_A, r_B, r_C

def create_network(opt):
	#parameter
	height = opt.height
	width = opt.width
	in_depth = opt.in_depth
	out_depth = opt.out_depth
	lambda_A = opt.lambda_A
	EPS = 1e-12
	starter_learning_rate = 0.0002
	end_learning_rate = 0.0
	start_decay_step = 200000
	decay_steps = 50000

	# start_decay_step = 200
	# decay_steps = 20
	beta1 = 0.5
	global_step_B = tf.Variable(0, trainable=False) # for blur generator
	global_step_S = tf.Variable(0, trainable=False) # for sharp generator
	global_step_T = tf.Variable(0, trainable=False) # for total discriminator
	Model = collections.namedtuple("Model", ['global_step_T','learning_rate_B', 'learning_rate_S', 'learning_rate_T', 'data', 'is_training', 'input_A', 'input_B', 'fake_blur_B', 'fake_B', 'd_B_solver',\
	'g_B_solver', 'd_S_solver', 'g_S_solver', 'd_T_solver', 'g_T_solver', 'g_B_loss_L1_summary', 'g_B_loss_GAN_summary', 'd_B_loss_sum', 'g_S_loss_L1_summary', 'g_S_loss_GAN_summary', 'd_S_loss_sum','g_T_loss_L1_summary', 'g_T_loss_GAN_summary', 'd_T_loss_sum'])
	
	#placeholder/input
	data = tf.placeholder(tf.float32, [None, height, width*3, in_depth], name ="data_AB")
	is_training = tf.placeholder(tf.bool, name ="is_training")

	input_B, input_A, blur_B = transform(data[:, :, :opt.width, :], data[:, :, opt.width:opt.width*2-1, :], data[:, :, opt.width*2:, :], width+10, width)

	#generator
	with tf.variable_scope("generatorB"): # blur generator
		fake_blur_B = generator(input_A, is_training, opt)


	with tf.variable_scope("generatorS"):	# sharp generator
		fake_B = generator(fake_blur_B, is_training, opt)

	#discriminator
	d_B_real = discriminator(input_A, blur_B, opt, update_collection=None,  name="discriminatorB")
	d_B_fake = discriminator(input_A, fake_blur_B, opt, update_collection="NO_OPS", name="discriminatorB", reuse=True)

	d_S_real = discriminator(blur_B, input_B, opt, update_collection=None,  name="discriminatorS")
	d_S_fake = discriminator(blur_B, fake_B, opt, update_collection="NO_OPS", name="discriminatorS", reuse=True)

	d_T_real = discriminator(input_A, input_B, opt, update_collection=None,  name="discriminatorT")
	d_T_fake = discriminator(input_A, fake_B, opt, update_collection="NO_OPS", name="discriminatorT", reuse=True)


	#loss
	with tf.variable_scope("discriminator_loss"):
		d_B_loss = tf.reduce_mean(-(tf.log(d_B_real + EPS) + tf.log(1 - d_B_fake + EPS)))
		d_S_loss = tf.reduce_mean(-(tf.log(d_S_real + EPS) + tf.log(1 - d_S_fake + EPS)))
		d_T_loss = tf.reduce_mean(-(tf.log(d_T_real + EPS) + tf.log(1 - d_T_fake + EPS)))

	with tf.variable_scope("generator_loss"):
		g_B_loss_GAN = tf.reduce_mean(-tf.log(d_B_fake + EPS))
		g_B_loss_L1 = tf.reduce_mean(tf.abs(blur_B - fake_blur_B))
		g_B_loss = g_B_loss_GAN  + g_B_loss_L1 * lambda_A

		g_S_loss_GAN = tf.reduce_mean(-tf.log(d_S_fake + EPS))
		g_S_loss_L1 = tf.reduce_mean(tf.abs(input_B - fake_B))
		g_S_loss = g_S_loss_GAN  + g_S_loss_L1 * lambda_A

		g_T_loss_GAN = tf.reduce_mean(-tf.log(d_T_fake + EPS))
		g_T_loss_L1 = tf.reduce_mean(tf.abs(input_B - fake_B))
		g_T_loss = g_T_loss_GAN  + g_T_loss_L1 * lambda_A

	#tensorboard summary
	g_B_loss_L1_summary = tf.summary.scalar("g_B_loss_L1", g_B_loss_L1)
	g_B_loss_GAN_summary = tf.summary.scalar("g_B_loss_GAN", g_B_loss_GAN)
	d_B_loss_sum = tf.summary.scalar("d_B_loss", d_B_loss)

	g_S_loss_L1_summary = tf.summary.scalar("g_S_loss_L1", g_S_loss_L1)
	g_S_loss_GAN_summary = tf.summary.scalar("g_S_loss_GAN", g_S_loss_GAN)
	d_S_loss_sum = tf.summary.scalar("d_S_loss", d_S_loss)

	g_T_loss_L1_summary = tf.summary.scalar("g_T_loss_L1", g_T_loss_L1)
	g_T_loss_GAN_summary = tf.summary.scalar("g_T_loss_GAN", g_T_loss_GAN)
	d_T_loss_sum = tf.summary.scalar("d_T_loss", d_T_loss)

	# optimizer
	learning_rate_B = (
		tf.where(
			tf.greater_equal(global_step_B, start_decay_step),
			tf.train.polynomial_decay(starter_learning_rate, global_step_B-start_decay_step,
										decay_steps, end_learning_rate,
										power=1.0),
			starter_learning_rate
		)
	)

	learning_rate_S = (
		tf.where(
			tf.greater_equal(global_step_S, start_decay_step),
			tf.train.polynomial_decay(starter_learning_rate, global_step_S-start_decay_step,
										decay_steps, end_learning_rate,
										power=1.0),
			starter_learning_rate
		)
	)

	learning_rate_T = (
		tf.where(
			tf.greater_equal(global_step_T, start_decay_step),
			tf.train.polynomial_decay(starter_learning_rate, global_step_T-start_decay_step,
										decay_steps, end_learning_rate,
										power=1.0),
			starter_learning_rate
		)
	)
	trainable_variables_DB = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminatorB')
	trainable_variables_GB = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generatorB')
	
	trainable_variables_DS = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminatorS')
	trainable_variables_GS = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generatorS')
	
	trainable_variables_DT = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminatorT')
	trainable_variables_GT = trainable_variables_GB + trainable_variables_GS
	# print(trainable_variables_GB)
	# print(trainable_variables_GS)
	# print(trainable_variables_GT)
	d_B_solver = tf.train.AdamOptimizer(learning_rate_T, 0.5).minimize(d_B_loss, global_step = global_step_B, var_list=trainable_variables_DB)
	g_B_solver = tf.train.AdamOptimizer(learning_rate_T, 0.5).minimize(g_B_loss, var_list=trainable_variables_GB)
	
	d_S_solver = tf.train.AdamOptimizer(0.0002, 0.5).minimize(d_S_loss, global_step = global_step_S, var_list=trainable_variables_DS)
	g_S_solver = tf.train.AdamOptimizer(0.0002, 0.5).minimize(g_S_loss, var_list=trainable_variables_GS)
	
	d_T_solver = tf.train.AdamOptimizer(learning_rate_T, 0.5).minimize(d_T_loss, global_step = global_step_T, var_list=trainable_variables_DT)
	g_T_solver = tf.train.AdamOptimizer(learning_rate_T, 0.5).minimize(g_T_loss, var_list=trainable_variables_GT)
	
	return Model(
		global_step_T=global_step_T,
		input_A=input_A,
		input_B=input_B,
		learning_rate_B=learning_rate_B,
		learning_rate_S=learning_rate_S,
		learning_rate_T=learning_rate_T,
		is_training=is_training,
	 	data=data,
	 	fake_blur_B=fake_blur_B,
	 	fake_B=fake_B,
	 	d_B_solver=d_B_solver,
	 	g_B_solver=g_B_solver,
	 	d_S_solver=d_S_solver,
	 	g_S_solver=g_S_solver,
	 	d_T_solver=d_T_solver,
	 	g_T_solver=g_T_solver,
	 	g_B_loss_L1_summary=g_B_loss_L1_summary,
	 	g_B_loss_GAN_summary=g_B_loss_GAN_summary,
	 	d_B_loss_sum=d_B_loss_sum,
	 	g_S_loss_L1_summary=g_S_loss_L1_summary,
	 	g_S_loss_GAN_summary=g_S_loss_GAN_summary,
	 	d_S_loss_sum=d_S_loss_sum,
	 	g_T_loss_L1_summary=g_T_loss_L1_summary,
	 	g_T_loss_GAN_summary=g_T_loss_GAN_summary,
	 	d_T_loss_sum=d_T_loss_sum)



def train(opt, model):

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		exp_name = opt.experiment_name
		writer = tf.summary.FileWriter("logs"+exp_name, sess.graph)
		saver = tf.train.Saver()
		if not os.path.exists('result'+exp_name):
			os.makedirs('result'+exp_name)
		if not os.path.exists('checkpoints'+exp_name):
			os.makedirs('checkpoints'+exp_name)

		with tf.name_scope("parameter_count"):
			parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
		print(sess.run(parameter_count))

		initial_step = 0

		ckpt = tf.train.latest_checkpoint('./checkpoints'+exp_name)
		if ckpt:
			saver.restore(sess, ckpt)
			initial_step  = int(os.path.basename(ckpt).split('-')[1])	
		print(initial_step)
		counter = initial_step//1000
		start_time = time.time()
		data_blur = glob(os.path.join('datasets', opt.dataset, 'train', '*.png'))
		data_clear = glob(os.path.join('datasets', opt.dataset, 'train_clear', '*.png'))
		for epoch in range(initial_step, opt.epoch):
			# if epoch<30:
			# 	data = data_blur
			# else:
			# 	data = data_clear

			t_lr = sess.run(model.learning_rate_T)
			g = sess.run(model.global_step_T)
			print('total learning rate :'+str(t_lr))
			print('step : '+str(g))
			data = data_blur
			np.random.shuffle(data)
			for i in range(0, len(data)):
				im = load_image(data[i])

				if epoch<=50:
					gen_B_iter = 2
				else:
					gen_B_iter = 1

				if epoch<150:
					for j in range(gen_B_iter):
						
						_, g_B_loss_L1_summary_str, g_B_loss_GAN_summary_str = sess.run([model.g_B_solver, model.g_B_loss_L1_summary,\
						 model.g_B_loss_GAN_summary], feed_dict ={model.data: im, model.is_training: True})
						writer.add_summary(g_B_loss_L1_summary_str, counter)
						writer.add_summary(g_B_loss_GAN_summary_str, counter)

						_, d_B_loss_sum_str = sess.run([model.d_B_solver, model.d_B_loss_sum],\
						 feed_dict ={model.data: im, model.is_training: True})
						writer.add_summary(d_B_loss_sum_str, counter)


						t = np.abs(len(data)-i-1)
						im = load_image(data[t])


					if epoch > 50:
						_, d_T_loss_sum_str = sess.run([model.d_T_solver, model.d_T_loss_sum],\
						 feed_dict ={model.data: im, model.is_training: True})
						writer.add_summary(d_T_loss_sum_str, counter)
						
						_, g_S_loss_L1_summary_str, g_S_loss_GAN_summary_str = sess.run([model.g_S_solver, model.g_S_loss_L1_summary,\
						 model.g_S_loss_GAN_summary], feed_dict ={model.data: im, model.is_training: True})
						writer.add_summary(g_S_loss_L1_summary_str, counter)
						writer.add_summary(g_S_loss_GAN_summary_str, counter)

						_, d_S_loss_sum_str = sess.run([model.d_S_solver, model.d_S_loss_sum],\
						 feed_dict ={model.data: im, model.is_training: True})
						writer.add_summary(d_S_loss_sum_str, counter)
				else:
					_, g_B_loss_L1_summary_str, g_B_loss_GAN_summary_str = sess.run([model.g_B_solver, model.g_B_loss_L1_summary,\
					model.g_B_loss_GAN_summary], feed_dict ={model.data: im, model.is_training: True})
					writer.add_summary(g_B_loss_L1_summary_str, counter)
					writer.add_summary(g_B_loss_GAN_summary_str, counter)

					_, d_B_loss_sum_str = sess.run([model.d_B_solver, model.d_B_loss_sum],\
					feed_dict ={model.data: im, model.is_training: True})
					writer.add_summary(d_B_loss_sum_str, counter)

					_, g_T_loss_L1_summary_str, g_T_loss_GAN_summary_str = sess.run([model.g_T_solver, model.g_T_loss_L1_summary,\
					model.g_T_loss_GAN_summary], feed_dict ={model.data: im, model.is_training: True})
					writer.add_summary(g_T_loss_L1_summary_str, counter)
					writer.add_summary(g_T_loss_GAN_summary_str, counter)

					_, d_T_loss_sum_str = sess.run([model.d_T_solver, model.d_T_loss_sum],\
					 feed_dict ={model.data: im, model.is_training: True})
					writer.add_summary(d_T_loss_sum_str, counter)

				if i % 100 == 0:
					print("Epoch: [%2d] [%4d/%4d]" % (epoch, i, len(data)))
				counter += 1
			input_A, input_B, gen_fake_blur_B, gen_fake_B = sess.run([model.input_A, model.input_B, model.fake_blur_B, model.fake_B], feed_dict ={model.data: im, model.is_training: False})
			save_image(gen_fake_B, data[i], 'train', '%3df'%epoch, exp_name)
			save_image(gen_fake_blur_B, data[i], 'train', '%3df_blur'%epoch, exp_name)
			save_image(input_A, data[i], 'train', '%3dA'%epoch, exp_name)
			save_image(input_B, data[i], 'train', '%3dB'%epoch, exp_name)
			if (epoch+1) % 10 == 0:
				saver.save(sess, 'checkpoints'+exp_name+'/pix2pix', int(counter/len(data)))
			
			
		

def test(opt, model):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		exp_name = opt.experiment_name
		ckpt = tf.train.get_checkpoint_state('./checkpoints'+exp_name)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print()
		if not os.path.exists('result'+exp_name):
			os.makedirs('result'+exp_name)

		data = glob(os.path.join('datasets', opt.dataset, 'test', '*.png'))
		np.random.shuffle(data)
		for i in range(0, len(data)):
			im = load_image(data[i])
			gen_fake_B = sess.run(model.fake_B, feed_dict ={model.data: im, model.is_training: False})
			save_image(gen_fake_B, data[i], 'test', 'result', exp_name)


def main():
	parser = argparse.ArgumentParser(description="pix2pix.")
	parser.add_argument('dataset', help='dataset directory', type=str)
	parser.add_argument('--test_dir', help='L1 weight', default='test', type=str)	
	parser.add_argument('--height', help='image height', default=128, type=int)
	parser.add_argument('--width', help='image width', default=128, type=int)
	parser.add_argument('--epoch', help='image epoch', default=200, type=int)
	parser.add_argument('--in_depth', help='image depth', default=3, type=int)	
	parser.add_argument('--out_depth', help='output image depth', default=3, type=int)	
	parser.add_argument('--lambda_A', help='L1 weight', default=100, type=int)
	parser.add_argument('--lr', help='learning rate', default=0.0002, type=float)
	parser.add_argument('--ndf', help='number of discriminator filer', default=64, type=int)
	parser.add_argument('--ngf', help='number of generator filer', default=64, type=int)
	parser.add_argument('--experiment_name', help='enter what you did in this experiment', default='_')

	opts = parser.parse_args()

	model = create_network(opts)
	train(opts, model)
	test(opts, model)

if __name__ == "__main__":
    main()
