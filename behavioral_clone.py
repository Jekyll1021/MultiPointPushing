import tensorflow as tf 
import numpy as np 

DIM_IN = 231

class BC:
	def __init__(self, lr=5e-7, num_hidden=64, num_layers=2, sess=tf.Session(), guassian_model_num=7):
		self.x = tf.placeholder("float", [None, DIM_IN])
		self.y_ = tf.placeholder("float", [None, 4])

		with tf.variable_scope('init'):
			w = tf.Variable(tf.random_normal([DIM_IN, num_hidden]))
			b = tf.Variable(tf.random_normal([num_hidden]))
			h = tf.nn.relu(tf.add(tf.matmul(self.x, w), b))

		for i in range(num_layers - 1):
			with tf.variable_scope(str(i)):
				w = tf.Variable(tf.random_normal([num_hidden, num_hidden]))
				b = tf.Variable(tf.random_normal([num_hidden]))
				h = tf.nn.relu(tf.add(tf.matmul(h, w), b))

		with tf.variable_scope('output'):
			w = tf.Variable(tf.random_normal([num_hidden, 4*guassian_model_num]))
			b = tf.Variable(tf.random_normal([4*guassian_model_num]))
			self.y = tf.add(tf.matmul(h, w), b)

		self.std = []
		shapes = []
		for i2 in range(guassian_model_num):
			with tf.variable_scope(str(i2)):
				self.std.append(tf.get_variable("logstdev", [4], initializer=tf.zeros_initializer()))
			shapes.append(4)
		split = tf.split(self.y, shapes, 1)

		diff_mean = self.y_ - split[0]
		# self.sampled_ac = self.y + tf.random_normal(shape=tf.shape(self.y))*tf.exp(self.std)
		logprob = -1/2*tf.reduce_sum(diff_mean**2/(tf.exp(self.std[0])**2), axis=1) - 4*0.5*tf.log(tf.constant(2*np.pi)) - tf.reduce_sum(self.std[0])
		for i in range(1, guassian_model_num):
			diff_mean = self.y_ - split[i]
			# self.sampled_ac = self.y + tf.random_normal(shape=tf.shape(self.y))*tf.exp(self.std)
			logprob = tf.maximum(logprob, -1/2*tf.reduce_sum(diff_mean**2/(tf.exp(self.std[i])**2), axis=1) - 4*0.5*tf.log(tf.constant(2*np.pi)) - tf.reduce_sum(self.std[i]))

		self.loss = -tf.reduce_sum(logprob)
		opt = tf.train.GradientDescentOptimizer(lr)
		self.train_op = opt.minimize(self.loss)

		min_distr_ind = 0
		for i in range(1, guassian_model_num):
			if tf.reduce_sum(self.std[i] ** 2) < tf.reduce_sum(self.std[min_distr_ind] ** 2):
				min_distr_ind = i
		self.sampled_ac = split[min_distr_ind] + tf.random_normal(shape=tf.shape(split[min_distr_ind]))*tf.exp(self.std[min_distr_ind])

		self.saver = tf.train.Saver()

		self.sess = sess
		self.sess.run(tf.global_variables_initializer())

	def train(self, batch_size, x, y_, steps=1000):
		"""x and y_ as numpy array"""
		
		for i in range(steps):
			ind = np.random.choice(len(x), batch_size, replace=False)
			curr_x = x[ind]
			curr_y_ = y_[ind]
			_, loss = self.sess.run([self.train_op, self.loss], {self.x:curr_x, self.y_:curr_y_})
			if i % 1000 == 0:
				print("iter "+str(i)+" loss: " + str(loss))

	def eval(self, x, y_):
		return self.sess.run([self.loss], {self.x:x, self.y_:y_})[0]

	def predict(self, x):
		return self.sess.run([self.sampled_ac], {self.x:x})

	def save_model(self, path='/nfs/diskstation/zdong/bc_model/model.ckpt'):
		save_path = self.saver.save(self.sess, path)
		print("Model saved in path: %s" % save_path)

	def load_model(self, path='/nfs/diskstation/zdong/bc_model/model.ckpt'):
		self.saver.restore(sess, path)
		print("Model restored.")

if __name__ == "__main__":
	path = "/nfs/diskstation/zdong/singulation_bc/"
	# path = ''
	model = BC()
	for i in range(1000000):
	# for i in range(1):
		ind = np.random.choice(1000)
		input_data = np.load(path + str(ind).zfill(3)+"in.npy")
		output_label = np.load(path + str(ind).zfill(3)+"out.npy")
		model.train(1024, input_data, output_label)
		if i % 1000 == 0:
			model.save_model()


