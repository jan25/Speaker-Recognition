
import random
import numpy as np
import accuracy

class Ann:
	
	def __init__(self, size):
		self.size = size # network size
		self.w = []
		self.init_weights() # parameters in ann
		self.error = 0.0 # total error
		self.acc = 0.0
		self.stopat = (0.0, 0)


	def rand_matrix(self, a, b):
		return 2 * np.random.random((a, b)) - 1
	
	def init_weights(self):
		for l in range(len(self.size) - 2):
			self.w.append(self.rand_matrix(self.size[l] + 1, self.size[l + 1] + 1))
		self.w.append(self.rand_matrix(self.size[l + 1] + 1, self.size[l + 2]))
 

	def add_bias(self, inputs):
		return np.concatenate( \
				(np.array([ \
					np.ones(inputs.shape[0])]).T, inputs), axis = 1)

	def train(self, inputs, target, train_accu = True, iter = 500, rate = 0.1):
		inputs = self.add_bias(inputs)

		for it in range(iter):
			self.error = 0.0
			for s in range(inputs.shape[0]):
				a = self.prop_forward([inputs[s]])
				self.back_prop(target[s], a, rate)
				self.error += self.sample_error(inputs[s], target[s])
			
			if it % 10 == 0:
				print ('iter:', it)
				print ('error:', self.error)
				if train_accu:
					self.accu = 0.0
					for s in range(inputs.shape[0]):
						if accuracy.check(self.test(inputs[s]), target[s]):
							self.accu += 1
					self.accu = (self.accu / inputs.shape[0]) * 100.0
					if self.stopat[0] < self.accu:
						self.stopat = (self.accu, it)
					print ('accuracy:', self.accu)
					print ('max acc:', self.stopat)

	def prop_forward(self, a):
		for l in range(len(self.w)):
			prod = np.dot(a[l], self.w[l])
			a.append(self.sigmoid(prod))
		return a

	def back_prop(self, t, a, rate):
		error = t - a[-1]
		d = [error * self.sigmoid_d(a[-1])]
		for l in range(len(a) - 2, 0, -1):
			d.append(d[-1].dot(self.w[l].T) * self.sigmoid_d(a[l]))
		d.reverse()

		for l in range(len(self.w)):
			self.w[l] += rate * np.array([a[l]]).T.dot(np.array([d[l]]))


	def sigmoid(self, x):
	    return 1.0 / (1.0 + np.exp(-x))

	def sigmoid_d(self, x):
	    sig = self.sigmoid(x)
	    return sig * (1.0 - sig)


	def sample_error(self, input, target):
		t = self.test(input)
		e = 0.0
		for i in range(len(t)):
			e += self.sme(target[i],t[i])
		return e

	def sme(self, a, b):
		return 0.5 * ((a - b) ** 2)

	def test(self, input_):
		a = input_
		if self.size[0] == len(input_):
			a = np.concatenate((np.ones(1).T, np.array(input_)), axis = 1)
		for l in range(len(self.w)):
			a = self.sigmoid(np.dot(a, self.w[l]))
		return a

	def load_w(self):
		self.w = np.load('weights.npy')

	def print_to_file(self):
		np.save('weights.npy' , self.w)


