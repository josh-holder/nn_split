import tensorflow as tensorflow

# %matplotlib inline

import logging
import config
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

from SPLT_loss import softmax_cross_entropy_with_logits

import loggers as lg

import keras.backend as K

from settings import run_folder, run_archive_folder

class Gen_Model():
	def __init__(self, reg_const, learning_rate, input_dim, output_dim):
		self.reg_const = reg_const
		self.learning_rate = learning_rate
		self.input_dim = input_dim
		self.output_dim = output_dim

	def predict(self, x):
		return self.model.predict(x)

	def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
		return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split = validation_split, batch_size = batch_size)

	def write(self, game, version):
		self.model.save(run_folder + 'models/version' + "{0:0>4}".format(version) + '.h5')

	def read(self, game, run_number, version):
		return load_model( run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + '.h5', custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

	def printWeightAverages(self):
		layers = self.model.layers
		for i, l in enumerate(layers):
			try:
				x = l.get_weights()[0]
				lg.logger_model.info('WEIGHT LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
			except:
				pass
		lg.logger_model.info('------------------')
		for i, l in enumerate(layers):
			try:
				x = l.get_weights()[1]
				lg.logger_model.info('BIAS LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
			except:
				pass
		lg.logger_model.info('******************')

class Residual_CNN(Gen_Model):
	def __init__(self, reg_const, learning_rate, input_dim,  output_dim, hidden_layers):
		Gen_Model.__init__(self, reg_const, learning_rate, input_dim, output_dim)
		self.hidden_layers = hidden_layers
		self.num_layers = len(hidden_layers)
		self.model = self._build_model()

	def residual_layer(self, input_block, filters, kernel_size):

		x = self.conv_layer(input_block, filters, kernel_size)

		x = Conv2D(
		filters = filters
		, kernel_size = kernel_size
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, data_format='channels_last'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=-1)(x)

		x = add([input_block, x])

		x = LeakyReLU()(x)
		return (x)

	def conv_layer(self, x, filters, kernel_size):

		x = Conv2D(
		filters = filters
		, kernel_size = kernel_size
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, data_format='channels_last'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=-1)(x)
		x = LeakyReLU()(x)
		return (x)

	def value_head(self, x):

		x = Conv2D(
		filters = 1
		, kernel_size = (1,1)
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, data_format='channels_last'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)


		x = BatchNormalization(axis=-1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			1
			, use_bias=False
			, activation='tanh'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'value_head'
			)(x)
		return (x)

	def policy_head(self, x):

		x = Conv2D(
		filters = 2
		, kernel_size = (1,1)
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, data_format='channels_last'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=-1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			self.output_dim
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'policy_head'
			)(x)
		return (x)

	def _build_model(self):
		main_input = Input(shape = self.input_dim, name = 'main_input')

		x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

		if len(self.hidden_layers) > 1:
			for h in self.hidden_layers[1:]:
				x = self.residual_layer(x, h['filters'], h['kernel_size'])

		vh = self.value_head(x)
		ph = self.policy_head(x)

		model = Model(inputs=[main_input], outputs=[vh, ph])
		model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
			optimizer=SGD(lr=self.learning_rate, momentum = config.MOMENTUM),	
			loss_weights={'value_head': 0.5, 'policy_head': 0.5}	
			)

		return model

	# def convertToModelInput(self, state):
	# 	inputToModel =  state.binary #np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
	# 	inputToModel = np.reshape(inputToModel, self.input_dim) 
	# 	return (inputToModel)



