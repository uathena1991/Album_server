# -*- coding=UTF8 -*-
"""
renference: https://blog.csdn.net/heyc861221/article/details/80131369
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import os, sys
import pdb
import tensorflow as tf
import pandas as pd


def build_model_columns():
	"""Builds a set of wide and deep feature columns."""
	distance = tf.feature_column.numeric_column('Distance')
	time = tf.feature_column.numeric_column('Time')
	expo_time = tf.feature_column.numeric_column('ExposureTime')
	flash = tf.feature_column.numeric_column('Flash')
	focal_len = tf.feature_column.numeric_column('FocalLength')
	shutter = tf.feature_column.numeric_column('ShutterSpeedValue')
	# scene_type = tf.feature_column.categorical_column_with_identity(key = 'SceneType', num_buckets = 2)
	scene_type = tf.feature_column.categorical_column_with_vocabulary_list('SceneType', [0, 1])
	# scene_type = tf.contrib.layers.sparse_column_with_vocabulary_list(column_name="SceneType", keys=[0, 1])
	# sensing_m = tf.feature_column.categorical_column_with_identity(key = 'SensingMethod',  num_buckets = 3)
	sensing_m = tf.feature_column.categorical_column_with_vocabulary_list('SensingMethod',  [0, 1, 2])
	#  0: (0,0), 1: (1,1), 2:(2,2), 3: (1,2) or (2,1), 4:(1,0) or (0,1), 5:(2,0) or (0,2)
	holiday = tf.feature_column.categorical_column_with_vocabulary_list('Holiday',  [0, 1, 2, 3, 4, 5])
	delta_closest_holiday = tf.feature_column.numeric_column('Delta_closest_holiday')
	average_closest_holiday = tf.feature_column.numeric_column('Average_closest_holiday')
	average_city_prop = tf.feature_column.numeric_column('Average_city_prop')

	# wide model
	base_columns = [
		distance,
		 time, expo_time, flash,
		# focal_len, shutter, tf.feature_column.indicator_column(scene_type), tf.feature_column.indicator_column(sensing_m)
		focal_len, shutter, scene_type, sensing_m,
		holiday, delta_closest_holiday, average_closest_holiday
		,average_city_prop
	]
	cross_columns = []
	#cross_columns = [
	#	tf.feature_column.crossed_column(['Distance', 'Time'], hash_bucket_size=1000)
	#]
	wide_columns = base_columns + cross_columns
	# deep model
	deep_columns = [
		distance,
		 time, expo_time, flash,
		focal_len, shutter, tf.feature_column.indicator_column(scene_type), tf.feature_column.indicator_column(sensing_m),
		tf.feature_column.indicator_column(holiday),
		delta_closest_holiday, average_closest_holiday
#, average_city_prop
	]
	return wide_columns, deep_columns


def build_estimator(model_dir, model_type, dnn_learning_rate, linear_learning_rate):
	"""Build an estimator appropriate for the given model type."""
	wide_columns, deep_columns = build_model_columns()
	# hidden_units = [100, 75, 50, 25]
	# hidden_units = [200, 120, 80, 20]
	# hidden_units = [150, 100, 50, 10] # 0.31, 0.48
	hidden_units = [150, 100, 50, 10]
	# hidden_units = [150, 75, 10]
	print("Hidden units:", hidden_units)

	# Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
	#  trains faster than GPU for this model.
	run_config = tf.estimator.RunConfig().replace(
		session_config=tf.ConfigProto(device_count={'GPU': 0}))

	if model_type == 'wide':
		return tf.estimator.LinearClassifier(
            model_dir = model_dir,
            feature_columns = wide_columns,
            config = run_config)
	elif model_type == 'deep':
		return tf.estimator.DNNClassifier(
            model_dir = model_dir,
            feature_columns = deep_columns,
            hidden_units = hidden_units,
            config = run_config)
	else:
		return tf.estimator.DNNLinearCombinedClassifier(
            model_dir = model_dir,
            linear_feature_columns = wide_columns,
            dnn_feature_columns = deep_columns,
            dnn_hidden_units = hidden_units,
			linear_optimizer = tf.train.AdagradOptimizer(learning_rate = linear_learning_rate),
            dnn_optimizer = tf.train.AdagradOptimizer(learning_rate = dnn_learning_rate),
            n_classes = 2,
            config = run_config)

def input_fn(data_file, num_epochs, shuffle, batch_size):
	"""Generate an input function for the Estimator."""
	assert tf.gfile.Exists(data_file), (
			'%s not found.' % data_file)
	def parse_csv(value):
		print('Parsing', data_file)
		columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
		features = dict(zip(_CSV_COLUMNS, columns))
		labels = features.pop('Label_e')
		features.pop('1st Image')
		features.pop('2nd Image')
		return features, tf.equal(labels, 1)

	# Extract lines from input files using the Dataset API.
	dataset = tf.data.TextLineDataset(data_file)
	# reader = tf.WholeFileReader()
	# dataset = reader.read(data_file)

	if shuffle:
		dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
	dataset = dataset.map(parse_csv, num_parallel_calls = 5)

	# We call repeat after shuffling, rather than before, to prevent separate
	#  epochs from blending together.
	dataset = dataset.repeat(num_epochs)
	dataset = dataset.batch(batch_size)

	iterator = dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()
	return features, labels


def main(self):
	# Clean up the model directory if present
	shutil.rmtree(os.path.join(FLAGS.servable_model_dir, FLAGS.model_dir), ignore_errors=True)
	model = build_estimator(os.path.join(FLAGS.servable_model_dir, FLAGS.model_dir), FLAGS.model_type, FLAGS.dnn_learning_rate, FLAGS.linear_learning_rate)

	# Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
	for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
		model.train(input_fn=lambda: input_fn(
			FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))
		results = model.evaluate(input_fn=lambda: input_fn(
			FLAGS.test_data, 1, False, FLAGS.batch_size))

		# Display evaluation metrics
		print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
		print('-' * 60)

		for key in sorted(results):
			print('%s: %s' % (key, results[key]))

	'''Export Trained Model for Serving'''
	wideColumns, DeepColumns = build_model_columns()
	# feature_columns = list(set(wideColumns+ DeepColumns))
	feature_columns = DeepColumns
	feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
	export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
	# servable_model_dir = "/Volumes/working/album_project/census_exported"
	servable_model_path = model.export_savedmodel(FLAGS.servable_model_dir, export_input_fn)
	print("*********** Done Exporting at PAth - %s", servable_model_path )
	print("Run parameters:")
	# pdb.set_trace()
	shutil.move(os.path.join(FLAGS.servable_model_dir, FLAGS.model_dir), os.path.join(servable_model_path.decode(), FLAGS.model_dir))
	print(FLAGS)


if __name__ == '__main__':
	global _CSV_COLUMNS,_CSV_COLUMN_DEFAULTS,FLAGS,_NUM_EXAMPLES

	_CSV_COLUMNS = [
    '1st Image', '2nd Image', 'Distance', 'Time', 'ExposureTime',
    'Flash', 'FocalLength', 'ShutterSpeedValue', 'SceneType','SensingMethod',
	'Holiday', 'Delta_closest_holiday', 'Average_closest_holiday', 'Average_city_prop',
	'Label_e', 'Label_s'
	]

	#_CSV_COLUMN_DEFAULTS = [[0], [0], [0.0], [0.0], [0.0],
	 #                       [0.0], [0.0], [0.0], [0], [0], [0]]
	_CSV_COLUMN_DEFAULTS = [['xx'], ['xx'], [-1.0], [-1.0], [-1.0],
	                       [-1.0], [-1.0], [-1.0], [-1], [-1], [-1], [-1.0], [-1.0], [-1.0], [0], [0]]

	parser = argparse.ArgumentParser()

	parser.add_argument(
	    '--working_path', type=str, default='/project/album_project/',
	    # '--working_path', type=str, default='/Volumes/working/album_project/',
	    help='Base working directory.')

	parser.add_argument(
	    '--model_dir', type=str, default='wide_deep_model/',
	    # '--model_dir', type=str, default='wide_deep_model/',
	    help='Base directory for the model.')

	parser.add_argument(
	    '--train_data', type=str, default= '/project/album_project/preprocessed_data/training/combine_train_0.98.csv',
	    # '--train_data', type=str, default= '/Volumes/working/album_project/preprocessed_data/training/hxl2016_training_data_0.98.csv',
	    help='Path to the training data.')

	parser.add_argument(
	    '--test_data', type=str, default='/project/album_project/preprocessed_data/predict/combine_predict_0.98.csv',
	    # '--test_data', type=str, default='/Volumes/working/album_project/preprocessed_data/test/hxl2016_test_data_0.98.csv',
	    help='Path to the test data.')

	parser.add_argument(
	    '--servable_model_dir', type = str, default = '/project/album_project/model_output/',
	    # '--servable_model_dir', type = str, default = '/Volumes/working/album_project/model_output/',
	    help = 'Path to save the trained model for serving')

	parser.add_argument(
	    '--model_type', type=str, default='wide_deep',
	    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

	parser.add_argument(
	    '--train_epochs', type=int, default = 100, help='Number of training epochs.')

	parser.add_argument(
	    '--epochs_per_eval', type=int, default= 2,
	    help='The number of training epochs to run between evaluations.')

	parser.add_argument(
	    '--batch_size', type=int, default=16, help='Number of examples per batch.')

	parser.add_argument(
	    '--dnn_learning_rate', type=float, default=0.00003, help='DNN Learning rate.')

	parser.add_argument(
	    '--linear_learning_rate', type=float, default=0.004, help='Linear Learning rate.')


	tf.logging.set_verbosity(tf.logging.INFO)
	FLAGS, unparsed = parser.parse_known_args()
	print(FLAGS)
	tmp_train = pd.read_csv(FLAGS.train_data, header = None)
	tmp_test = pd.read_csv(FLAGS.test_data, header = None)

	_NUM_EXAMPLES = {
	    'train': len(tmp_train),
	    'validation': len(tmp_test),
	}
	print(_NUM_EXAMPLES)
	del tmp_test, tmp_train

	tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
