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
import ast

def build_model_columns():
	"""Builds a set of wide and deep feature columns."""
	# distance = tf.feature_column.numeric_column('Distance')
	sec = tf.feature_column.numeric_column('Sec')
	day = tf.feature_column.numeric_column("Day")
	sec_in_day = tf.feature_column.numeric_column("Sec_in_day")
	time_freq = tf.feature_column.numeric_column("Time_freq")



	holiday = tf.feature_column.categorical_column_with_vocabulary_list('Holiday',  [0, 1, 2, 3, 4, 5])
	delta_closest_holiday = tf.feature_column.numeric_column('Delta_closest_holiday')
	average_closest_holiday = tf.feature_column.numeric_column('Average_closest_holiday')
	# average_city_prop = tf.feature_column.numeric_column('Average_city_prop')

	# wide model
	base_columns = [
		# distance,
		# sec, day, sec_in_day, time_freq,
	#	expo_time, flash, focal_len, shutter, scene_type, sensing_m,
		# holiday, delta_closest_holiday, average_closest_holiday
		# ,average_city_prop
	]

	cross_columns = []
	#cross_columns = [
	#	tf.feature_column.crossed_column(['Distance', 'Time'], hash_bucket_size=1000)
	#]
	wide_columns = base_columns + cross_columns
	# deep model
	deep_columns = [
		# distance,
		sec, day, sec_in_day, time_freq,
		# expo_time, flash, focal_len, shutter, tf.feature_column.indicator_column(scene_type), tf.feature_column.indicator_column(sensing_m),
		tf.feature_column.indicator_column(holiday), delta_closest_holiday, average_closest_holiday
#, average_city_prop
	]
	return wide_columns, deep_columns


def build_estimator(model_dir, model_type, dnn_learning_rate, linear_learning_rate, dnn_dropout):
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
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = linear_learning_rate),
            config = run_config)
	elif model_type == 'deep':
		return tf.estimator.DNNClassifier(
            model_dir = model_dir,
            feature_columns = deep_columns,
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate = dnn_learning_rate), # error
            # optimizer = tf.train.AdamOptimizer(learning_rate = dnn_learning_rate),
            # optimizer = tf.train.RMSPropOptimizer(learning_rate = dnn_learning_rate), # best??
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate = dnn_learning_rate), #
            optimizer = tf.train.AdagradOptimizer(learning_rate = dnn_learning_rate), # default, stable, don't need to adjust params
            hidden_units = hidden_units,
			n_classes = 2,
			dropout= dnn_dropout,
			# batch_norm = True,
            config = run_config)
	else:
		return tf.estimator.DNNLinearCombinedClassifier(
            model_dir = model_dir,
            linear_feature_columns = wide_columns,
            dnn_feature_columns = deep_columns,
            dnn_hidden_units = hidden_units,
            linear_optimizer = tf.train.GradientDescentOptimizer(learning_rate = linear_learning_rate),
            # dnn_optimizer = tf.train.GradientDescentOptimizer(learning_rate = dnn_learning_rate),
            dnn_optimizer = tf.train.AdadeltaOptimizer(learning_rate = dnn_learning_rate),
            n_classes = 2,
			dropout = dnn_dropout,
			# bath_norm = True,
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
		# labels = features['Label_e']
		features.pop('Label_s')
		features.pop('1st Image')
		features.pop('2nd Image')
		features.pop('ExposureTime')
		features.pop("Flash")
		features.pop("FocalLength")
		features.pop("ShutterSpeedValue")
		features.pop("SceneType")
		features.pop("SensingMethod")
		print("LENGTH", len(features))
		# pdb.set_trace()
		return features, tf.equal(labels, 1)

	# Extract lines from input files using the Dataset API.
	dataset = tf.data.TextLineDataset(data_file)
	# reader = tf.WholeFileReader()
	# dataset = reader.read(data_file)
	# pdb.set_trace()
	if shuffle:
	# if False:
		dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
	dataset = dataset.map(parse_csv, num_parallel_calls = 1)

	# We call repeat after shuffling, rather than before, to prevent separate
	#  epochs from blending together.
	dataset = dataset.repeat(num_epochs)
	dataset = dataset.batch(batch_size)

	iterator = dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()
	# print(features, labels)
	# pdb.set_trace()
	# ss = tf.InteractiveSession()
	# print(ss.run([features, labels]))
	return features, labels


def main(self):
	# Clean up the model directory if present
	shutil.rmtree(os.path.join(FLAGS.servable_model_dir, FLAGS.model_dir), ignore_errors=True)
	model = build_estimator(os.path.join(FLAGS.servable_model_dir, FLAGS.model_dir), FLAGS.model_type,
	                        FLAGS.dnn_learning_rate, FLAGS.linear_learning_rate, FLAGS.dnn_dropout)

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
	print("*********** Done Exporting at PAth - %s", servable_model_path)
	print("Run parameters:")
	# pdb.set_trace()
	shutil.move(servable_model_path.decode(), os.path.join(FLAGS.servable_model_dir, FLAGS.model_rename))
	shutil.move(os.path.join(FLAGS.servable_model_dir, FLAGS.model_dir), os.path.join(FLAGS.servable_model_dir, FLAGS.model_rename))
	print(FLAGS)


if __name__ == '__main__':
	global _CSV_COLUMNS,_CSV_COLUMN_DEFAULTS,FLAGS,_NUM_EXAMPLES

	_CSV_COLUMNS = [
    '1st Image', '2nd Image', 'Sec', 'Day', 'Sec_in_day', "Time_freq",
    'ExposureTime', 'Flash', 'FocalLength', 'ShutterSpeedValue', 'SceneType', 'SensingMethod',
    'Holiday', 'Delta_closest_holiday', 'Average_closest_holiday',
    'Label_e', "Label_s"
	]

	#_CSV_COLUMN_DEFAULTS = [[0], [0], [0.0], [0.0], [0.0],
	 #                       [0.0], [0.0], [0.0], [0], [0], [0]]
	_CSV_COLUMN_DEFAULTS = [['xx'], ['xx'], [-1.0], [-1.0], [-1.0], [-1.0],
	                       [-1.0], [-1.0], [-1.0], [-1.0], [-1], [-1],
	                        [-1], [-1.0], [-1.0],
	                        [1], [1]]

	parser = argparse.ArgumentParser()

	parser.add_argument(
	    # '--servable_model_dir', type = str, default = '/project/album_project/model_output/',
	    '--model_rename', type = str, default = 'tmp/',
	    help = 'Path to rename the trained model for serving')

	parser.add_argument(
	    '--working_path', type=str, default='/project/album_project/',
	    # '--working_path', type=str, default='/Volumes/working/album_project/',
	    help='Base working directory.')

	parser.add_argument(
	    '--model_dir', type=str, default='deep_model/',
	    # '--model_dir', type=str, default='wide_deep_model/',
	    help='Base directory for the model.')

	parser.add_argument('--dnn_dropout', type=float, default=0.0, help='DNN dropout rate.')

	parser.add_argument('--dnn_batch_norm', type=ast.literal_eval, default=False, help='DNN batch normalization.')

	parser.add_argument(
	    '--train_data', type=str, default= '/project/album_project/preprocessed_data/timeonly/training/timeonly_combine_training_0.98.csv',
	    # '--train_data', type=str, default= '/Volumes/working/album_project/preprocessed_data/timeonly/training/timeonly_combine_training_0.98.csv',
	    help='Path to the training data.')

	parser.add_argument(
	    '--test_data', type=str, default='/project/album_project/preprocessed_data/timeonly/predict/timeonly_combine_predict_0.98.csv',
	    # '--test_data', type=str, default='/Volumes/working/album_project/preprocessed_data/timeonly/predict/timeonly_combine_predict_0.98.csv',
	    help='Path to the test data.')

	parser.add_argument(
	    '--servable_model_dir', type = str, default = '/project/album_project/model_output/',
	    # '--servable_model_dir', type = str, default = '/Volumes/working/album_project/model_output/',
	    help = 'Path to save the trained model for serving')



	parser.add_argument(
	    '--model_type', type=str, default='deep',
	    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

	parser.add_argument(
	    '--train_epochs', type=int, default = 100, help='Number of training epochs.')

	parser.add_argument(
	    '--epochs_per_eval', type=int, default= 2,
	    help='The number of training epochs to run between evaluations.')

	parser.add_argument(
	    '--batch_size', type=int, default=16, help='Number of examples per batch.')

	parser.add_argument(
	    '--dnn_learning_rate', type=float, default=0.0003, help='DNN Learning rate.')

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
