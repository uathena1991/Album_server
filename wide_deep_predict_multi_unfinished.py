# -*- coding=utf-8 -*-
"""
renference: https://blog.csdn.net/heyc861221/article/details/80131369
prediction from trained model
python wide_deep_predict.py --working_path=/projects/album_project/ --predict_data=preprocessed_data/hxl_test_data_0.00.csv --predict_output=
"""
import os
import multiprocessing
import pdb
import argparse
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf



def main(FLAGS):
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.4
	print(FLAGS.model_folder_name)
	output_fn = os.path.join(FLAGS.working_path, FLAGS.prediction_path, "timegps_pred_%s_%s_%s.csv" % (
	FLAGS.usr_nm, FLAGS.model_folder_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
	headers = ['1st Image', '2nd Image', 'Distance', 'Sec', 'Day', 'Sec_in_day', "Delta_time_freq",
	           'ExposureTime', 'Flash', 'FocalLength', 'ShutterSpeedValue', 'SceneType', 'SensingMethod',
	           'Holiday', 'Delta_closest_holiday', 'Average_closest_holiday', 'Average_city_prop', 'Label_e', "Label_s"]
	inputs = pd.read_csv(FLAGS.predict_input, header=None, index_col=False, names=headers)
	len_input = len(inputs)

	###################################################
	def sub_process(cond, res):
		subinputs = inputs[int(cond[0] * len_input):int(cond[1] * len_input)].reset_index(drop = True)
		sess =  tf.Session(config=config)
		# load the saved model
		tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
		                           os.path.join(FLAGS.working_path, FLAGS.model_exported_path,
		                                        FLAGS.model_folder_name))
		predictor = tf.contrib.predictor.from_saved_model(os.path.join(FLAGS.working_path, FLAGS.model_exported_path, FLAGS.model_folder_name))

		# get the predictor , refer tf.contrib.predictor
		model_input = []
		for count, row in subinputs.iterrows():
			if count % 2000 == 0 and count != 0:
				print("%d cases finished" % count)
			# Read data, using python, into our features
			first, second, distance, sec, day, sec_in_day, delta_time_freq, expo_time, flash, focal_len, shutter, scene_type, sensing_m, \
			holiday, delta_closest_holiday, average_closest_holiday, average_city_prop, label_e, label_s = row
			# true_label.append(int(label_e))
			# Create a feature_dict for train.example - Get Feature Columns using
			feature_dict = {
				'Distance': _float_feature(value=float(distance)),
				'Sec': _float_feature(value=float(sec)),
				'Day': _float_feature(value=int(day)),
				'Sec_in_day': _float_feature(value=float(sec_in_day)),
				'Delta_time_freq': _float_feature(value=float(delta_time_freq)),
				'ExposureTime': _float_feature(value=float(expo_time)),
				'Flash': _float_feature(value=float(flash)),
				'FocalLength': _float_feature(value=float(focal_len)),
				'ShutterSpeedValue': _float_feature(value=float(shutter)),
				'SceneType': _int64_feature(value=int(scene_type)),
				'SensingMethod': _int64_feature(value=int(sensing_m)),
				'Holiday': _int64_feature(value=int(holiday)),
				'Delta_closest_holiday': _float_feature(value=float(delta_closest_holiday)),
				'Average_closest_holiday': _float_feature(value=float(average_closest_holiday)),
				'Average_city_prop': _float_feature(value=float(average_city_prop)),
			}

			# Prepare model input
			model_input.append(
				tf.train.Example(features=tf.train.Features(feature=feature_dict)).SerializeToString())
		predictions = predictor({"inputs": model_input})
		predict_label = np.argmax(predictions['scores'], axis=1)
		predict_prob = np.array([x[i] for i, x in zip(predict_label, predictions['scores'])])
		predict_df = pd.DataFrame([predict_label, predict_prob], index=['predict_label', 'probability']).transpose()
		predict_df['predict_label'] = predict_df['predict_label'].apply(int)
		outputs_df = pd.concat([subinputs, predict_df], axis = 1,  ignore_index = True)
		# sess.close()
		res.append(outputs_df)
		sess.close()
		return outputs_df

	###########################################################################################################
	manager = multiprocessing.Manager()
	res = manager.list()
	num_proc = multiprocessing.cpu_count()
	print("Number of physical cpus:%d" % num_proc)

	p_list = []
	for i in range(num_proc):
		p_list.append(multiprocessing.Process(target=sub_process, args=([i/num_proc, (i+1)/num_proc], res)))
		p_list[-1].start()
	pdb.set_trace()
	for pp in p_list:
		pp.join()
	combined_df = pd.concat(res, axis=0, ignore_index = True)
	new_headers = headers + ['predict_label', 'probability']
	combined_df.columns = new_headers
	combined_df.to_csv(output_fn, header=new_headers, index=False)

	# analyze precision, recall and so
	true_label = combined_df['Label_e']
	predict_label = combined_df['predict_label']
	x = tf.placeholder(tf.int32)
	y = tf.placeholder(tf.int32)
	acc, acc_op = tf.metrics.accuracy(labels=x, predictions=y)
	prec, prec_op = tf.metrics.precision(labels=x, predictions=y)
	rec, rec_op = tf.metrics.recall(labels=x, predictions=y)
	auc, auc_op = tf.metrics.auc(labels=x, predictions=y)

	sess = tf.InteractiveSession(config=config)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	print("--------------------W&D model prediction results (Event)-------------------------")
	print(FLAGS.model_folder_name)
	print('Number of samples:', len(true_label))
	v1 = sess.run([acc, acc_op], feed_dict={x: true_label, y: predict_label})
	print("Accuracy", v1[-1])
	v2 = sess.run([prec, prec_op], feed_dict={x: true_label, y: predict_label})
	print('Precision', v2[-1])
	v3 = sess.run([rec, rec_op], feed_dict={x: true_label, y: predict_label})
	print('Recall', v3[-1])
	v4 = sess.run([auc, auc_op], feed_dict={x: true_label, y: predict_label})
	print('AUC', v4[-1])
	sess.close()

	return len(true_label), v1[-1], v2[-1], v3[-1], v4[-1], output_fn


def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--working_path', type=str,
		default='/project/album_project/',
		# default='/Volumes/working/album_project/',
		help='Base working directory.')

	parser.add_argument('--usr_nm', type=str, default='zd',
	                    help='User name for saving files')

	parser.add_argument('--predict_input', type=str,
	                    default='/project/album_project/preprocessed_data/predict/zd_predict_data_0.00.csv',
	                    # default='/Volumes/working/album_project/preprocessed_data/predict/zd_predict_data_0.00.csv',
	                    help='Full path to the csv files to be predicted')

	parser.add_argument(
		'--model_exported_path', type=str, default='model_output/',
		help='Base directory for the model.')
	parser.add_argument(
		'--model_folder_name', type=str, default='new_timegps_Adadelta_L3_noDO_noBN_00003_004_0',
		help='Base directory for the model.')

	parser.add_argument(
		'--prediction_path', type=str, default='model_prediction/',
		help='Output file for predictions(.cvs).')

	FLAGS, unparsed = parser.parse_known_args()
	# import cProfile
	# cProfile.run('main(FLAGS)', filename="result_multi.out", sort="cumulative")
	import time
	t1 = time.time()
	cts, acc, prec, rec, auc, output_fn = main(FLAGS)
	t2 = time.time()
	print(t2 - t1)
