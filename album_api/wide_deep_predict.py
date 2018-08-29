# -*- coding=utf-8 -*-
"""
renference: https://blog.csdn.net/heyc861221/article/details/80131369
prediction from trained model
python wide_deep_predict.py --working_path=/projects/album_project/ --predict_data=preprocessed_data/hxl_test_data_0.00.csv --predict_output=
"""
import os
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd



def main(FLAGS, inputs):
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction=0.4
	print(FLAGS.model_folder_name)
	with tf.Session(config=config) as sess:
		# load the saved model
		tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], os.path.join(FLAGS.working_path, FLAGS.model_exported_path, FLAGS.model_folder_name))
		# get the predictor , refer tf.contrib.predictor
		predictor = tf.contrib.predictor.from_saved_model(os.path.join(FLAGS.working_path, FLAGS.model_exported_path, FLAGS.model_folder_name))

		model_input = []

		for count, row in inputs.iterrows():
			if count % 2000 == 0:
				print("%d cases finished" %count)
			# Read data, using python, into our features
			first, second, distance, sec, day, sec_in_day, delta_time_freq, expo_time, flash, focal_len, shutter, scene_type, sensing_m, \
			holiday, delta_closest_holiday, average_closest_holiday, average_city_prop, label_e, label_s = row
			# Create a feature_dict for train.example - Get Feature Columns using
			if FLAGS.model_type == 'timegps':
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
					'SceneType':_int64_feature(value=int(scene_type)),
					'SensingMethod':_int64_feature(value=int(sensing_m)),
					'Holiday':_int64_feature(value=int(holiday)),
					'Delta_closest_holiday': _float_feature(value=float(delta_closest_holiday)),
					'Average_closest_holiday': _float_feature(value=float(average_closest_holiday)),
					'Average_city_prop': _float_feature(value=float(average_city_prop)),
				}
			elif FLAGS.model_type == 'timeonly':
				feature_dict = {
				'Sec': _float_feature(value=float(sec)),
				'Day': _float_feature(value=int(day)),
				'Sec_in_day': _float_feature(value=float(sec_in_day)),
				'Time_freq': _float_feature(value=float(delta_time_freq)),
				'Holiday':_int64_feature(value=int(holiday)),
				'Delta_closest_holiday': _float_feature(value=float(delta_closest_holiday)),
				'Average_closest_holiday': _float_feature(value=float(average_closest_holiday)),
				}
			else:
				raise('!!!!Error!!! \n Unknown model type %s' %FLAGS.model_type)

			# Prepare model input
			model_input.append(tf.train.Example(features=tf.train.Features(feature=feature_dict)).SerializeToString())

		predictions = predictor({"inputs": model_input})
		predict_label = np.argmax(predictions['scores'], axis = 1)
		predict_prob = np.array([x[i] for i,x in zip(predict_label, predictions['scores'])])
		predict_df = pd.DataFrame([predict_label, predict_prob], index = ['predict_event', 'prob_event']).transpose()
		predict_df['predict_event'] = predict_df['predict_event'].apply(int)
		outputs_df = pd.concat([inputs, predict_df], axis = 1)
	sess.close()
	return outputs_df




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
		# default='/project/album_project/',
		default='/Volumes/working/album_project/',
	    help='Base working directory.')

	parser.add_argument('--usr_nm', type=str, default='zd',
	                help='User name for saving files')

	parser.add_argument('--model_type', type=str, default='timeonly',
	                help='Model type: timegps, or timeonly; Raise error otherwise.')

	parser.add_argument('--model_folder_name', type=str, default='timeonly_Adadelta_L3_noDO_noBN_00003_1',
	                    help='Base directory for the model.')


	parser.add_argument('--predict_input', type=str,
	                    # default='/project/album_project/preprocessed_data/predict/lf_predict_data_0.98.csv',
	                    default='/Volumes/working/album_project/preprocessed_data/predict/zd_predict_data_0.00.csv',
	                help='Full path to the csv files to be predicted')


	parser.add_argument(
	    '--model_exported_path', type=str, default='model_output/',
	    help='Base directory for the model.')

	parser.add_argument(
	    '--prediction_path', type=str, default='model_prediction/',
	    help='Output file for predictions(.cvs).')





	FLAGS, unparsed = parser.parse_known_args()
	# import cProfile
	# cProfile.run('main(FLAGS)', filename="result_single.out", sort="cumulative")
	output_df = main(FLAGS)
