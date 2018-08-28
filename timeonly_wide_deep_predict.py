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
import datetime



def main(FLAGS):
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction=0.4
	print(FLAGS.model_folder_name)
	with tf.Session(config=config) as sess:
		# load the saved model
		tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], os.path.join(FLAGS.working_path, FLAGS.model_exported_path, FLAGS.model_folder_name))
		
		# get the predictor , refer tf.contrib.predictor
		predictor = tf.contrib.predictor.from_saved_model(os.path.join(FLAGS.working_path, FLAGS.model_exported_path, FLAGS.model_folder_name))

		output_fn = os.path.join(FLAGS.working_path, FLAGS.prediction_path, "timeonly_pred_%s_%s_%s.csv" %(FLAGS.usr_nm, FLAGS.model_folder_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
		# input_fn = os.path.join(FLAGS.working_path, FLAGS.model_input_path, "%s_training_")
		prediction_OutFile = open(output_fn, 'w')
		
		#Write Header for naCSV file
		prediction_OutFile.write("first, second, Sec, Day, Sec_in_day, Delta_time_freq, holiday, delta_closest_holiday, average_closest_holiday, true_label, predicted_label, probability")
		prediction_OutFile.write('\n')
		
		# Read file and create feature_dict for each record
		true_label = []
		predict_label = []
		with open(FLAGS.predict_input) as inf:
			# Skip header
			next(inf)
			count = 0
			for line in inf:
				if count % 2000 == 0:
					print("%d cases finished" %count)
				# Read data, using python, into our features
				first, second, distance, sec, day, sec_in_day, delta_time_freq, expo_time, flash, focal_len, shutter, scene_type, sensing_m, \
				holiday, delta_closest_holiday, average_closest_holiday, average_city_prop, label_e, label_s = line.strip().split(",")
				true_label.append(int(label_e))
				# Create a feature_dict for train.example - Get Feature Columns using
				feature_dict = {
					'Sec': _float_feature(value=float(sec)),
					'Day': _float_feature(value=int(day)),
					'Sec_in_day': _float_feature(value=float(sec_in_day)),
					'Delta_time_freq': _float_feature(value=float(delta_time_freq)),
					'Holiday':_int64_feature(value=int(holiday)),
					'Delta_closest_holiday': _float_feature(value=float(delta_closest_holiday)),
					'Average_closest_holiday': _float_feature(value=float(average_closest_holiday)),
				}
				
				# Prepare model input
				
				model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
				
				model_input = model_input.SerializeToString()
				output_dict = predictor({"inputs": [model_input]})
				
				# print(" prediction Label is ", output_dict['classes'])
				# print('Probability : ' + str(output_dict['scores']))
				
				# Positive label = 1
				prediction_OutFile.write(first + "," + second + ',' + str(sec)+ "," + str(day)+ "," + str(sec_in_day) + "," + str(delta_time_freq) + "," +
				                         str(holiday) +  "," + str(delta_closest_holiday) + "," + str(average_closest_holiday) + "," +
				                         str(label_e) + ",")
				label_index = np.argmax(output_dict['scores'])
				prediction_OutFile.write(str(label_index))
				prediction_OutFile.write(',')
				prediction_OutFile.write(str(output_dict['scores'][0][label_index]))
				prediction_OutFile.write('\n')
				predict_label.append(label_index)

				count += 1

	
	prediction_OutFile.close()
	sess.close()
	# analyze precision, recall and so
	# pdb.set_trace()
	x = tf.placeholder(tf.int32)
	y = tf.placeholder(tf.int32)
	acc, acc_op = tf.metrics.accuracy(labels = x, predictions = y)
	prec, prec_op = tf.metrics.precision(labels = x, predictions = y)
	rec, rec_op = tf.metrics.recall(labels = x, predictions = y)
	auc, auc_op = tf.metrics.auc(labels = x, predictions = y)

	sess = tf.InteractiveSession(config=config)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	print("--------------------W&D model prediction results (Event)-------------------------")
	print('Number of samples:', len(true_label))
	v1 = sess.run([acc, acc_op], feed_dict={x: true_label, y: predict_label})
	print("Accuracy", v1[-1])
	v2= sess.run([prec, prec_op], feed_dict={x: true_label, y: predict_label})
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
		# default='/project/album_project/',
		default='/Volumes/working/album_project/',
	    help='Base working directory.')

	parser.add_argument('--usr_nm', type=str, default='hxl',
	                help='User name for saving files')

	parser.add_argument('--predict_input', type=str,
	                    # default='/project/album_project/preprocessed_data/timeonly/predict/lf_predict_data_0.98.csv',
	                    default='/Volumes/working/album_project/preprocessed_data/timeonly/original/hxl_original_data_0.98.csv',
	                help='Full path to the csv files to be predicted')

	parser.add_argument(
	    '--model_exported_path', type=str, default='model_output/timeonly_????_0',
	    help='Base directory for the model.')

	parser.add_argument(
	'--model_folder_name', type=str, default='timeonly_Adadelta_L3_noDO_noBN_1',
	help='Base directory for the model.')

	parser.add_argument(
	    '--prediction_path', type=str, default='model_prediction/',
	    help='Output file for predictions(.cvs).')





	FLAGS, unparsed = parser.parse_known_args()
	cts, acc, prec, rec, auc, output_fn = main(FLAGS)
