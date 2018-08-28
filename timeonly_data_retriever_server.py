import os
import ast
import multiprocessing
import argparse
import datetime
import pdb
import numpy as np
import pandas as pd

import data_retriever_server as drs


def cal_time_freq(time_list):
	if len(time_list) <= 1:
		return 1
	delta_time = max(time_list) - min(time_list)
	return len(time_list) / max(delta_time[0].total_seconds(), 1)


def construct_pair_feature_matrix(file_names, usr_nm, wanted_time, time_freq, exif_info, wanted_secs, wanted_holiday,
                                  wanted_closest_holiday,
                                  image_cluster_idx, save_path, filter_range=96 * 60 * 60):
	"""construct feature matrix
    # $\Delta(time)$[ from exif "DateTimeDigitized", not from gps],
    delta(day), delta(time in a day),
    delta(time_freq),
    $\Delta(ExposureTime)$,
    # $\Delta(flash)$, $\Delta(FocalLength)$, $\Delta(ShutterSpeedValue)$,
    # $\Delta(SceneType)$, $\Delta(SensingMethod)$,
    $\Delta(holiday)$, $\Delta(closest holiday)$, $Average(closest_holiday)$
    """
	length = int(len(file_names) * (len(file_names) + 1) / 2)
	print('Total possible pairs: %d' % length)
	dict_list = {('0', '0'): 0, ('1', '1'): 1, ('2', '2'): 2, ('1', '2'): 3, ('2', '1'): 3, ('1', '0'): 4,
	             ('0', '1'): 4, ('2', '0'): 5, ('0', '2'): 5}

	## multiprocessing##
	def sub_process(cond, res):
		features_m = np.empty(shape=(0,
		                             17))  # the first two columns are the indexes of the two images, the last two columns are event label, and scene label
		count = 0
		count2 = 0
		len_fn = len(file_names)
		for idx_i, fn_i in enumerate(file_names):
			for idx_j, fn_j in enumerate(file_names):
				count2 += 1
				if cond[0] * len_fn <= idx_i < (cond[1] * len_fn + 1) and idx_j > idx_i and abs(
						(wanted_time[idx_i] - wanted_time[idx_j])[0].total_seconds()) <= filter_range:
					tmp = np.empty(shape=(1, 17))
					# feature
					tmp[0][:2] = [idx_i, idx_j]  # index of first event, second event
					# tmp[0][2] = drs.altlalong2distance(tuple(wanted_gps[idx_i]),
					#                             tuple(wanted_gps[idx_j]))  # Eclidean distance
					tmp[0][2] = abs(wanted_secs[idx_i] - wanted_secs[idx_j])  # seconds
					# pdb.set_trace()
					tmp[0][3] = abs(wanted_time[idx_i][0].date() - wanted_time[idx_j][0].date()).days  # day
					# print(wanted_time[idx_i][0].date(), wanted_time[idx_j][0].date(), tmp[0][3])
					# print(wanted_time[idx_i][0].time(), wanted_time[idx_j][0].time(), tmp[0][3])
					tmp[0][4] = abs(datetime.datetime.combine(datetime.date.min, wanted_time[idx_i][0].time()) -
					                datetime.datetime.combine(datetime.date.min, wanted_time[idx_j][
						                0].time())).total_seconds()  # seconds in a day
					# print(wanted_time[idx_i][0].time(), wanted_time[idx_j][0].time(), tmp[0][4]/3600)
					tmp[0][5] = abs(time_freq[idx_i] - time_freq[idx_j])

					tmp[0][6] = abs(
						exif_info[idx_i]["ExposureTime"] - exif_info[idx_j]["ExposureTime"])  # Exposure time
					tmp[0][7] = abs(exif_info[idx_i]["Flash"] - exif_info[idx_j]["Flash"])  # flash
					tmp[0][8] = abs(exif_info[idx_i]["FocalLength"] - exif_info[idx_j]["FocalLength"])  # FocalLength
					tmp[0][9] = abs(exif_info[idx_i]["ShutterSpeedValue"] - exif_info[idx_j][
						"ShutterSpeedValue"])  # ShutterSpeedValue
					tmp[0][10] = abs(exif_info[idx_i]["SceneType"] - exif_info[idx_j]["SceneType"])  # SceneType
					tmp[0][11] = abs(
						exif_info[idx_i]["SensingMethod"] - exif_info[idx_j]["SensingMethod"])  # SensingMethod

					# tmp[0][10]  = hol_tab[wanted_time[idx_i].strtime("%Y%m%d")][0]
					# label
					# holiday 0: (0,0), 1: (1,1), 2:(2,2), 3: (1,2) or (2,1), 4:(1,0) or (0,1), 5:(2,0) or (0,2)

					tmp[0][12] = dict_list[(wanted_holiday[idx_i][0], wanted_holiday[idx_j][0])]
					# holiday delta time
					tmp[0][13] = abs(wanted_closest_holiday[idx_i] - wanted_closest_holiday[idx_j])
					# holiday average time
					tmp[0][14] = (wanted_closest_holiday[idx_i] + wanted_closest_holiday[idx_j]) / 2
					# city average proportion
					# tmp[0][14] = (wanted_city_prop[idx_i] + wanted_city_prop[idx_j]) / 2

					tmp[0][15] = 1 if image_cluster_idx[fn_i][0] == image_cluster_idx[fn_j][0] else 0  # event label
					tmp[0][16] = 1 if image_cluster_idx[fn_i] == image_cluster_idx[fn_j] else 0  # scene label

					features_m = np.concatenate((features_m, tmp), axis=0)
					count += 1
				if count % 5000 == 0 and count != 0:
					print("cond:%s, %d pairs finished, %d pairs passed" % (cond, count, count2))
				if count % 100000 == 0 and count != 0:
					np.save(os.path.join(save_path, 'time_pair_feature_' + usr_nm + str(cond) + '.npy'), features_m)
		res.append(features_m)
		return features_m

	manager = multiprocessing.Manager()
	res = manager.list()
	num_proc = multiprocessing.cpu_count()
	print("Number of physical cpus:%d" % num_proc)
	p_list = []
	for i in range(num_proc):
		p_list.append(multiprocessing.Process(target=sub_process, args=([i / num_proc, (i + 1) / num_proc], res)))
		p_list[-1].start()
	for pp in p_list:
		pp.join()
	## get features_m
	features_m = np.unique(np.concatenate(res), axis=0)
	np.save(os.path.join(save_path, 'time_pair_feature_' + usr_nm + '.npy'), features_m)
	print("Total image pairs: %d, paired_samples: %d" % (length, len(features_m)))
	print("1/0 ratio:", sum(features_m[:, -2] == 1) / len(features_m))
	return features_m


def save2csv(features_m, file_names, usr_nm, train_ratio, save_path, file_type='original', convert_idx_fn=True):
	# (optional) save it to a file
	# pdb.set_trace()
	columns_name = ['1st Image', '2nd Image', 'Sec', 'Day', 'Sec_in_day', "Time_freq",
	                'ExposureTime', 'Flash', 'FocalLength', 'ShutterSpeedValue', 'SceneType', 'SensingMethod',
	                'Holiday', 'Delta_closest_holiday', 'Average_closest_holiday',
	                'Label_e', "Label_s"]
	try:
		df = pd.DataFrame(features_m, columns=columns_name)
		# pdb.set_trace()
		if convert_idx_fn:
			df.loc[:, '1st Image'] = file_names[df['1st Image'].apply(int)]  # convert A to an int
			df.loc[:, '2nd Image'] = file_names[df['2nd Image'].apply(int)]  # convert A to an int
		# pdb.set_trace()
		df.loc[:, 'SceneType'] = df['SceneType'].apply(int)  # convert A to an int
		df.loc[:, 'SensingMethod'] = df['SensingMethod'].apply(int)  # convert A to an int

		df.loc[:, 'Label_e'] = df['Label_e'].apply(int)  # convert label event to an int
		df.loc[:, 'Label_s'] = df['Label_s'].apply(int)  # convert label scene to an int
		df.loc[:, 'Holiday'] = df['Holiday'].apply(int)  # convert holiday to an int

		df.to_csv(os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" % (usr_nm, file_type, train_ratio)),
		          header=None, index=False)
		print("Number of %s samples is %d" % (file_type, len(df)))
	except Exception as e:
		print('Error: save %s failed!!!' % file_type)
		print(str(e))
		return os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" % (usr_nm, file_type, train_ratio))
	return os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" % (usr_nm, file_type, train_ratio))


def main(FLAGS):
	"""usr_nm, train_ratio = 0.98, filter_range = 3*24 * 60 * 60 , generate_feature_idx = False, half_win_size = 2,
         working_path = "/Volumes/working/album_project/",
         plist_path = "serving_data",
         feature_path = "preprocessed_data/feature_matrix",
         save_path = "preprocessed_data/pair_feature",
         model_input_path = "preprocessed_data/timeonly"):"""
	try:
		if FLAGS.generate_holiday_tab:
			print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			print('Getting holiday information (2009-2018)....')
			failed = drs.create_holiday_table(os.path.join(FLAGS.working_path,FLAGS.holiday_file))
			print(failed)

		# load image label
		file_names, gps_info, exif_info, img_label = drs.get_plist_from_json(
			os.path.join(FLAGS.working_path, FLAGS.plist_folder, FLAGS.usr_nm + "_plist.json"))

		# load feature matrix
		if FLAGS.generate_plist_idx:
			wanted_gps, wanted_time, wanted_exif, wanted_holiday, wanted_closest_holiday, \
			wanted_city, wanted_city_prop = drs.compile_plist_info(file_names, gps_info, exif_info,
			                                                   FLAGS.usr_nm,
			                                                   os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'feature_matrix'),
			                                                   os.path.join(FLAGS.working_path, FLAGS.holiday_file))
			df = pd.DataFrame.from_dict(city_table, orient="index")
			df.to_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header = ['city'])
		else:
			npz_features = np.load(os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'feature_matrix', 'feature_matrix_' + FLAGS.usr_nm.split('/')[0] + '.npz'))
			wanted_time = npz_features['wanted_time']
			# pdb.set_trace()
			exif_info = [drs.dict2defaultdict(x) for x in npz_features['exif_info']]
			file_names = npz_features['file_names']
			wanted_holiday = npz_features['wanted_holiday']
			wanted_closest_holiday = npz_features['wanted_closest_holiday']
			# wanted_city = npz_features['wanted_city']
		print('Done!')
		wanted_secs = drs.convert_datetime_seconds(wanted_time)
		# pdb.set_trace()


		# calc img frequency in time
		sort_time_idx = sorted(range(len(wanted_time)), key=lambda k: wanted_time[k])
		sort_wtime = sorted(wanted_time)
		time_freq0 = [cal_time_freq(
			sort_wtime[max(0, idx - FLAGS.half_win_size): min(len(sort_wtime) - 1, idx + FLAGS.half_win_size + 1)])
		              for idx in range(len(sort_wtime))]
		time_freq = np.copy(time_freq0)
		for idx, x in enumerate(sort_time_idx):
			time_freq[x] = time_freq0[idx]

		# create paired feature
		if FLAGS.generate_feature_idx:
			features_m = construct_pair_feature_matrix(file_names, FLAGS.usr_nm, wanted_time, time_freq, exif_info,
			                                           wanted_secs, wanted_holiday,
			                                           wanted_closest_holiday, img_label,
			                                           os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'pair_feature'),
			                                           FLAGS.filter_range)
		else:
			features_m = np.load(
				os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'pair_feature',  'time_pair_feature_' + FLAGS.usr_nm + '.npy'))
		print('Done!')
		print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print('Saving to .csv files...')
		original_path = save2csv(features_m, file_names, FLAGS.usr_nm, FLAGS.train_ratio,
		                         os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'timeonly'), 'original')
		train_data, predict_data = drs.seperate_train_val(original_path, FLAGS.train_ratio)
		train_path = save2csv(train_data, file_names, FLAGS.usr_nm, FLAGS.train_ratio,
		                      os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'timeonly'), 'training', False)
		predict_path = save2csv(predict_data, file_names, FLAGS.usr_nm, FLAGS.train_ratio,
		                        os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'timeonly'), 'predict', False)
		print('Done!')
		return original_path, train_path, predict_path
	except Exception as e:
		print(e)
		return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Construct feature matrix (and labels) from plist files.")


	parser.add_argument('--usr_nm', type=str, default='zd',
	                    help='User name for saving files')

	parser.add_argument('--working_path', type=str,
                    # default = '/project/album_project/')
                    default='/Volumes/working/album_project/')

	# parser.add_argument('--pic_path_label', type=str, default='_label_raw',
	#                     help='Full path to pictures')

	parser.add_argument('--plist_folder', type=str,
	                    # default='/project/album_project/serving_data/hw_plist.json',
	                    default='serving_data/',
	                    help=' Path to the saved plist json file (input)')




	parser.add_argument('--model_input_path', type=str, default='preprocessed_data',
	                    help='Full path to save the image features(npz) and pair feature(npy), training/testing data (csv)')

	parser.add_argument("--holiday_file", type=str, default='preprocessed_data/holidays.csv',
	                    help="Full path to the holiday lookup table.")
	parser.add_argument('--city_lonlat', type=str, default='preprocessed_data/city_lonlat.csv',
	                help='Full path to save and load the excel files (with city-latitude-longitude pairs')


	parser.add_argument('--train_ratio', type=float, default=0.0,
	                    help='Ratio between train/validation samples (use 0 if for test/prediction)')

	parser.add_argument('--filter_range', type=int, default=96 * 60 * 60,
	                    help='Time range to choose two images (s)')

	parser.add_argument('--half_win_size', type=int, default= 2,
            help='Time window to calc time freq')

	parser.add_argument('--generate_plist_idx', type=ast.literal_eval, default=False,
	                    help='if True, generate features from plist info, otherwise, load from the .npy file ')

	parser.add_argument('--generate_feature_idx', type=ast.literal_eval, default=False,
	                    help='if True, generate features for two-two compare, otherwise, load from the .npy file ')

	parser.add_argument('--generate_holiday_tab', type=ast.literal_eval, default=False,
	                    help='if True, generate holiday table using api')

	##
	global FLAGS
	global city_table
	FLAGS, unparsed = parser.parse_known_args()
	print(FLAGS)

	original_path, train_path, predict_path = main(FLAGS)
