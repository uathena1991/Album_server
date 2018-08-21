import pdb
import os
import ast

import argparse
import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import shutil

import data_retriever_server as drs
import common_lib as clb
import rank_cluster as rcr

def cluster_into_scnenes_timegps(wanted_time, wanted_gps, file_names, min_pic_num, max_pic, thres=0.16, show_idx=False):
	def sub_cluster(wanted_time, wanted_gps, file_names, min_pic_num, thres=0.16, show_idx=False):
		wanted_xyz = [drs.lonlat2xyz(x[0], x[1], x[2]) for x in wanted_gps]
		norm_xyz = np.array((wanted_xyz - np.mean(wanted_xyz, 0)) / (np.std(wanted_xyz, 0) + np.array([1e-15, 1e-15, 1e-15])))
		wanted_secs = drs.convert_datetime_seconds(wanted_time)
		norm_secs = np.array((wanted_secs - np.mean(wanted_secs)) / (np.std(wanted_secs) + np.array([1e-15, 1e-15, 1e-15])))
		norm_info = np.array([np.array([x[0], x[1], x[2], y[0]]) for x, y in zip(norm_xyz, norm_secs)])

		clust = OPTICS(min_pic_num)
		clust.fit(norm_info)
		img_cl_idx, res_eps, min_noise = rcr.find_best_thres(clust, len(wanted_secs))
		# plotting
		if show_idx:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			title = "threshold: %f, number of clusters: %d" % (thres, len(np.unique(img_cl_idx)))
			ax.set_title(title)
			plt.show()
		# find file names in each cluster:
		res_fn = [[] for _ in np.unique(np.append(img_cl_idx, [-1]))]
		for idx, cl_idx in enumerate(img_cl_idx):
			try:
				res_fn[cl_idx + 1].append(file_names[idx])
			except:
				pdb.set_trace()
		res_noise = res_fn[0]
		res_fn = np.array(res_fn[1:])
		res_unchosen = res_fn[[len(x) < min_pic_num for x in res_fn]]
		res_fn = res_fn[[len(x) >= min_pic_num for x in res_fn]]
		return np.array([set(x) for x in res_fn]), np.array(img_cl_idx), res_noise, np.array([set(x) for x in res_unchosen])

	###############
	to_seperate_list = [file_names]
	res_noise = []
	res_final = []
	res_unchosen = []
	while len(to_seperate_list) > 0:
		tmp_res = []
		tmp_unchosen = []
		for rr in to_seperate_list:
			idx = [np.where(file_names == x)[0][0] for x in rr]
			tmp_cluster, img_cluster_idx, tmp_noise, tmp_unchosen = sub_cluster(wanted_time[idx], wanted_gps[idx], file_names[idx], min_pic_num, thres, show_idx)
			tmp_res = np.append(tmp_res, tmp_cluster)
			res_noise = np.append(res_noise, tmp_noise)
			tmp_unchosen = np.append(tmp_unchosen, res_unchosen)
		to_seperate_list = tmp_res[[len(x) > max_pic for x in tmp_res]]
		res_final = np.append(res_final, tmp_res[[len(x) <= max_pic for x in tmp_res]])
		res_unchosen = np.append(res_unchosen, tmp_unchosen)
	return res_final, res_noise, res_unchosen


####################################################################
def main_f(FLAGS0):
	global FLAGS, city_table
	FLAGS = FLAGS0
	city_table = (pd.read_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header=0, names=["city"], dtype='str')).T.to_dict('list')
	hol_tab = (pd.read_csv(os.path.join(FLAGS.working_path, FLAGS.holiday_file), header=0, names=["date", "index"],
	                       dtype='str')).set_index('date').T.to_dict('list')

	######################### Get events ##########################################
	file_names, gps_info, exif_info, img_label = drs.get_plist_from_json(FLAGS.plist_json)
	wanted_time = np.empty(shape=(len(file_names), 1), dtype=datetime.datetime)
	wanted_gps = np.empty(shape=(len(file_names), 3))  # only gps (3 dimensions)
	wanted_holiday = np.empty(shape=(len(file_names), 1), dtype='str')  # holiday
	wanted_closest_holiday = np.empty(shape=(len(file_names), 1), dtype=np.int)  # closest holiday(exclude weekend)
	for idx,ff in enumerate(exif_info):
		wanted_time[idx] = datetime.datetime.strptime(exif_info[idx]['DateTimeDigitized'], '%Y:%m:%d %H:%M:%S')
		wanted_gps[idx, 0] = gps_info[idx]['Altitude']
		if gps_info[idx]['LatitudeRef'] == 'N':
			wanted_gps[idx, 1] = gps_info[idx]['Latitude']
		else:
			wanted_gps[idx, 1] = -1 * gps_info[idx]['Latitude']
		if gps_info[idx]['LongitudeRef'] == 'E':
			wanted_gps[idx, 2] = gps_info[idx]['Longitude']
		else:
			wanted_gps[idx, 2] = -1 * gps_info[idx]['Longitude']
		wanted_holiday[idx] = hol_tab[wanted_time[idx][0].strftime("%Y%m%d")][0]
		wanted_closest_holiday[idx] = drs.find_closest_holiday(wanted_time[idx][0].date(), hol_tab)
	# convert image label to events

	res = clb.find_all_file_name(os.path.join(FLAGS.image_parent_path, FLAGS.usr_nm + "_label_raw"), '.jpg', '')
	subfolder_sep = 'Event'
	events_dict = dict()
	for fn, fn_true in res:
		# img_name = fn.split(subfolder_sep)[1].split(os.path.sep)[1]
		event_name = subfolder_sep + fn.split(subfolder_sep)[1].split(os.path.sep)[0]
		if event_name not in events_dict:
			events_dict[event_name] = [fn_true]
		else:
			events_dict[event_name].append(fn_true)
	# pdb.set_trace()
	new_path = os.path.join(FLAGS.image_parent_path, FLAGS.usr_nm + '_label_scene')
	for x in events_dict:
		if not os.path.exists(os.path.join(new_path, x)):
			os.makedirs(os.path.join(new_path, x))
		curr_files =set(events_dict[x]).intersection(set(file_names))
		miss_files = set(events_dict[x]).difference(set(file_names))
		if len(curr_files) > FLAGS.max_pic:
			idx = [np.where(file_names == y)[0][0] for y in curr_files]
			res_scenes, res_noise, res_unchosen = cluster_into_scnenes_timegps(wanted_time[idx], wanted_gps[idx], file_names[idx], FLAGS.min_pic_num, FLAGS.max_pic, FLAGS.thres_scene, show_idx=False)
			# move images to subfolder according to the seperation
			if len(res_noise) > 0:
				if not os.path.exists(os.path.join(new_path, x, 'Scene_noise')):
					os.makedirs(os.path.join(new_path, x, 'Scene_noise'))
				for fn in res_noise:
					if not os.path.exists(os.path.join(new_path, x, 'Scene_noise', fn)):
						shutil.copy(os.path.join(FLAGS.image_parent_path, FLAGS.usr_nm + "_label_raw", x, fn),
					            os.path.join(new_path, x, 'Scene_noise', fn))
			if len(res_unchosen) > 0:
				for idx, cluster in enumerate(res_unchosen):
					if not os.path.exists(os.path.join(new_path, x, 'Scene_unchosen_' + str(idx))):
						os.makedirs(os.path.join(new_path, x, 'Scene_unchosen_' + str(idx)))
					for fn in cluster:
						if not os.path.exists(os.path.join(new_path, x, 'Scene_unchosen_' + str(idx), fn)):
							shutil.copy(os.path.join(FLAGS.image_parent_path, FLAGS.usr_nm + "_label_raw", x, fn),
							            os.path.join(new_path, x, 'Scene_unchosen_'+ str(idx), fn))

			for i, cluster in enumerate(res_scenes):
				if not os.path.exists(os.path.join(new_path, x, 'Scene_'+str(i))):
					os.makedirs(os.path.join(new_path, x, 'Scene_'+str(i)))
				for fn in cluster:
					if not os.path.exists(os.path.join(new_path, x, 'Scene_'+str(i), fn)):
						shutil.copy(os.path.join(FLAGS.image_parent_path, FLAGS.usr_nm + "_label_raw", x, fn),
						            os.path.join(new_path, x, 'Scene_'+str(i), fn))
		else:
			if len(curr_files) > 0:
				if not os.path.exists(os.path.join(new_path,x,'Scene_0')):
					os.makedirs(os.path.join(new_path,x,'Scene_0'))
				for fn in curr_files:
					if not os.path.exists(os.path.join(new_path, x, 'Scene_0', fn)):
						shutil.copy(os.path.join(FLAGS.image_parent_path, FLAGS.usr_nm + "_label_raw", x, fn),
						            os.path.join(new_path, x, 'Scene_0', fn))
			if len(miss_files) > 0:
				str_miss = "Scene_miss"
				if os.path.exists(os.path.join(new_path, x, 'Scene_0')):
					str_miss = 'Scene_0'
				if not os.path.exists(os.path.join(new_path,x,str_miss)):
					os.makedirs(os.path.join(new_path,x,str_miss))
				for fn in miss_files:
					if not os.path.exists(os.path.join(new_path, x, str_miss, fn)):
						shutil.copy(os.path.join(FLAGS.image_parent_path, FLAGS.usr_nm + "_label_raw", x, fn),
						            os.path.join(new_path, x, str_miss, fn))





if __name__ == "__main__":
	parser = argparse.ArgumentParser("Cluster based on model prediction, rank clusters, and choose given numbers of albums")

	parser.add_argument('--usr_nm', type=str, default='hhl',
	                    help='User name (must remain the same across plist, picture folders')


	parser.add_argument('--vis_idx_final', type=ast.literal_eval, default = True,
	                    help='Bool value: whether to show selected albums.')



	parser.add_argument('--plist_json', type=str,
                    default='serving_data/',
                    help=' Path to the saved plist json file (input)')




	parser.add_argument(
		'--working_path', type=str, default='/Volumes/working/album_project/',
		help='working dir for the project.')

	parser.add_argument('--model_input_path', type=str, default='preprocessed_data',
	                help='Full path to save the image features(npz) and pair feature(npy),  training/testing data (csv)')



	parser.add_argument(
		'--image_parent_path', type=str, default='/Volumes/working/album_project/album_data/',
		help='Model prediction file.')
	parser.add_argument('--feature_save_path', type=str, default='preprocessed_data/feature_matrix',
                    help=' Path to save plist in the format of .npy (exclude user name)')

	parser.add_argument('--min_pic_num', type=int, default = 10,
	                    help='Minimum numbers of pics in target clusters')

	parser.add_argument('--max_album', type=int, default= 50,
	                    help='Max numbers of albums needed')
	parser.add_argument('--max_pic', type=int, default = 30,
	                    help='Max numbers of albums needed')
	parser.add_argument('--thres_scene', type=float, default = 0.16,
	                    help='threshold(normalized) for cluster small scenes')




	parser.add_argument(
		'--print_parser', type=ast.literal_eval, default=False,
		help='Bool value: whether to show parsed FLAGS and unparsed valfues')

	parser.add_argument('--city_lonlat', type=str, default='preprocessed_data/city_lonlat.csv',
	                help='Full path to save and load the excel files (with city-latitude-longitude pairs')

	parser.add_argument("--holiday_file", type = str, default = 'preprocessed_data/holidays.csv',
	                help = "Full path to the holiday lookup table.")

	FLAGS, unparsed = parser.parse_known_args()

	if FLAGS.print_parser:
		print("FLAGS", FLAGS)
		print("unparsed", unparsed)
	FLAGS.label_pic_path = FLAGS.usr_nm + '_label_raw'
	FLAGS.plist_json = os.path.join(FLAGS.working_path, FLAGS.plist_json, FLAGS.usr_nm + '_plist.json')

	main_f(FLAGS)