import pdb
import os
import ast

import json
import argparse
import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import data_retriever_server as drs
from common_lib import visualize_cluster
import rank_cluster as rcr

def cluster_into_scnenes_time(wanted_time, file_names, min_pic_num, max_pic, thres=0.16, show_idx=False):
	def sub_cluster(wanted_time, file_names, min_pic_num, thres=0.16, show_idx=False):
		wanted_secs = drs.convert_datetime_seconds(wanted_time)
		norm_secs = np.array((wanted_secs - np.mean(wanted_secs)) / (np.std(wanted_secs) + np.array([1e-15, 1e-15, 1e-15])))
		clust = OPTICS(min_pic_num)
		clust.fit(norm_secs)
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
		return np.array([set(x) for x in res_fn]), np.array(img_cl_idx), res_noise, np.array(res_unchosen)

	###############
	to_seperate_list = [file_names]
	res_noise = []
	res_final = []
	res_unchosen = []
	while len(to_seperate_list) > 0:
		tmp_res = []
		for rr in to_seperate_list:
			idx = [np.where(file_names == x)[0][0] for x in rr]
			tmp_cluster, img_cluster_idx, tmp_noise, tmp_unchosen = sub_cluster(wanted_time[idx], file_names[idx], min_pic_num, thres, show_idx)
			tmp_res = np.append(tmp_res, tmp_cluster)
			res_noise = np.append(res_noise, tmp_noise)
			res_unchosen = np.append(res_unchosen, tmp_unchosen)
		to_seperate_list = tmp_res[[len(x) > max_pic for x in tmp_res]]
		res_final = np.append(res_final, tmp_res[[len(x) <= max_pic for x in tmp_res]])
	return res_final, res_noise, res_unchosen


####################################################################
def main_f(FLAGS0):
	global FLAGS, city_table
	FLAGS = FLAGS0
	city_table = (pd.read_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header=0, names=["city"], dtype='str')).T.to_dict('list')
	hol_tab = (pd.read_csv(os.path.join(FLAGS.working_path, FLAGS.holiday_file), header=0, names=["date", "index"],
	                       dtype='str')).set_index('date').T.to_dict('list')

	######################### cluster all scenes into events,  using union find algorithm##########################################
	file_names, gps_info, exif_info, true_label = drs.get_plist_from_json(FLAGS.plist_json)
	wanted_time = np.empty(shape=(len(file_names), 1), dtype=datetime.datetime)
	wanted_holiday = np.empty(shape=(len(file_names), 1), dtype='str')  # holiday
	wanted_closest_holiday = np.empty(shape=(len(file_names), 1), dtype=np.int)  # closest holiday(exclude weekend)
	for idx,ff in enumerate(exif_info):
		wanted_time[idx] = datetime.datetime.strptime(exif_info[idx]['DateTimeDigitized'], '%Y:%m:%d %H:%M:%S')
		wanted_holiday[idx] = hol_tab[wanted_time[idx][0].strftime("%Y%m%d")][0]
		wanted_closest_holiday[idx] = drs.find_closest_holiday(wanted_time[idx][0].date(), hol_tab)
	# print("Clustering images into events based on predictions"

	res_scenes, res_noise, res_unchosen = cluster_into_scnenes_time(wanted_time, file_names, FLAGS.min_pic_num, FLAGS.max_pic, FLAGS.thres_scene, show_idx=False)

	################################## rank ###############################

	# convert to each image's event#, scene # --- since the algorithm has nothing to do with event, default 0.
	# noise: sign different scene #
	predict_label = dict()
	for fn in file_names:
		predict_label[fn] = [-1, -1]
	predict_label = rcr.labeling_image_cluster(predict_label, np.append(res_scenes, res_unchosen), res_noise, 1)

	if not os.path.exists(os.path.join(FLAGS.working_path, FLAGS.feature_save_path,
	                                   'feature_matrix_' + FLAGS.usr_nm.split('/')[0] + '.npz')):
		# print("Warning: feature file not existed!")
		drs.FLAGS.usr_nm = FLAGS.usr_nm.split('/')[0]
		drs.FLAGS.generate_holiday_tab, drs.FLAGS.generate_plist_idx, drs.FLAGS.generate_feature_idx = False, False, False
		drs.FLAGS.train_ratio = 0
		drs.FLAGS.working_path, drs.FLAGS.pic_path_label, drs.FLAGS.image_parent_path, drs.FLAGS.plist_path = FLAGS.working_path, FLAGS.label_pic_path, \
		                                                                                                     FLAGS.image_parent_path, FLAGS.plist_path
		drs.main(1)


	rank_parent_val = [0 for _ in res_scenes]
	sudo_wanted_city_prop = wanted_closest_holiday
	res_final, res_scenes_val, rank_parent_val = rcr.cal_rank_total(0, res_scenes, [], wanted_closest_holiday, file_names, sudo_wanted_city_prop, wanted_holiday,
	                                          predict_label, rank_parent_val, w0 = 1/4, w1 = 1/4, w2 = 1/4, w3 = 0, w4 = 1/4)


	res_noise = np.array([set(res_noise)])

	for rr in res_final:
		print(rr)
	print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	# pdb.set_trace()
	for rr in res_noise:
		print(rr)

	###################### save to json file ########################
		tmp_dict = dict()
	tmp_dict['res_final'] = json.dumps([list(x) for x in res_final])
	tmp_dict['res_noise'] = json.dumps([list(x) for x in res_noise])
	tmp_dict['res_unchosen'] = json.dumps([list(x) for x in res_unchosen])
	tmp_dict['res_scenes_val'] = json.dumps([x for x in res_scenes_val])
	output_fn = FLAGS.output_json
	with open(output_fn, 'w') as file:
		json.dump(json.dumps(tmp_dict), file)
	print('Final cluster results is saved in %s' %output_fn)
	file.close()

	# pdb.set_trace() 2
	################################ Visualize #########################################
	if FLAGS.vis_idx_final:
		visualize_cluster(res_final, os.path.join(FLAGS.image_parent_path, FLAGS.label_pic_path))
	# visualize_cluster(res_noise, os.path.join(FLAGS.image_parent_path, FLAGS.label_pic_path))
# print('All done!')
	######################## calculate accracy, precision, recall #######################
	acc, rec, prec, auc = rcr.cal_accuracy(true_label, predict_label, file_names)
	print("Accuracy: %1.4f" %acc)
	print("Precision: %1.4f" %prec)
	print("Recall: %1.4f" %rec)
	print("AUC: %1.4f" %auc)

	return res_final, res_noise, res_unchosen



if __name__ == "__main__":
	parser = argparse.ArgumentParser("Cluster based on model prediction, rank clusters, and choose given numbers of albums")

	parser.add_argument('--usr_nm', type=str, default='hhl',
	                    help='User name (must remain the same across plist, picture folders')


	parser.add_argument('--vis_idx_final', type=ast.literal_eval, default = False,
	                    help='Bool value: whether to show selected albums.')



	parser.add_argument('--plist_json', type=str,
                    default='serving_data/',
                    help=' Path to the saved plist json file (input)')

	parser.add_argument(
		'--output_json', type=str, default='final_result/',
		help='Model prediction file.')


	parser.add_argument('--plist_path', type=str, default='gps_time_info/',
	                    help=' Path to save plist in the format of .npy (exclude user name)')



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
		help='Bool value: whether to show parsed FLAGS and unparsed values')

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
	FLAGS.output_json = os.path.join(FLAGS.working_path, FLAGS.output_json, FLAGS.usr_nm + '_OPTICS.json')

	res_final, res_noise,res_unchosen = main_f(FLAGS)
	print(res_final)
	print(res_noise)