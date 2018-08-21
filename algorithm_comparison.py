import os
import json
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import common_lib as clb
# import collections

from sudo_get_input_json import get_image_event_scene_label

# % pdb on
def cal_similarity(set_a, set_b):
	# return (len(set_a.intersection(set_b))/(1e-16+ len(set_b)) + len(set_b.intersection(set_a))/(1e-6+len(set_b)))/2
	return len(set_a.intersection(set_b))/(len(set_b)+1e-6)


def func_compare(alg, gt, vis):
	""" Compare algorithm with ground truth, alg, gt
	return: precision, recall, F1 score
	"""

	matrix_similar = np.array([[cal_similarity(set(xa), set(xb)) for xb in gt] for xa in alg])
	paired_gt = [[] for _ in alg]
	for i in range(len(alg)):
		try:
			alg_idx, gt_idx = np.unravel_index(matrix_similar.argmax(),matrix_similar.shape)
		except:
			pdb.set_trace()
		paired_gt[alg_idx] = gt[gt_idx]
		matrix_similar[alg_idx, gt_idx] = 0

	matrix_similar1 = np.array([[cal_similarity(set(xa), set(xb)) for xb in paired_gt] for xa in alg])
	if vis:
		print("len(gt)=%d, len(opt) = %d" %(len(gt), len(alg)))
		plt.imshow(matrix_similar1)
		plt.show()
	# calculate accuracy, recall, precision, auc

	rec, prec, f1 = [],[],[]
	for idx, calg in enumerate(alg):
		inters = set(calg).intersection(set(paired_gt[idx]))
		rec.append(len(inters)/(1e-6+len(paired_gt[idx])))
		prec.append(len(inters)/(1e-6+len(calg)))
		f1.append(2*prec[-1]*rec[-1]/(prec[-1]+rec[-1] + 1e-6))
	# [print(x,y,z) for x,y,z in zip(prec, rec, f1)]
	return np.mean(prec), np.mean(rec), np.mean(f1)


# def get_image_event_scene_label(name, image_path = "/Volumes/working/album_project/album_data/"):
# 	"""
# 	Based on image locations (folders names), get image event, label information
# 	:param name: usr name
# 	:param image_path: img parent path
# 	:return:
# 	img_label: each image's (event, scene),
# 	scene_dict: (event, scene): [img_names]
# 	scene_gt: image cluster based on all scenes (no event info)
# 	"""
#
#
# 	# ground truth
# 	res = clb.find_all_file_name(os.path.join(image_path, name + "_label_scene"), '.jpg', '')
# 	img_label = collections.defaultdict()
# 	for fn, fn_true in res:
# 		# img_name = fn.split(subfolder_sep)[1].split(os.path.sep)[1]
# 		event_name = fn.split("Event")[1].split(os.path.sep + 'Scene')[0]
# 		try:
# 			scene_name = fn.split("Scene_")[1].split(os.path.sep)[0]
# 		except:
# 			pdb.set_trace()
# 		img_label[fn_true] = (event_name, scene_name)
# 	scene_dict = collections.defaultdict(list)
# 	for fn_true in img_label:
# 		scene_dict[img_label[fn_true]].append(fn_true)
# 	scene_gt = list(scene_dict.values())
# 	return img_label, scene_dict, scene_gt







def main():
	name_list = ['hxl', 'hxl2016', 'hw', 'zzx', 'zt', 'zd', 'wy_tmp', 'lf', 'hhl']
	res_wdl_opt = np.empty(shape = (3, len(name_list)))
	res_gt_wdl = np.empty(shape = (3, len(name_list)))
	res_gt_opt= np.empty(shape = (3, len(name_list)))
	vis = False
	common_path = "/Volumes/working/album_project/final_result/"
	for idx,name in enumerate(name_list):
		img_label, scene_dict, scene_gt = get_image_event_scene_label(name)
		# compare two algorithms directly
		tmp_file = open(os.path.join(common_path, name + "_WDL.json"), 'r')
		wdl = json.loads(json.load(tmp_file))
		tmp_file = open(os.path.join(common_path, name + "_OPTICS.json"), 'r')
		optics = json.loads(json.load(tmp_file))
		wdl_scene = json.loads(wdl['res_final'])
		opt_scene = json.loads(optics['res_final'])
		res_wdl_opt[:, idx] = func_compare(opt_scene, wdl_scene, vis)
		print("WDL (as GT) vs opt:\n %s: precision: %1.2f, recall: %1.2f, F1 score: %1.2f\n" %(name, res_wdl_opt[0, idx], res_wdl_opt[1, idx], res_wdl_opt[2, idx]))
		# wdl and ground truth
		res_gt_wdl[:, idx] = func_compare(wdl_scene, scene_gt, vis)
		print("WDL vs Ground Truth:\n %s: precision: %1.2f, recall: %1.2f, F1 score: %1.2f\n" %(name, res_gt_wdl[0, idx], res_gt_wdl[1, idx], res_gt_wdl[2, idx]))
		# opt and ground truth
		res_gt_opt[:, idx] = func_compare(opt_scene, scene_gt, vis)
		print("OPTICS vs Ground Truth:\n %s: precision: %1.2f, recall: %1.2f, F1 score: %1.2f\n\n" %(name, res_gt_opt[0, idx], res_gt_opt[1, idx], res_gt_opt[2, idx]))

	# save results to a table
	df_wdl_opt = pd.DataFrame(res_wdl_opt)
	df_gt_wdl = pd.DataFrame(res_gt_wdl)
	df_gt_opt = pd.DataFrame(res_gt_opt)
	writer = pd.ExcelWriter(os.path.join(common_path, 'two_algorithms_evaluation.xlsx'), engine='xlsxwriter')

	# Write each dataframe to a different worksheet.
	df_wdl_opt.to_excel(writer, sheet_name='WDL_OPT', header = name_list, index= True,  index_label = {'Precision', "Recall", "F1 score"})
	df_gt_wdl.to_excel(writer, sheet_name='GT_WDL', header = name_list, index= True,  index_label = {'Precision', "Recall", "F1 score"})
	df_gt_opt.to_excel(writer, sheet_name='GT_OPT', header = name_list, index= True, index_label = {'Precision', "Recall", "F1 score"})

	# Close the Pandas Excel writer and output the Excel file.
	writer.save()
main()
