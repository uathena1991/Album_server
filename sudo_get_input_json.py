import os
import pdb
import plistlib
import json
import common_lib as cl
import collections
import numpy as np

def dict2defaultdict(dic):
	def default_factory():
		return 0

	res = collections.defaultdict(default_factory)
	for v in dic:
		try:
			res[v] = dic[v]
		except Exception as e:
			print(str(e))
			pdb.set_trace()
	return res

def get_image_event_scene_label(name, img_path_parent = "/Volumes/working/album_project/album_data/", sepe = 'Event', seps = "Scene_"):
	"""
	Based on image locations (folders names), get image event, label information
	:param name: usr name
	:param image_path: img parent path
	:return:
	img_label: each image's (event, scene),
	scene_dict: (event, scene): [img_names]
	scene_gt: image cluster based on all scenes (no event info)
	"""


	# ground truth
	res = cl.find_all_file_name(os.path.join(img_path_parent, name + "_label_scene"), '.jpg', '')
	img_label = collections.defaultdict()
	for fn, fn_true in res:
		# img_name = fn.split(subfolder_sep)[1].split(os.path.sep)[1]
		event_name = fn.split(sepe)[1].split(os.path.sep)[0]
		try:
			scene_name = fn.split(seps)[1].split(os.path.sep)[0]
		except:
			pdb.set_trace()
		img_label[fn_true] = [event_name, scene_name]
	scene_dict = collections.defaultdict(list)
	for fn_true in img_label:
		scene_dict[img_label[fn_true][0] + '_' + img_label[fn_true][1]].append(fn_true)
	scene_clusters = list(scene_dict.values())
	return img_label, scene_dict, scene_clusters



def get_plist_info(save_usr_nm = 'hxl', plist_path_parent='/Volumes/working/album_project/gps_time_info/',
                   img_path_parent = '/Volumes/working/album_project/album_data'):
	plist_path = os.path.join(plist_path_parent, save_usr_nm)
	## get plist information
	# json_obj =
	fns_gps = cl.find_all_file_name(plist_path, '.plist', 'gps')
	fns_exif = cl.find_all_file_name(plist_path, '.plist', 'exif')
	gps_info = []
	exif_info = []
	info_dict = dict()
	file_names = np.array([f[1].split('-gps')[0] + '.jpg' for f in fns_gps])
	image_cluster_idx, scene_dict, scene_clusters = get_image_event_scene_label(save_usr_nm, img_path_parent)
	for idx, (f, f_true) in enumerate(fns_gps):
		f_name = f.split('gps.')[0]  # including path
		if (f_name + 'exif.plist', f_true.replace('gps', 'exif')) in fns_exif:
			gps_info.append(dict2defaultdict(plistlib.readPlist(f)))
			exif_info.append(dict2defaultdict(plistlib.readPlist(f_name + 'exif.plist')))
			info_dict[file_names[idx]] = dict()
			info_dict[file_names[idx]]['gps_info'] = gps_info[-1]
			info_dict[file_names[idx]]['exif_info'] = exif_info[-1]
			try:
				info_dict[file_names[idx]]['label'] = image_cluster_idx[file_names[idx]]
			except:
				print(file_names[idx], f)
				pdb.set_trace()
			if idx % 500 == 0:
				print("%d files finished" % idx)
	print("Sample length (gps, exif, image name):\n", len(gps_info), len(exif_info), len(file_names))
	# info_dict['scene_dict'] = scene_dict
	# info_dict['scene_clusters'] = scene_clusters
	# pdb.set_trace()
	json_str = json.dumps([info_dict, scene_dict, scene_clusters])
	json_obj = json.loads(json_str)
	# save json object
	with open('/Volumes/working/album_project/serving_data/%s_plist.json' %save_usr_nm, 'w') as file:
		json.dump(json_str, file)
	file.close()

	return json_obj, info_dict


# name_list = ['hxl', 'hxl2016', 'hw', 'zzx', 'zt', 'zd', 'wy_tmp', 'lf']
# for nm in name_list:
# 	get_plist_info(nm)

# get_plist_info('hhl')