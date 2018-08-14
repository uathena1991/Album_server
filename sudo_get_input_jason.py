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

def label_each_pic(pic_path: object, file_names: object, file_type: object = '.jpg', subfolder_sep: object = 'Event',
                   check_idx: object = True) -> object:
	"""
    params pic_path: path to the image folders (by users)
    params file_type: image type
    params subfolder_sep: keyword for each subfolder
    params check_idx: whether to check label with image location in the subfolder
    return:
        image_cluster_idx: which event each image belongs to
    """
	res = cl.find_all_file_name(pic_path, file_type, '')
	label_dict = dict()
	for fn, fn_true in res:
		# img_name = fn.split(subfolder_sep)[1].split(os.path.sep)[1]
		img_name = fn_true
		event_name = fn.split(subfolder_sep)[1].split(os.path.sep)[0]
		label_dict[img_name] = event_name
	image_cluster_idx = collections.defaultdict()
	for idx, fn in enumerate(file_names):
		if fn in label_dict:
			image_cluster_idx[fn] = label_dict[fn]
		else:
			image_cluster_idx[fn] = -idx
	# pdb.set_trace()
	print("Pic total: %d, plist total: %d, pic with label total: %d" % (len(res), len(file_names), len(image_cluster_idx)))
	# check label
	if check_idx:
		count = 0
		for key in (image_cluster_idx):
			full_name = os.path.join(pic_path, '%s%s' %(subfolder_sep, image_cluster_idx[key]), key)
			if not os.path.exists(full_name):
				print("Error: %s not exist!!!!" % full_name)
			else:
				count += 1
	return image_cluster_idx


def get_plist_info(plist_path='/Volumes/working/album_project/gps_time_info/wy_tmp/',
                   img_path = '/Volumes/working/album_project/album_data/wy_tmp_label_raw',
                   save_usr_nm = 'wy_tmp'):
	## get plist information
	# json_obj =
	fns_gps = cl.find_all_file_name(plist_path, '.plist', 'gps')
	fns_exif = cl.find_all_file_name(plist_path, '.plist', 'exif')
	gps_info = []
	exif_info = []
	info_dict = dict()
	file_names = np.array([f[1].split('-gps')[0] + '.jpg' for f in fns_gps])
	image_cluster_idx = label_each_pic(img_path, file_names)
	for idx, (f, f_true) in enumerate(fns_gps):
		f_name = f.split('gps.')[0]  # including path
		if (f_name + 'exif.plist', f_true.replace('gps', 'exif')) in fns_exif:
			gps_info.append(dict2defaultdict(plistlib.readPlist(f)))
			exif_info.append(dict2defaultdict(plistlib.readPlist(f_name + 'exif.plist')))
			info_dict[file_names[idx]] = dict()
			info_dict[file_names[idx]]['gps_info'] = gps_info[-1]
			info_dict[file_names[idx]]['exif_info'] = exif_info[-1]
			info_dict[file_names[idx]]['label'] = image_cluster_idx[file_names[idx]]
			if idx % 500 == 0:
				print("%d files finished" % idx)
	print("Sample length (gps, exif, image name:\n", len(gps_info), len(exif_info), len(file_names))
	# pdb.set_trace()
	json_str = json.dumps(info_dict)
	json_obj = json.loads(json_str)
	# save json object
	with open('/Volumes/working/album_project/serving_data/%s_plist.json' %save_usr_nm, 'w') as file:
		json.dump(json_str, file)
	file.close()

	return json_obj, info_dict

get_plist_info()