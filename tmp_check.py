import os
import numpy as np
import data_retriever_server as drs
# import rank_cluster as rcr
import common_lib as clb

# file = "/Volumes/working/album_project/serving_data/zzx_plist.json"
# fnames, gps, exif, labels = drs.get_plist_from_json(file)
# eve = dict()
# for k in labels:
# 	if labels[k] not in eve:
# 		eve[labels[k]] = []
# 	eve[labels[k]].append(k)

com = "/Volumes/working/album_project/album_data/hxl2016_label_raw/"
cc = 0
for k in eve:
	path = os.path.join(com, 'Event'+k)
	all_file = clb.find_all_file_name(path)
	if set(eve[k]) > set([x[1] for x in all_file]):
		cc += 1
		print('FALSE:%s' %k)
		print(eve[k])
		print([x[1] for x in all_file])
print(cc)

# rindx = 300
# # print(fnames[rindx])
# # print(gps[rindx])
# # print(exif[rindx])
# # print(labels[fnames[rindx]])
#
# file1 = "/Volumes/working/album_project/preprocessed_data/feature_matrix/feature_matrix_zzx.npz"
# npz_features = np.load(file1)
# wanted_gps = npz_features['wanted_gps']
# wanted_time = npz_features['wanted_time']
# # pdb.set_trace()
# exif_info = [drs.dict2defaultdict(x) for x in npz_features['exif_info']]
# file_names = npz_features['file_names']
# wanted_holiday = npz_features['wanted_holiday']
# wanted_closest_holiday = npz_features['wanted_closest_holiday']
# # wanted_city = npz_features['wanted_city']
# wanted_city_prop = npz_features['wanted_city_prop']
# print(wanted_gps[rindx])
# print(wanted_time[rindx])
# print(exif_info[rindx])
# print(file_names[rindx])
# print(wanted_holiday[rindx])
# print(wanted_closest_holiday[rindx])
# # print(wanted_city_prop[rindx])
