# -*- coding=utf-8 -*-

"""retrieve feature matrix and labels from plist files, and save those to csv"""

import os
import pdb
import argparse
import time
import ast
import math
import collections
import datetime
import multiprocessing
from datetime import timedelta
import numpy as np
import random

from geopy.geocoders import Nominatim
import pandas as pd
import json
import urllib.request

################################################################################################################################
# convert dict into a defaultdict
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


################################################################################################################################
# convert datetime to seconds
def convert_datetime_seconds(dt_list):
	#  - min_time
	min_time = min(dt_list[dt_list != None])
	dt_list_new = np.empty(shape=(len(dt_list), 1))
	for i in range(len(dt_list)):
		if dt_list[i] != None:
			dt_list_new[i] = ((dt_list[i] - min_time)[0].total_seconds())
	return dt_list_new


################################################################################################################################
def lonlat2xyz(alt, lat, lon):
	return alt * math.cos(lat) * math.sin(lon), alt * math.sin(lat), alt * math.cos(lat) * math.cos(lon)

def altlalong2distance(origin, destination):
	"""
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (alt, lat, long)
    destination : tuple of float
        (alt, lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    # >>> origin = (48.1372, 11.5756)  # Munich
    # >>> destination = (52.5186, 13.4083)  # Berlin
    # >>> round(distance(origin, destination), 1)
    504.2
    """


	alt1, lat1, lon1 = origin
	alt2, lat2, lon2 = destination
	radius = 6371  # km
	x1, y1, z1 = lonlat2xyz(alt1 + radius, lat1, lon1)
	x2, y2, z2 = lonlat2xyz(alt2 + radius, lat2, lon2)

	#     dlat = math.radians(lat2 - lat1)
	#     dlon = math.radians(lon2 - lon1)
	#     x1 = (math.sin(dlat / 2) * math.sin(dlat / 2) +
	#          math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
	#          math.sin(dlon / 2) * math.sin(dlon / 2))
	#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	#     d = radius * c
	return np.linalg.norm(np.array([x1, y1, z1]) - np.array([x2, y2, z2]))


################################################################################################################################


def create_holiday_table(filename, start_date=datetime.date(2009, 1, 1),
                         end_date=datetime.date(2018, 12, 31)):
	"""
    create holiday table from three sources:
    http://www.easybots.cn/api/holiday.php?d=
    http://api.goseek.cn/Tools/holiday?date=
    http://tool.bitefu.net/jiari/?d=
    if they conflict with each other, use juhe to check (100 requests/day)
    :return:  save table to .csv file
    """

	def daterange(start_date, end_date):
		for n in range(int((end_date - start_date).days)):
			yield start_date + datetime.timedelta(n)

	# end_date = datetime.date.today()
	server_url1 = "http://www.easybots.cn/api/holiday.php?d=" # 2010 - 2018
	easy_key = ""
	# server_url2 = "http://api.goseek.cn/Tools/holiday?date="  # after 2017
	server_url3 = "http://tool.bitefu.net/jiari/?d="  # 2009 - 2018
	server_url4 = "http://v.juhe.cn/calendar/day?date="
	juhe_key = "8daf04450c53f5612a0a4de11d219925"
	res = dict()
	failed = []
	idx = 0
	for single_date in daterange(start_date, end_date):
		#     print(single_date.strftime("%Y%m%d"))
		#     pdb.set_trace()
		try:
			# source: easybots
			# vop_data = dict()
			# vop_url_request = urllib.request.Request(server_url1 + single_date.strftime("%Y%m%d"))
			# vop_response = urllib.request.urlopen(vop_url_request)
			# vop_data = json.loads(vop_response.read())
			# tmp1 = vop_data[single_date.strftime("%Y%m%d")]
			# # source2: goseek
			# vop_url_request = urllib.request.Request(server_url2 + single_date.strftime("%Y%m%d"))
			# vop_response = urllib.request.urlopen(vop_url_request)
			# vop_data = json.loads(vop_response.read())
			# tmp2 = vop_data['data']
			# source3: bitefu
			vop_url_request = urllib.request.Request(server_url3 + single_date.strftime("%Y%m%d"))
			vop_response = urllib.request.urlopen(vop_url_request)
			vop_data = json.loads(vop_response.read())
			tmp3 = vop_data
			# if tmp1 == str(tmp3):
			res[single_date.strftime("%Y%m%d")] = tmp3
			time.sleep(0.1)
			# else:
			# 	# source check:
			# 	idx_c += 1
			# 	print(idx_c)
			#
			# 	vop_url_request = urllib.request.Request("%s%s&key=%s" %(server_url4, single_date.strftime("%Y-%-m-%-d"), juhe_key))
			# 	vop_response = urllib.request.urlopen(vop_url_request)
			# 	vop_data = json.loads(vop_response.read())
			# 	if vop_data["reason"] == 'Success':
			# 		tmp_data = vop_data["result"]["data"]
			# 		if "holiday" in tmp_data:
			# 			res[single_date.strftime("%Y%m%d")] = '2'
			# 		elif tmp_data['weekday'] in ["星期六", "星期日"]:
			# 			res[single_date.strftime("%Y%m%d")] = '1'
			# 		else:
			# 			res[single_date.strftime("%Y%m%d")] = "0"


		#         print(single_date.strftime("%Y%m%d"), vop_data['data'])
		#         time.sleep(0.05)
		except Exception as e:
			print(str(e))
			failed.append(single_date.strftime("%Y-%m-%d"))
			idx += 1
			if idx % 100 == 0:
				print("%d %s" % (idx, single_date.strftime("%Y-%m-%d")))
			continue
		idx += 1
		if idx % 100 == 0:
			print("%d %s" % (idx, single_date.strftime("%Y-%m-%d")))
		if idx % 1000 == 0:
			df = pd.DataFrame.from_dict(res, orient='index')
			df.to_csv(filename, mode='w')
	df = pd.DataFrame.from_dict(res, orient='index')
	df.to_csv(filename, mode='w')
	print("%d %s" % (idx, single_date.strftime("%Y-%m-%d")))
	# print(vop_data)

	# if vop_data[date]=='0':
	#     print("this day is weekday")
	# elif vop_data[date]=='1':
	#     print('This day is weekend')
	# elif vop_data[date]=='2':
	#     print('This day is holiday')
	# else:
	#     print('Error!')
	return failed


################################################################################################################################


def find_closest_holiday(date, holi_tab):
	"""

    :param date: datetime.date
    :param holi_tab: dict: date, holiday or not
    :return:
    """
	date_forw = date
	date_back = date
	stat1 = False
	stat2 = False
	while not (stat1 or stat2):
		if holi_tab[date_forw.strftime("%Y%m%d")][0] == '2':
			stat1 = True
		else:
			date_forw += timedelta(days=1)

		if holi_tab[date_back.strftime("%Y%m%d")][0] == '2':
			stat2 = True
		else:
			date_back -= timedelta(days=1)
	return abs((date_forw - date).days) if stat1 else abs((date_back - date).days)


def get_city_from_lonlat(lat, lon):

	def return_format(loc_dict):
		city = ''
		state = ''
		county = ''
		if 'state' in loc_dict:
			state = loc_dict['state']
		if 'city' in loc_dict:
			city = loc_dict['city']
		elif 'state_district' in loc_dict:
			city = loc_dict['state_district']
		if 'county' in loc_dict:
			county = loc_dict['county']
		return [county + ',' + city + ',' + state]

	count = 0

	if str((round(lat, 2), round(lon, 2))) in city_table:
		return city_table[str((round(lat, 2), round(lon, 2)))][0]
	elif (round(lat, 2), round(lon, 2)) == (0.00, 0.00):
		return "Undefined_"+str(random.randint(0,10000))
	else:

		# apikey = "PlGgD24OjLvpOYLPz8NPAOavsL4Hh5hi" # baidu
		apikey = "AIzaSyDANC55ry30yB0b4Tbs5wc76kSo_EXyjcs"  # google
		apikey = "Ywj3Q83cI03E0EjLz5NPumH9uaj2MxyX"  # Nominatim
		apikey = "AmL_Wf6NSGIjl65YC7hHuO46xFIFhxa0jwkfmlESIyI2zOq2Mj0FMb0CioQJWd3w"  # bing
		# geolocator = baidu.Baidu(apikey)
		# geolocator = GoogleV3(apikey)
		geolocator = Nominatim()
		# geolocator = Bing(apikey)
		while count < 10:
			try:
				location = geolocator.reverse("%f, %f" % (lat, lon))
				time.sleep(1)
				# update city_table
				city_table[str((round(lat, 2), round(lon, 2)))] = return_format(location.raw['address'])

				return city_table[str((round(lat, 2), round(lon, 2)))][0]
			except Exception as e:
				print(str(e), lat, lon)
				time.sleep(1)
				# pdb.set_trace()
				df = pd.DataFrame.from_dict(city_table, orient="index")
				df.to_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header = ['city'])
				count += 1
				continue
	print("Failed: %f, %f" % (lat, lon))
	return "Undefined_"+str(random.randint(0,10000))


####################################################################################################################################
def get_plist_from_json(file_path):
	"""

	:param file_names:
	:return: file_names, wanted_gps, wanted_exif, img_label
	"""
	tmp_file = open(file_path, 'r')
	plist_data = json.loads(json.load(tmp_file))
	raw_data = plist_data[0]
	# scene_dict = plist_data[1]
	# scene_clusters = plist_data[2]
	file_names, wanted_gps, wanted_exif = [], [], []
	img_label = dict()
	for key in raw_data:
		file_names.append(key)
		try:
			wanted_gps.append(dict2defaultdict(raw_data[key]['gps_info']))
			wanted_exif.append(dict2defaultdict(raw_data[key]['exif_info']))
			img_label[key] = raw_data[key]['label'] # include event and scene
		except:
			pdb.set_trace()
	return np.array(file_names), np.array(wanted_gps), np.array(wanted_exif), img_label


####################################################################################################################################
def compile_plist_info(file_names, gps_info, exif_info,
                       save_user_name = 'hxl',
                       save_file="/Volumes/working/album_project/album_data/",
                       hol_file="/Volumes/working/album_project/Album/holidays.csv"):
	# holiday information (dict: date--> [index]           0: weekday, 1: weekend, 2:holiday)
	hol_tab = (pd.read_csv(hol_file, header=0, names=["date", "index"],
	                       dtype='str')).set_index('date').T.to_dict('list')


	wanted_gps = np.empty(shape=(len(file_names), 3))  # only gps (3 dimensions)
	wanted_time = np.empty(shape=(len(file_names), 1),
	                       dtype=datetime.datetime)  # only date (9 dimensions: year, month, day, hour, min, sec, wday (Mon - Sun), day in year, daylight savings time
	wanted_exif = np.empty(shape=(len(file_names), 4),
	                       dtype=np.float32)  # exposuretime, flash, focallength, shutterspeedvalue
	# pdb.set_trace()
	wanted_holiday = np.empty(shape=(len(file_names), 1), dtype='str')  # holiday
	wanted_closest_holiday = np.empty(shape=(len(file_names), 1), dtype=np.int)  # closest holiday(exclude weekend)
	wanted_city = []
	wanted_city_prop = np.empty(shape=(len(file_names), 1), dtype=np.float64)  # proportion of city occurance in all cities
	failed_idx = []
	for idx, f_name in enumerate(file_names):

		wanted_gps[idx, 0] = gps_info[idx]['Altitude']
		if gps_info[idx]['LatitudeRef'] == 'N':
			wanted_gps[idx, 1] = gps_info[idx]['Latitude']
		else:
			wanted_gps[idx, 1] = -1 * gps_info[idx]['Latitude']
		if gps_info[idx]['LongitudeRef'] == 'E':
			wanted_gps[idx, 2] = gps_info[idx]['Longitude']
		else:
			wanted_gps[idx, 2] = -1 * gps_info[idx]['Longitude']

		# time.strptime(tmp, "%Y:%m:%d:%H:%M:%S")
		try:
			wanted_time[idx] = datetime.datetime.strptime(exif_info[idx]['DateTimeDigitized'], '%Y:%m:%d %H:%M:%S')
			wanted_holiday[idx] = hol_tab[wanted_time[idx][0].strftime("%Y%m%d")][0]
			wanted_closest_holiday[idx] = find_closest_holiday(wanted_time[idx][0].date(), hol_tab)

			# city (hierarchic cluster with a threshold (50 km)
			wanted_city.append(np.array(get_city_from_lonlat(wanted_gps[idx][1], wanted_gps[idx][2])))

		except Exception as e:
			print("get_plist_info", str(e))
			failed_idx.append(f_name + '.plist')
			continue
		if idx % 500 == 0:
			print("%d files finished" % idx)
	print("%d files finished" % idx)
	wanted_city = np.array(wanted_city)
	unique, counts = np.unique(wanted_city, return_counts=True)
	city_prop_dict = dict(zip(unique, counts / counts.sum()))
	for idx, xx in enumerate(wanted_city):
		wanted_city_prop[idx] = city_prop_dict[wanted_city[idx]]
	print("gps:\n", gps_info[0])
	print("exif:\n", exif_info[0])
	print("image name:\n", file_names[0])
	print("Failed exif: %d" % len(failed_idx))
	# save to file
	np.savez(os.path.join(save_file, 'feature_matrix_' + save_user_name.split('/')[0] + '.npz'), wanted_gps = wanted_gps, wanted_time = wanted_time,
	         wanted_exif = wanted_exif, gps_info = np.array([dict(x) for x in gps_info]),
	         exif_info = np.array([dict(x) for x in exif_info]), file_names = file_names, wanted_holiday = wanted_holiday,
	         wanted_closest_holiday = wanted_closest_holiday, wanted_city = wanted_city, wanted_city_prop = wanted_city_prop)

	# df = pd.DataFrame.from_dict(city_table, orient="index")
	# df.to_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header = ['city'])

	return wanted_gps, wanted_time, wanted_exif, wanted_holiday, wanted_closest_holiday, wanted_city, wanted_city_prop



####################################################################################################################################
def seperate_train_val(filename, train_size=0.98):
	features_m = (pd.read_csv(filename)).values
	np.random.shuffle(features_m)
	# split into training and validation data set
	sep_idx = int(len(features_m) * train_size)
	return features_m[:sep_idx, :], features_m[sep_idx:, :]


####################################################################################################################################
def construct_pair_feature_matrix(file_names, wanted_gps, wanted_time, exif_info, wanted_secs, wanted_holiday, wanted_closest_holiday,
                             wanted_city_prop, image_cluster_idx, save_path, filter_range=168 * 60 * 60):
	"""construct feature matrix
    # $\Delta(distance)$,  $\Delta(time)$[ from exif "DateTimeDigitized", not from gps], $\Delta(ExposureTime)$,
    # $\Delta(flash)$, $\Delta(FocalLength)$, $\Delta(ShutterSpeedValue)$, Holiday_idx, nearest_holiday, city freq,
    # $\Delta(SceneType)$, $\Delta(SensingMethod)$, $\Delta(holiday)$, $\Delta(closest holiday)$, $Average(closest_holiday)$, $\average(city proportion)$
    """
	length = int(len(file_names) * (len(file_names) + 1) / 2)
	print('Total possible pairs: %d' %length)
	dict_list = {('0', '0'): 0, ('1', '1'): 1, ('2', '2'): 2, ('1', '2'): 3, ('2', '1'): 3, ('1', '0'): 4,
             ('0', '1'): 4, ('2', '0'): 5, ('0', '2'): 5}
	## multiprocessing##
	def sub_process(cond, res):
		features_m = np.empty(shape=(0, 16))  # the first two columns are the indexes of the two images, the last two columns are event label, and scene label
		count = 0
		count2 = 0
		len_fn = len(file_names)
		for idx_i,fn_i in enumerate(file_names):
			for idx_j,fn_j in enumerate(file_names):
				count2 += 1
				if  cond[0]*len_fn <= idx_i < (cond[1]*len_fn+1) and idx_j >= idx_i and abs((wanted_time[idx_i] - wanted_time[idx_j])[0].total_seconds()) <= filter_range:
					tmp = np.empty(shape=(1, 16))
					# feature
					tmp[0][:2] = [idx_i, idx_j]  # index of first event, second event
					tmp[0][2] = altlalong2distance(tuple(wanted_gps[idx_i]),
					                            tuple(wanted_gps[idx_j]))  # Eclidean distance
					tmp[0][3] = abs(wanted_secs[idx_i] - wanted_secs[idx_j])  # seconds
					tmp[0][4] = abs(exif_info[idx_i]["ExposureTime"] - exif_info[idx_j]["ExposureTime"])  # Exposure time
					tmp[0][5] = abs(exif_info[idx_i]["Flash"] - exif_info[idx_j]["Flash"])  # flash
					tmp[0][6] = abs(exif_info[idx_i]["FocalLength"] - exif_info[idx_j]["FocalLength"])  # FocalLength
					tmp[0][7] = abs(exif_info[idx_i]["ShutterSpeedValue"] - exif_info[idx_j]["ShutterSpeedValue"])  # ShutterSpeedValue
					tmp[0][8] = abs(exif_info[idx_i]["SceneType"] - exif_info[idx_j]["SceneType"])  # SceneType
					tmp[0][9] = abs(
						exif_info[idx_i]["SensingMethod"] - exif_info[idx_j]["SensingMethod"])  # SensingMethod

					# tmp[0][10]  = hol_tab[wanted_time[idx_i].strtime("%Y%m%d")][0]
					# label
					# holiday 0: (0,0), 1: (1,1), 2:(2,2), 3: (1,2) or (2,1), 4:(1,0) or (0,1), 5:(2,0) or (0,2)

					tmp[0][10] = dict_list[(wanted_holiday[idx_i][0], wanted_holiday[idx_j][0])]
					# holiday delta time
					tmp[0][11] = abs(wanted_closest_holiday[idx_i] - wanted_closest_holiday[idx_j])
					# holiday average time
					tmp[0][12] = (wanted_closest_holiday[idx_i] + wanted_closest_holiday[idx_j]) / 2
					# city average proportion
					tmp[0][13] = (wanted_city_prop[idx_i] + wanted_city_prop[idx_j]) / 2

					tmp[0][14] = 1 if image_cluster_idx[fn_i][0] == image_cluster_idx[fn_j][0] else 0  # event
					tmp[0][15] = 1 if image_cluster_idx[fn_i] == image_cluster_idx[fn_j] else 0  # scene

					features_m = np.concatenate((features_m, tmp), axis=0)
					count += 1
				if count % 5000 == 0  and count != 0:
					print("cond:%s, %d pairs finished, %d pairs passed" %(cond, count, count2))
				if count % 100000 == 0 and count != 0:
					np.save(os.path.join(save_path, 'pair_feature_' + FLAGS.usr_nm + str(cond) + '.npy'), features_m)
		res.append(features_m)
		return features_m

	manager = multiprocessing.Manager()
	res = manager.list()
	num_proc = multiprocessing.cpu_count()
	print("Number of physical cpus:%d" %num_proc)
	p_list = []
	for i in range(num_proc):
		p_list.append(multiprocessing.Process(target = sub_process,args= ([i/num_proc, (i+1)/num_proc], res)))
		p_list[-1].start()
	for pp in p_list:
		pp.join()
## get features_m
	features_m = np.unique(np.concatenate(res), axis = 0)
	np.save(os.path.join(save_path, 'pair_feature_' + FLAGS.usr_nm  + '.npy'), features_m)
	print("Total image pairs: %d, paired_samples: %d" % (length, len(features_m)))
	print("1/0 ratio:", sum(features_m[:, -1] == 1) / len(features_m))
	return features_m


################################################################################################################################
def save2csv(features_m, file_names, usr_nm, train_ratio, save_path, file_type='original', convert_idx_fn=True):
	# (optional) save it to a file
	# pdb.set_trace()
	columns_name = ['1st Image', '2nd Image', 'Distance', 'Time', 'ExposureTime',
	                'Flash', 'FocalLength', 'ShutterSpeedValue', 'SceneType', 'SensingMethod',
	                'Holiday', 'Delta_closest_holiday', 'Average_closest_holiday', 'Average_city_prop', 'Label_e', "Label_s"]
	try:
		df = pd.DataFrame(features_m, columns = columns_name)
		# pdb.set_trace()
		if convert_idx_fn:
			df.loc[:, '1st Image'] = file_names[df['1st Image'].apply(int)]  # convert A to an int
			df.loc[:, '2nd Image'] = file_names[df['2nd Image'].apply(int)]  # convert A to an int

		df.loc[:, 'SceneType'] = df['SceneType'].apply(int)  # convert A to an int
		df.loc[:, 'SensingMethod'] = df['SensingMethod'].apply(int)  # convert A to an int
		df.loc[:, 'Label_e'] = df['Label_e'].apply(int)  # convert A to an int
		df.loc[:, 'Label_s'] = df['Label_s'].apply(int)  # convert A to an int
		df.loc[:, 'Holiday'] = df['Holiday'].apply(int)  # convert A to an int
		df.to_csv(os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio)), header=None, index=False)
		print("Number of %s samples is %d" %(file_type, len(df)))
	except Exception as e:
		print('Error: save %s failed!!!' %file_type)
		print(str(e))
		return os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio))
	return os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio))


################################################################################################################################
def combine_csv(name_list, output_nm, common_path):
	"""
	name_list = ['hxl', 'hw', 'zzx', 'zt', 'zd', 'wy_tmp', 'lf', 'hhl', 'hxl2016']
	:param name_list:
	:param output_nm:
	:param common_path:
	:return:
	"""
	try:
		fout = open(os.path.join(common_path, output_nm), 'a')
		for nm in name_list:
			fnm = open(os.path.join(common_path, nm))
			[fout.write(line) for line in fnm]
		fout.close()
	except Exception as e:
		print(e)
		return False
	return True



################################################################################################################################
def main(FLAGS0):
	try:

		global city_table, FLAGS
		FLAGS = FLAGS0
		if os.path.exists(os.path.join(FLAGS.working_path, FLAGS.city_lonlat)):
			city_table = (pd.read_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header=0, names=["city"], dtype='str')).T.to_dict('list')
		else:
			# city_table = pd.DataFrame({"lonlat":[], "city":[]})
			city_table = dict()
		if FLAGS.generate_holiday_tab:
			print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			print('Getting holiday information (2009-2018)....')
			failed = create_holiday_table(os.path.join(FLAGS.working_path,FLAGS.holiday_file))
			print(failed)
		print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print('Getting plist information....')
		file_names, gps_info, exif_info, img_label = get_plist_from_json(FLAGS.plist_json)
		print("%d files found" %len(file_names))
		print('Done!')
		print('Compile plist information....')
		if FLAGS.generate_plist_idx:
			wanted_gps, wanted_time, wanted_exif, wanted_holiday, wanted_closest_holiday, \
			wanted_city, wanted_city_prop = compile_plist_info(file_names, gps_info, exif_info,
			                                                   FLAGS.usr_nm,
			                                                   os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'feature_matrix'),
			                                                   os.path.join(FLAGS.working_path, FLAGS.holiday_file))
			df = pd.DataFrame.from_dict(city_table, orient="index")
			df.to_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header = ['city'])
		else:
			npz_features = np.load(os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'feature_matrix', 'feature_matrix_' + FLAGS.usr_nm.split('/')[0] + '.npz'))
			wanted_gps = npz_features['wanted_gps']
			wanted_time = npz_features['wanted_time']
			# pdb.set_trace()
			exif_info = [dict2defaultdict(x) for x in npz_features['exif_info']]
			file_names = npz_features['file_names']
			wanted_holiday = npz_features['wanted_holiday']
			wanted_closest_holiday = npz_features['wanted_closest_holiday']
			# wanted_city = npz_features['wanted_city']
			wanted_city_prop = npz_features['wanted_city_prop']
		print('Done!')
		wanted_secs = convert_datetime_seconds(wanted_time)
		print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("Constructing feature matrix (and label)...")
		# pdb.set_trace()
		if FLAGS.generate_feature_idx:
			features_m = construct_pair_feature_matrix(file_names, wanted_gps, wanted_time, exif_info, wanted_secs, wanted_holiday,
			                                           wanted_closest_holiday, wanted_city_prop, img_label,
			                                           os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'pair_feature'),
			                                           FLAGS.filter_range)
			np.save(os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'pair_feature', 'pair_feature_' + FLAGS.usr_nm + '.npy'), features_m)

		else:
			features_m = np.load(os.path.join(FLAGS.working_path, FLAGS.model_input_path, 'pair_feature', 'pair_feature_' + FLAGS.usr_nm + '.npy'))
		print('Done!')
		print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print('Saving to .csv files...')
		original_path = save2csv(features_m, file_names, FLAGS.usr_nm, FLAGS.train_ratio, os.path.join(FLAGS.working_path, FLAGS.model_input_path), 'original')
		train_data, predict_data = seperate_train_val(original_path, FLAGS.train_ratio)
		train_path = save2csv(train_data, file_names, FLAGS.usr_nm, FLAGS.train_ratio, os.path.join(FLAGS.working_path, FLAGS.model_input_path), 'training', False)
		predict_path = save2csv(predict_data, file_names, FLAGS.usr_nm, FLAGS.train_ratio, os.path.join(FLAGS.working_path, FLAGS.model_input_path), 'predict', False)
		print('Done!')
		return original_path, train_path, predict_path
	except Exception as e:
		print(e)
		df = pd.DataFrame.from_dict(city_table, orient="index")
		df.to_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header = ['city'])
		return




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Construct feature matrix (and labels) from plist files.")

	##

	parser.add_argument('--usr_nm', type=str, default='hhl',
	                help='User name for saving files')

	parser.add_argument('--pic_path_label', type=str, default='hhl_label_raw',
	                help='Full path to pictures')

	parser.add_argument('--plist_json', type=str,
	                    # default='/project/album_project/serving_data/hw_plist.json',
	                    default='/Volumes/working/album_project/serving_data/hhl_plist.json',
                    help=' Path to the saved plist json file (input)')


	parser.add_argument('--working_path', type = str,
	                    # default = '/project/album_project/')
	                    default = '/Volumes/working/album_project/')

	parser.add_argument('--image_parent_path', type = str,
	                    # default = '/project/album_project/album_data/')
	                    default = '/Volumes/working/album_project/album_data/')



	parser.add_argument('--model_input_path', type=str, default='preprocessed_data',
	                help='Full path to save the image features(npz) and pair feature(npy), training/testing data (csv)')

	parser.add_argument('--city_lonlat', type=str, default='preprocessed_data/city_lonlat.csv',
	                help='Full path to save and load the excel files (with city-latitude-longitude pairs')

	parser.add_argument("--holiday_file", type = str, default = 'preprocessed_data/holidays.csv',
	                help = "Full path to the holiday lookup table.")



	parser.add_argument('--train_ratio', type=float, default=1,
	                help='Ratio between train/validation samples (use 0 if for test/prediction)')

	parser.add_argument('--filter_range', type=int, default=96 * 60 * 60,
	                help='Time range to choose two images (s)')


	parser.add_argument('--generate_plist_idx', type = ast.literal_eval, default = True,
	                help='if True, generate features from plist info, otherwise, load from the .npy file ')


	parser.add_argument('--generate_feature_idx', type = ast.literal_eval, default = True,
	                help='if True, generate features for two-two compare, otherwise, load from the .npy file ')

	parser.add_argument('--generate_holiday_tab', type = ast.literal_eval, default= False,
	                help='if True, generate holiday table using api')


	##
	global FLAGS
	global city_table
	FLAGS, unparsed = parser.parse_known_args()
	print(FLAGS)
	## add city_longlat lookup table
	if os.path.exists(os.path.join(FLAGS.working_path, FLAGS.city_lonlat)):
		city_table = (pd.read_csv(os.path.join(FLAGS.working_path, FLAGS.city_lonlat), header=0, names=["city"], dtype='str')).T.to_dict('list')
	else:
		# city_table = pd.DataFrame({"lonlat":[], "city":[]})
		city_table = dict()

	original_path, train_path, predict_path = main(FLAGS)
