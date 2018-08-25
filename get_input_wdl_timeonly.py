import os
import datetime
import pdb
import numpy as np
import pandas as pd
import data_retriever_server as drs
import multiprocessing

def cal_harmonic_mean(time_list):
    if len(time_list) <= 1:
        return 1
    delta_time = max(time_list) - min(time_list)
    return len(time_list)/max(delta_time[0].total_seconds(), 1)


def construct_pair_feature_matrix(file_names, usr_nm, wanted_time, time_freq, exif_info, wanted_secs, wanted_holiday, wanted_closest_holiday,
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
    print('Total possible pairs: %d' %length)
    dict_list = {('0', '0'): 0, ('1', '1'): 1, ('2', '2'): 2, ('1', '2'): 3, ('2', '1'): 3, ('1', '0'): 4,
             ('0', '1'): 4, ('2', '0'): 5, ('0', '2'): 5}
    ## multiprocessing##
    def sub_process(cond, res):
        features_m = np.empty(shape=(0, 17))  # the first two columns are the indexes of the two images, the last two columns are event label, and scene label
        count = 0
        count2 = 0
        len_fn = len(file_names)
        for idx_i,fn_i in enumerate(file_names):
            for idx_j,fn_j in enumerate(file_names):
                count2 += 1
                if  cond[0]*len_fn <= idx_i < (cond[1]*len_fn+1) and idx_j > idx_i and abs((wanted_time[idx_i] - wanted_time[idx_j])[0].total_seconds()) <= filter_range:
                    tmp = np.empty(shape=(1, 17))
                    # feature
                    tmp[0][:2] = [idx_i, idx_j]  # index of first event, second event
                    # tmp[0][2] = drs.altlalong2distance(tuple(wanted_gps[idx_i]),
                    #                             tuple(wanted_gps[idx_j]))  # Eclidean distance
                    tmp[0][2] = abs(wanted_secs[idx_i] - wanted_secs[idx_j])  # seconds
                    # pdb.set_trace()
                    tmp[0][3] = abs(wanted_time[idx_i][0].date() - wanted_time[idx_j][0].date()).days # day
                    # print(wanted_time[idx_i][0].date(), wanted_time[idx_j][0].date(), tmp[0][3])
                    # print(wanted_time[idx_i][0].time(), wanted_time[idx_j][0].time(), tmp[0][3])
                    tmp[0][4] = abs(datetime.datetime.combine(datetime.date.min, wanted_time[idx_i][0].time()) -
                                    datetime.datetime.combine(datetime.date.min, wanted_time[idx_j][0].time())).total_seconds() # seconds in a day
                    # print(wanted_time[idx_i][0].time(), wanted_time[idx_j][0].time(), tmp[0][4]/3600)
                    tmp[0][5] = abs(time_freq[idx_i] - time_freq[idx_j])

                    tmp[0][6] = abs(exif_info[idx_i]["ExposureTime"] - exif_info[idx_j]["ExposureTime"])  # Exposure time
                    tmp[0][7] = abs(exif_info[idx_i]["Flash"] - exif_info[idx_j]["Flash"])  # flash
                    tmp[0][8] = abs(exif_info[idx_i]["FocalLength"] - exif_info[idx_j]["FocalLength"])  # FocalLength
                    tmp[0][9] = abs(exif_info[idx_i]["ShutterSpeedValue"] - exif_info[idx_j]["ShutterSpeedValue"])  # ShutterSpeedValue
                    tmp[0][10] = abs(exif_info[idx_i]["SceneType"] - exif_info[idx_j]["SceneType"])  # SceneType
                    tmp[0][11] = abs(exif_info[idx_i]["SensingMethod"] - exif_info[idx_j]["SensingMethod"])  # SensingMethod

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
                if count % 5000 == 0  and count != 0:
                    print("cond:%s, %d pairs finished, %d pairs passed" %(cond, count, count2))
                if count % 100000 == 0 and count != 0:
                    np.save(os.path.join(save_path, 'time_pair_feature_' + usr_nm + str(cond) + '.npy'), features_m)
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
    np.save(os.path.join(save_path, 'time_pair_feature_' + usr_nm  + '.npy'), features_m)
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
        df = pd.DataFrame(features_m, columns = columns_name)
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

        df.to_csv(os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio)), header=None, index=False)
        print("Number of %s samples is %d" %(file_type, len(df)))
    except Exception as e:
        print('Error: save %s failed!!!' %file_type)
        print(str(e))
        return os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio))
    return os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio))


def main(usr_nm, train_ratio = 0.98, filter_range = 3*24 * 60 * 60 , gen_feature_idx = False, half_win_size = 2,
         working_path = "/Volumes/working/album_project/",
         plist_path = "serving_data",
         feature_path = "preprocessed_data/feature_matrix",
         save_path = "preprocessed_data/pair_feature",
         model_input_path = "preprocessed_data/timeonly"):
    try:
        # load image label
        file_names, gps_info, exif_info, img_label = drs.get_plist_from_json(os.path.join(working_path, plist_path, usr_nm + "_plist.json"))
        # load feature matrix
        npz_features = np.load(os.path.join(working_path, feature_path, "feature_matrix_" + usr_nm + ".npz"))
        wanted_time = npz_features['wanted_time']
        exif_info = [drs.dict2defaultdict(x) for x in npz_features['exif_info']]
        file_names = npz_features['file_names']
        wanted_holiday = npz_features['wanted_holiday']
        wanted_closest_holiday = npz_features['wanted_closest_holiday']

        wanted_secs = drs.convert_datetime_seconds(wanted_time)
        # pdb.set_trace()
        # calc img frequency in time
        sort_time_idx = sorted(range(len(wanted_time)), key=lambda k: wanted_time[k])
        sort_wtime = sorted(wanted_time)
        time_freq0 = [cal_harmonic_mean(sort_wtime[max(0, idx - half_win_size): min(len(sort_wtime) -1, idx + half_win_size+1)])
                     for idx in range(len(sort_wtime))]
        time_freq = np.copy(time_freq0)
        for idx, x in enumerate(sort_time_idx):
            time_freq[x] = time_freq0[idx]
        # create paired feature
        if gen_feature_idx:
            features_m = construct_pair_feature_matrix(file_names, usr_nm, wanted_time, time_freq, exif_info, wanted_secs, wanted_holiday,
                                                       wanted_closest_holiday, img_label, os.path.join(working_path, save_path), filter_range)
        else:
            features_m = np.load(os.path.join(working_path, save_path, 'time_pair_feature_' + usr_nm + '.npy'))
        print('Done!')
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print('Saving to .csv files...')
        original_path = save2csv(features_m, file_names, usr_nm, train_ratio, os.path.join(working_path, model_input_path), 'original')
        train_data, predict_data = drs.seperate_train_val(original_path, train_ratio)
        train_path = save2csv(train_data, file_names, usr_nm, train_ratio, os.path.join(working_path, model_input_path), 'training', False)
        predict_path = save2csv(predict_data, file_names, usr_nm, train_ratio, os.path.join(working_path, model_input_path), 'predict', False)
        print('Done!')
        return original_path, train_path, predict_path
    except Exception as e:
        print(e)
        return

name_list = ['hxl', 'hw', 'zzx', 'zt', 'zd', 'wy_tmp', 'lf', 'hhl', "hxl2016"]
# name_list = ['hxl2016']
original_path, train_path, predict_path  = dict(), dict(), dict()
working_path = "/project/album_project/"
# working_path = "/Volumes/working/album_project/"

range_filter = [3,3,3, 3,3,3, 3,3,1]
# name_list = ['hxl2016']
for idx, nm in enumerate(name_list):
    print(nm, range_filter[idx])
    original_path[nm], train_path[nm], predict_path[nm] = main(nm, working_path = working_path,  gen_feature_idx = True, filter_range = range_filter[idx] * 24*60*60)
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Combine all data...")
drs.combine_csv(original_path.values(), 'timeonly_combine_original_0.98.csv', os.path.join(working_path,"preprocessed_data/timeonly/original"))
drs.combine_csv(train_path.values(), 'timeonly_combine_training_0.98.csv', os.path.join(working_path, "preprocessed_data/timeonly/training"))
drs.combine_csv(predict_path.values(), 'timeonly_combine_predict_0.98.csv', os.path.join(working_path, "preprocessed_data/timeonly/predict"))
