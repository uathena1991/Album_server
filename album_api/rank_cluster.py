import os, pdb
import argparse
import ast
import collections

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from sklearn.cluster import OPTICS
import data_retriever_server as drs
from common_lib import UnionFind, visualize_cluster

######################################################################

def cluster_into_events(predict_f, img_path, vis_idx):
                        # img_path, vis_idx = False):
    all_pics = np.unique(pd.concat([predict_f['1st Image'], predict_f['2nd Image']]).values)
    uf = UnionFind(list(all_pics))
    for idx, row in predict_f.iterrows():
        if row['predict_event'] == 1 and row['1st Image'] != row['2nd Image']:
            uf.union(row['1st Image'], row['2nd Image'])
    # pdb.set_trace()
    ## Visualization
    clusters = np.array(uf.components())
    if vis_idx:
    	visualize_cluster(clusters, img_path)
    return clusters


######################################################################
def cal_rank_num_pic(clusters):
    """

    :param clusters:  clusters above the threshold
    :return: array: rank of each cluster (higher, better)

    """
    len_clusters = np.array([len(x) for x in clusters])
    rank_idx = np.array(len_clusters).argsort().argsort()
    return rank_idx


######################################################################
def cal_rank_holiday_freq(clusters, wanted_holiday, file_names):
    """

    :param clusters:
    :param wanted_holiday:
    :param file_names:
    :return:  ramk of frequencies of holiday (2:1 weights -- 3:1) in each cluster (higher, better)
    """
    res_average = []
    for cl in clusters:
        tmp_days2 = [wanted_holiday[file_names == c] == '2' for c in cl]
        tmp_days1 = [wanted_holiday[file_names == c] == '1' for c in cl]
        res_average.append(np.mean(tmp_days1) * 3 / 4 + np.mean(tmp_days2) * 1 / 4)
    return np.array(res_average).argsort().argsort()


######################################################################
def cal_rank_closest_holiday(clusters, wanted_closest_holiday, file_names):
    """
    Use average closest holiday for each cluster
    :param clusters:
    :param wanted_closest_holiday:
    :param file_names:
    :return: rank for each cluster (lower, better)
    """
    res_average = []
    for cl in clusters:
        tmp_days = [wanted_closest_holiday[file_names == c] for c in cl]
        res_average.append(np.mean(tmp_days))
    return np.array(res_average).argsort()[::-1].argsort()


######################################################################
def cal_rank_city_freq(clusters, wanted_city_prop, file_names):
    """
    Use average city frequency for each cluster
    :param clusters:
    :param wanted_city_prop:
    :param file_names:
    :return: rank for each cluster (lower, better)
    """
    res_average = []
    for cl in clusters:
        tmp_days = [wanted_city_prop[file_names == c] for c in cl]
        res_average.append(np.mean(tmp_days))
    return np.array(res_average).argsort()[::-1].argsort()


######################################################################
def cal_rank_ave_event(struct_clus, img_label, rank_events):
    """
    return rank for each given cluster, and update new event ranke
    :param struct_clus:
    :param img_label:
    :param rank_events:
    :return:
    """
    res = []
    new_rank = np.array(rank_events)
    for idx, events in enumerate(struct_clus):
        tmp = []
        for scene in events[0]:
            # pdb.set_trace()
            res.append(np.mean([rank_events[img_label[x][0]] for x in scene]))
            tmp.append(res[-1])
        if len(tmp) > 0:
            new_rank[idx] = np.mean(tmp)

    return np.array(res), list(new_rank)

####################################################################
def cal_rank_total(cond, cluster, struct_cluster, wanted_closest_holiday, file_names, wanted_city_prop, wanted_holiday, img_label, rank_events_val,
                   w0 = 1/5, w1 = 1/5, w2 = 1/5, w3 = 1/5, w4 = 1/5):
    """

    :param cond: 0: cluster events, 1: cluster scenes
    :param cluster: clusters to rank (exclude those # of pics not in the given range)
    :param wanted_closest_holiday, file_names, wanted_city_prop, wanted_holiday, img_label:
    :param rank_parent:
    :param w0, w1, w2, w3, w4: weights for event, num of pics, closet holiday, city, freq_holiday
    :return: ranked clusters, and each cluster's ranking
    """
    rank_numpic = cal_rank_num_pic(cluster)
    rank_clohol = cal_rank_closest_holiday(cluster, wanted_closest_holiday, file_names)
    rank_city = cal_rank_city_freq(cluster, wanted_city_prop, file_names)
    rank_freq_holiday = cal_rank_holiday_freq(cluster, wanted_holiday, file_names)
    if cond == 1: # cluster scenes
        rank_event, rank_events_val = cal_rank_ave_event(struct_cluster, img_label, rank_events_val)
    else: # cluster events
        rank_event = np.array(rank_events_val)
    # weights for each rank
    rank_val = w0 * rank_event + w1 * rank_numpic + w2 * rank_clohol + w3 * rank_city + w4 * rank_freq_holiday
    # print("Number of albums selected: %d" % min(len(rank_average), FLAGS.max_album))
    # res = cluster[rank_val.argsort()[::-1][:min(len(rank_val), FLAGS.max_album)]]
    res = cluster[rank_val.argsort()[::-1]]
    if cond == 0:
        rank_events_val = rank_val[rank_val.argsort()[::-1]]
    return res, rank_val[rank_val.argsort()[::-1]], rank_events_val


#####################################
def labeling_image_cluster(img_label, clusters, noise_cl, cond_idx):
    """
    memorize event and scene numbers for each image(including res_event, res_unchosen, res_noise)
    cond_idx: 0: event, 1: scene
    label: from 0 to len(clusters) - 1
    """
    for idx, rr in enumerate(clusters):
        for fn in rr:
            img_label[fn][cond_idx] = idx
    for idx,fn in enumerate(noise_cl):
        img_label[fn][cond_idx] = -idx
    return img_label


############################################################################################
def find_best_thres(clust, len_cluster, eps_range = [0, 1], step = 0.01):
            """
            to have fewest noisy sample
            """
            min_noise = len_cluster
            res_eps = 1.0
            res_label = []
            for eps in np.arange(eps_range[0], eps_range[1], step):
                _, tmp_label = clust.extract_dbscan(eps)
                min_noise, res_eps, res_label = (sum(tmp_label == -1), eps, tmp_label) if (sum(tmp_label == -1) < min_noise and len(np.unique(tmp_label)) > 1)  else (min_noise, res_eps, res_label)
            return res_label, res_eps, min_noise





################################################################
def cluster_into_scenes(res_rank, wanted_gps, wanted_time, file_names, min_pic_num, max_pic,
                         show_idx=False):
    """

    :param res_rank:
    :param wanted_gps:
    :param wanted_time:
    :param file_names:
    :param min_pic_num:
    :param max_pic:
    :param thres:
    :param show_idx:
    :return:
    """
    def sub_cluster(wanted_gps, wanted_time, file_names, min_pic_num, show_idx=False):

        # normalization
        wanted_xyz = [drs.lonlat2xyz(x[0], x[1], x[2]) for x in wanted_gps]
        norm_xyz = np.array((wanted_xyz - np.mean(wanted_xyz, 0)) / (np.std(wanted_xyz, 0) + np.array([1e-15, 1e-15, 1e-15])))
        wanted_secs = drs.convert_datetime_seconds(wanted_time)
        norm_secs = np.array((wanted_secs - np.mean(wanted_secs)) / (np.std(wanted_secs) + np.array([1e-15, 1e-15, 1e-15])))
        #         norm_info = np.array([np.array([x[1], x[2], y[0]]) for x, y in zip(norm_xyz, norm_secs)])
        norm_info = np.array([np.array([x[0], x[1], x[2], y[0]]) for x, y in zip(norm_xyz, norm_secs)])
        #         img_cl_idx = hcluster.fclusterdata(norm_info, thres, criterion="distance", method = 'centroid')
        clust = OPTICS(min_pic_num)
        clust.fit(norm_info)
        img_cl_idx, res_eps, min_noise = find_best_thres(clust, len(wanted_gps))
        # img_cl_idx = DBSCAN(thres, 3).fit_predict(norm_info)
        # plotting
        if show_idx:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            title = "threshold: %f, number of clusters: %d" % (res_eps, len(np.unique(img_cl_idx)))
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


    ###########################################################################
    new_final = [[[], [], []] for _ in res_rank] # for each event, record: chosen, unchosen, noise
    for idx, rr in enumerate(res_rank):
        if len(rr) <= max_pic:
            if len(rr) >= min_pic_num:
                new_final[idx][0] = np.array([rr])
            else:
                new_final[idx][1] = np.array([rr])
        else:
            to_split = [rr]
            while len(to_split) > 0:
                tmp_tosplit = []
                for ts in to_split:
                    ii = [np.where(file_names == x)[0][0] for x in ts]
                    tcluster, img_cluster_idx, tnoise, tunchosen = sub_cluster(wanted_gps[ii], wanted_time[ii], file_names[ii], min_pic_num, show_idx)
                    tmp_tosplit = np.append(tmp_tosplit, tcluster[[len(x) > max_pic for x in tcluster]])
                    new_final[idx][0] = np.append(new_final[idx][0], tcluster[[0 < len(x) <= max_pic for x in tcluster]])
                    new_final[idx][1] = np.append(new_final[idx][1], tunchosen[[len(x) > 0 for x in tunchosen]])
                    new_final[idx][2] = np.append(new_final[idx][2], tnoise)
                to_split = tmp_tosplit
    res_final = np.concatenate([x[0] for x in new_final])
    res_unchosen = np.concatenate([x[1] for x in new_final])
    res_noise = np.concatenate([x[2] for x in new_final])
    return new_final, res_final, res_unchosen, res_noise



######################################################################

def main(FLAGS0, npz_features, predict_df):
    global FLAGS
    FLAGS = FLAGS0
    ############################################### load features for each image (for test images) ########################################################
    file_names = npz_features['file_names']
    wanted_holiday = npz_features['wanted_holiday']
    wanted_closest_holiday = npz_features['wanted_closest_holiday']
    wanted_city_prop = npz_features['wanted_city_prop']
    wanted_gps = npz_features['wanted_gps']
    wanted_time = npz_features['wanted_time']

    ######################### cluster all scenes into events, based on WDL model prediction (using union find) ##########################################
    print("Clustering images into events based on predictions")
    res_events = cluster_into_events(predict_df,
                               os.path.join(FLAGS.image_parent_path, FLAGS.label_pic_path),
                               vis_idx = FLAGS.vis_idx_cluster)
    # record event # and scene # for each image
    predict_img_label = dict()
    for idx, fn in enumerate(file_names):
        predict_img_label[fn] = [-idx, -idx]
    predict_img_label = labeling_image_cluster(predict_img_label, res_events, [], 0)

    print('Done...%d events found' %len(res_events))


    ########################################## ranking events ######################################################################
    print("Ranking events...")
    rank_events_val = [0 for _ in res_events]

    res_event_rank1, rank_events1, rank_events_val = cal_rank_total(0, res_events, [], wanted_closest_holiday, file_names, wanted_city_prop, wanted_holiday,
                                              predict_img_label, rank_events_val, 0)


    ######################################### choose scenes within events (if # pic > FLAGS.max_pic) ###############################
    print('Splitting events into scenes....')
    structed_res, res_scenes, res_unchosen, res_noise = cluster_into_scenes(res_event_rank1, wanted_gps, wanted_time, file_names, FLAGS.min_pic_num, FLAGS.max_pic, show_idx = False)
    # record scene # for each image
    predict_img_label = labeling_image_cluster(predict_img_label, res_scenes, res_noise, 1)

    print('Done...%d scenes found' %len(res_scenes))

    ####################rank scene (two-levels, consider ranking of events)#################
    print("Scene ranking and final selection")
    res_final, rank_scenes_val, rank_events_val = cal_rank_total(1, res_scenes, structed_res, wanted_closest_holiday, file_names, wanted_city_prop, wanted_holiday,
                                              predict_img_label, rank_events_val)

    ################ save to json file #######################
    res_dict = collections.defaultdict()
    tmp_struc_res = []
    for xx in structed_res:
        if len(xx[0]) > 0:
            xx[0] = list(xx[0][0])
        else:
            xx[0] = list(xx[0])
        if len(xx[1]) > 0:
            xx[1] = list(xx[1][0])
        else:
            xx[1] = list(xx[1])
        # if len(xx[2]) > 0:
        xx[2] = list(xx[2])
        tmp_struc_res.append(xx)
    res_dict['predict_img_label'] = predict_img_label
    res_dict['structured_res'] = json.dumps(tmp_struc_res)
    res_dict['res_final'] = json.dumps([list(x) for x in res_final])
    res_dict['res_noise'] = json.dumps([list(x) for x in res_noise])
    res_dict['res_unchosen'] = json.dumps([list(x) for x in res_unchosen])
    res_dict['res_rank_event'] = json.dumps([x for x in rank_events_val])
    res_dict['res_rank_scene'] = json.dumps([x for x in rank_scenes_val])
    res_dict['res_rank_scene'] = json.dumps([x for x in rank_scenes_val])
    output_fn = os.path.join(FLAGS.working_path, FLAGS.final_save_path, FLAGS.usr_nm + "_" + FLAGS.model_type + "_" + FLAGS.model_folder_name + ".json")
    res_dict['file_path'] = output_fn
    with open(output_fn, 'w') as file:
        json.dump(json.dumps(res_dict), file)
    print('Final cluster results is saved in %s' %output_fn)
    file.close()

    ############################ visualize ########################
    if FLAGS.vis_idx_final:
        visualize_cluster(res_final, os.path.join(FLAGS.image_parent_path, FLAGS.label_pic_path))
    print('All done!')
    print("Final scene", res_final)

    return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cluster based on model prediction, rank clusters, and choose given numbers of albums")

    parser.add_argument(
        '--working_path', type=str, default='/Volumes/working/album_project/',
        help='working dir for the project.')

    parser.add_argument('--usr_nm', type=str, default='zd',
                        help='User name (must remain the same across plist, picture folders')


    parser.add_argument('--model_type', type=str, default='timeonly',
                    help='Model type: timegps, or timeonly; Raise error otherwise.')

    parser.add_argument(
        '--model_folder_name', type=str, default='new_timegps_Adadelta_L3_noDO_BN_00003_004_0',
        help='Base directory for the model.')

    parser.add_argument(
        '--predict_output', type=str, default='/Volumes/working/album_project/model_prediction/timegps_pred_zd_new_timegps_Adadelta_L3_noDO_BN_00003_004_0_20180828092832.csv',
        help='Model prediction file.')


    parser.add_argument('--plist_folder', type=str,
                        # default='/project/album_project/serving_data/hw_plist.json',
                        default='serving_data/',
                    help=' Path to the saved plist json file (input)')





    parser.add_argument(
        '--final_save_path', type=str, default='final_result/',
        help='Path to save the final result.')

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

    parser.add_argument('--vis_idx_cluster', type=ast.literal_eval, default = False,
                        help='Bool value: whether to show clusters.')


    parser.add_argument('--vis_idx_final', type=ast.literal_eval, default = False,
                        help='Bool value: whether to show selected albums.')


    parser.add_argument(
        '--print_parser', type=ast.literal_eval, default=False,
        help='Bool value: whether to show parsed FLAGS and unparsed values')

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.print_parser:
        print("FLAGS", FLAGS)
        print("unparsed", unparsed)

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.plist_json = os.path.join(FLAGS.working_path, FLAGS.plist_folder, FLAGS.usr_nm + "_plist.json")
    # FLAGS.label_pic_path = os.path.join(FLAGS.image_parent_path, FLAGS.usr_nm + "_label_raw")
    output_fn = main(FLAGS)
    # print(output_fn)
