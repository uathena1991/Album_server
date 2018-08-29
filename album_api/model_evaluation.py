import pdb
import numpy as np
import tensorflow as tf
from sklearn import metrics

###############################################################################################
# cluster based on gps_threshold, and time threshold
def cal_similarity(set_a, set_b):
    # return (len(set_a.intersection(set_b))/(1e-16+ len(set_b)) + len(set_b.intersection(set_a))/(1e-6+len(set_b)))/2
    return len(set_a.intersection(set_b))/(len(set_b)+1e-6)

###############################################################################################
def eval_scene_cluster(alg, gt, vis):
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
################################################################

def eval_scene_pair(strue, spredict, file_names):
    """
    calculate one-one pair accuracy and so (scnee)
    :param strue: true label (scene)
    :param spredict: predict label (scene)
    :param file_names:
    :return:
    """
    res_true = np.array([])
    res_predict = np.array([])
    for idx1, fn1 in enumerate(file_names):
        tmp1 = np.array([strue[fn1] == strue[fn2] for fn2 in file_names[idx1+1:]])
        tmp2 = np.array([spredict[fn1] == spredict[fn2] for fn2 in file_names[idx1+1:]])
        res_true = np.append(res_true, tmp1)
        res_predict = np.append(res_predict, tmp2)
    acc = metrics.accuracy_score(res_true, res_predict)
    prec = metrics.precision_score(res_true, res_predict)
    rec = metrics.recall_score(res_true, res_predict)
    auc = metrics.roc_auc_score(res_true, res_predict)
    return acc, prec, rec, auc

#############################################################################################
def eval_WDL_prediction(true_label, predict_label):
    # analyze precision, recall and so
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.4

    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)
    acc, acc_op = tf.metrics.accuracy(labels = x, predictions = y)
    prec, prec_op = tf.metrics.precision(labels = x, predictions = y)
    rec, rec_op = tf.metrics.recall(labels = x, predictions = y)
    auc, auc_op = tf.metrics.auc(labels = x, predictions = y)

    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print("--------------------W&D model prediction results (Event)-------------------------")
    ev = np.empty(shape=[4,2])
    ev[0,:] = sess.run([acc, acc_op], feed_dict={x: true_label, y: predict_label})
    print("Accuracy", ev[0,-1])
    ev[1,:]= sess.run([prec, prec_op], feed_dict={x: true_label, y: predict_label})
    print('Precision', ev[1,-1])
    ev[2,:] = sess.run([rec, rec_op], feed_dict={x: true_label, y: predict_label})
    print('Recall', ev[2,-1])
    ev[3,:] = sess.run([auc, auc_op], feed_dict={x: true_label, y: predict_label})
    print('AUC', ev[3,-1])
    sess.close()

    return ev

####################################################################################################################################
def img_label_2_clusters(img_labels):
    """

    :param img_labels:
    :return:
    """
    res = {}
    for k,v in img_labels.items():
        if ''.join(v) not in res:
            res[''.join(v)] = [k]
        else:
            res[''.join(v)].append(k)
    return res

####################################################################################################################################

def main(predict_event, true_event,
         predict_scene, true_scene,
         predict_scene_cluster,
         file_names,
         min_pic_num, max_pic, vis = False):
    """

    :param predict_event: list, img pair
    :param true_event:  list, img, pair
    :param predict_scene: dict, key: img name, value: predict events, predict scenes
    :param true_scene:  dict, key: img name, values: (event with scene)
    :param predict_scene_cluster: set, grouped by different scenes
    :param file_names:
    :param scene_gt0:  scene ground truth, set, (should be loaded from json file)
    :param min_pic_num:
    :param max_pic:
    :return:
    """
    ############################## (evnet) calculate accuracy, precision, recall #####################################
    print("+++++++++++++Event prediction results (one-one image pair):+++++++++++++++++")
    ev = eval_WDL_prediction(true_event, predict_event)

    ############################## (scene) calculate accuracy, precision, recall #####################################
    acc, rec, prec, auc = eval_scene_pair(true_scene, predict_scene, file_names)
    print("+++++++++++++Scenes cluster results (one-one pair results):+++++++++++++++++")
    print("Accuracy: %1.4f" %acc)
    print("Precision: %1.4f" %prec)
    print("Recall: %1.4f" %rec)
    print("AUC: %1.4f" %auc)

    ################ (scene) calculate precision, recall, F1 score based on scene cluster ############################
    print("+++++++++++++Scenes cluster results (based on scene cluster):+++++++++++++++++")
    # calculate true_scene_cluster based on true_scene
    true_scene_label = img_label_2_clusters(true_scene)
    scene_gt0 = np.array([x for x in true_scene_label.values()])
    scene_gt = scene_gt0[[min_pic_num<=len(x) <= max_pic for x in scene_gt0]]
    res_gt_wdl = eval_scene_cluster(predict_scene_cluster, scene_gt, vis)
    print("F1 score: %1.4f" %res_gt_wdl[2])
    print("Precision: %1.4f" %res_gt_wdl[0])
    print("Recall: %1.4f\n" %res_gt_wdl[1])

    res = dict()
    res['event_eval'] = ev
    res['scene_pair'] = (acc, rec, prec, auc)
    res['scene_cluster'] = res_gt_wdl
    return res