# -*- coding=utf-8 -*-

"""retrieve feature matrix and labels from plist files, and save those to csv"""
import os
import pdb
import argparse
import ast
import datetime
import json

import data_retriever_server as drs
import wide_deep_predict as wdp
import rank_cluster as rcr
import model_evaluation as me
##
parser = argparse.ArgumentParser(description="All parameters")

######################common string#####################

parser.add_argument('--usr_nm', type=str, default='hxl',
                    help='User name')

parser.add_argument('--working_path', type = str,
                    default = '/Volumes/working/album_project/',
                    # default = '/project/album_project/',
                    help='Working path')

parser.add_argument('--image_parent_path', type = str,
                    default = '/Volumes/working/album_project/album_data/',
                    # default = '/data/album_data/',
                    help='Parent path of images')


parser.add_argument('--model_type', type=str, default='timegps',
                help='Model type: timegps, or timeonly; Raise error otherwise.')

parser.add_argument('--model_folder_name', type=str, default='new_timegps_Adadelta_L4_noDO_BN_00003_004_0',
                    help='Base directory for the model.')

################################ bool ################################
parser.add_argument('--save_feature_idx', type = ast.literal_eval, default = True,
                    help='if True, save feature_matrix and pair_feature_matrix to .npy and .npz respectively ')

parser.add_argument('--save_predict_idx', type = ast.literal_eval, default = True,
                    help='if True, save model prediction to .csv file')

parser.add_argument('--eval_index', type=ast.literal_eval, default = True,
                    help='Bool value: whether to evaluate the model results')

parser.add_argument('--vis_idx_cluster', type=ast.literal_eval, default = False,
                    help='Bool value: whether to show clusters.')

parser.add_argument('--vis_idx_final', type=ast.literal_eval, default = True,
                    help='Bool value: whether to show final selected albums.')

parser.add_argument('--print_parser', type=ast.literal_eval, default = True,
                    help='Bool value: whether to show parsed FLAGS and unparsed values')


###################### int params ################################
parser.add_argument('--min_pic_num', type=int, default = 10,
                    help='Minimum numbers of pics in target clusters')

parser.add_argument('--max_pic', type=int, default = 30,
                    help='Max numbers of albums')

parser.add_argument('--filter_range', type=int, default=96 * 60 * 60,
                    help='Time range to choose two images (s)')

parser.add_argument('--max_album', type=int, default= 50,
                    help='Max numbers of albums needed')



parser.add_argument('--train_ratio', type=float, default = 0.0,
                    help='Ratio between train/test')

parser.add_argument('--half_win_size', type=int, default= 2,
                help='Time window to calc time freq')


#################### Normally unchanged #########################
parser.add_argument('--model_exported_path', type=str, default='model_output/',
                    help='Base directory for the model.')

parser.add_argument('--model_input_path', type = str, default = 'preprocessed_data/',
                    help='Partial path to save preprocessed data')

parser.add_argument('--plist_folder', type=str,
                    default='serving_data/',
                    help=' Path to the saved plist json file (input)')


parser.add_argument('--prediction_path', type=str, default='model_prediction/',
                    help='Model prediction output path')

parser.add_argument('--final_save_path', type=str, default='final_result/',
                    help='Path to save the final result.')

parser.add_argument('--feature_save_path', type=str, default='preprocessed_data/feature_matrix',
                    help=' Path to save plist in the format of .npy (exclude user name)')


parser.add_argument('--city_lonlat', type=str, default='preprocessed_data/city_lonlat.csv',
                    help='Full path to save and load the excel files (with city-latitude-longitude pairs')

parser.add_argument("--holiday_file", type = str, default = 'preprocessed_data/holidays.csv',
                    help = "Full path to the holiday lookup table.")

parser.add_argument('--pic_path_label', type=str, default='_label_raw',
	                help='Full path to pictures')
##############################################################################################


FLAGS, unparsed = parser.parse_known_args()
if FLAGS.print_parser:
	print(FLAGS)



def main(FLAGS):
	print(FLAGS.usr_nm)
	res = dict()
	FLAGS.plist_json = os.path.join(FLAGS.working_path, FLAGS.plist_folder, FLAGS.usr_nm + '_plist.json')
	###### data preprocessing ############
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print("Data preprocessing....")
	model_inputs, npz_features = drs.main(FLAGS)
	####### model prediction #############
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print("Model prediction....")
	predict_df = wdp.main(FLAGS, model_inputs)
	if FLAGS.save_predict_idx:
		new_headers = list(model_inputs.columns) + ['predict_label', 'probability']
		output_fn = os.path.join(FLAGS.working_path, FLAGS.prediction_path,
                     "%s_pred_%s_%s_%s.csv" % (FLAGS.model_type, FLAGS.usr_nm, FLAGS.model_folder_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))	######## rank cluster #############
		predict_df.to_csv(output_fn, header=new_headers, index = False)

	####### clustering based on model prediction #############
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('Scene splitting...')
	FLAGS.label_pic_path = FLAGS.usr_nm + '_label_raw'
	res_dict = rcr.main(FLAGS, npz_features, predict_df)
	res['Scene_res'] = res_dict
	if FLAGS.eval_index:
		res['Evaluation'] = me.main(predict_df['predict_event'], predict_df['Label_e'],
		                            res_dict['predict_img_label'], npz_features['true_label'],
		                            json.loads(res_dict['res_final']),
		                            npz_features['file_names'],
		                            FLAGS.min_pic_num, FLAGS.max_pic)

	return res





if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	if FLAGS.print_parser:
		print(FLAGS)

	res_final = main(FLAGS)

