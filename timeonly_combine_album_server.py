# -*- coding=utf-8 -*-

"""retrieve feature matrix and labels from plist files, and save those to csv"""
import os
import argparse
import ast
import get_input_wdl_timeonly as giwt
import timeonly_wide_deep_predict as wdp
import rank_cluster as rcr
##
parser = argparse.ArgumentParser(description="All parameters")

######################common string#####################

parser.add_argument('--usr_nm', type=str, default='hxl2016',
                    help='User name')

parser.add_argument('--working_path', type = str,
                    default = '/Volumes/working/album_project/',
                    # default = '/project/album_project/',
                    help='Working path')

parser.add_argument('--image_parent_path', type = str,
                    default = '/Volumes/working/album_project/album_data/',
                    # default = '/data/album_data/',
                    help='Parent path of images')

parser.add_argument('--model_cond', type=str, default='_WDL_timegps/',
                    help='Path to save the final result.')



##################### bool #####################


parser.add_argument('--generate_plist_idx', type = ast.literal_eval, default = False,
                help='if True, generate features from plist info, otherwise, load from the .npy file ')

parser.add_argument('--generate_feature_idx', type = ast.literal_eval, default = False,
                help='if True, generate features for two-two compare, otherwise, load from the .npy file ')

parser.add_argument('--generate_holiday_tab', type = ast.literal_eval, default= False,
                help='if True, generate holiday table using api')


parser.add_argument('--vis_idx_cluster', type=ast.literal_eval, default = False,
                    help='Bool value: whether to show clusters.')

parser.add_argument('--vis_idx_rank', type=ast.literal_eval, default = False,
                    help='Bool value: whether to show ranked albums.')

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


parser.add_argument('--thres_scene', type=float, default = 0.16,
                    help='Threshold(normalized) for clustering small scenes')

parser.add_argument('--train_ratio', type=float, default = 0.0,
                    help='Ratio between train/test')


#################### Normally unchanged #########################

parser.add_argument(
    '--model_exported_path', type=str, default='model_output/timeonly_????_0/',
    help='Saved model path')

parser.add_argument('--model_input_path', type = str, default = 'preprocessed_data/',
                    help='Partial path to save preprocessed data')

parser.add_argument('--plist_json', type=str,
                    default='serving_data/',
                    help=' Path to the saved plist json file (input)')


parser.add_argument(
    '--prediction_path', type=str, default='model_prediction/',
    help='Model prediction output path')

parser.add_argument(
	'--final_save_path', type=str, default='final_result/',
	help='Path to save the final result.')

parser.add_argument('--feature_save_path', type=str, default='preprocessed_data/feature_matrix',
                    help=' Path to save plist in the format of .npy (exclude user name)')


parser.add_argument('--city_lonlat', type=str, default='preprocessed_data/city_lonlat.csv',
                    help='Full path to save and load the excel files (with city-latitude-longitude pairs')

parser.add_argument("--holiday_file", type = str, default = 'preprocessed_data/holidays.csv',
                    help = "Full path to the holiday lookup table.")

##############################################################################################


FLAGS, unparsed = parser.parse_known_args()
FLAGS.plist_json = os.path.join(FLAGS.working_path, FLAGS.plist_json, FLAGS.usr_nm + '_plist.json')
if FLAGS.print_parser:
	print(FLAGS)



def main(FLAGS):
	###### data preprocessing ############
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print("Data preprocessing....")
	# if FLAGS.run_prediction:
	original_path, train_path, predict_path = giwt.main(FLAGS.usr_nm, 0.0, working_path=FLAGS.working_path, gen_feature_idx=FLAGS.generate_feature_idx)
	FLAGS.predict_input = predict_path
	####### model prediction #############
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print("Model prediction....")
	cts, acc, prec, rec, auc, predict_fn = wdp.main(FLAGS)
	FLAGS.predict_output = predict_fn
	print("Double-check: prediction file: %s" %FLAGS.predict_output)
	######## rank cluster #############
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('Scene splitting...')
	FLAGS.label_pic_path = FLAGS.usr_nm + '_label_raw'
	res_file = rcr.main(FLAGS)
	return res_file




if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	FLAGS.plist_json = os.path.join(FLAGS.working_path, FLAGS.plist_json, FLAGS.usr_nm + '_plist.json')
	if FLAGS.print_parser:
		print(FLAGS)

	res_final = main(FLAGS)

