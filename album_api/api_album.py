import os, subprocess
import json
import argparse
from multiprocessing.pool import ThreadPool

from flask import Flask, request, Response

######################common string#####################
parser = argparse.ArgumentParser(description="All parameters")
parser.add_argument('--json_save_path', type=str, default='tmp',
                    help='User name')

parser.add_argument('--working_path', type = str,
                    # default = '/Volumes/working/album_project/',
                    default = '/project/album_api/',
                    help='Working path')
FLAGS, unparsed = parser.parse_known_args()
save_path = os.path.join(FLAGS.working_path, FLAGS.json_save_path) # path to temporarily save plist on server




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.debug = True
# user info
# user = {
#     'ubuntu': ['cloudwalk2018']
# }



@app.route('/')

def login():
	return 'Hello kitty'



@app.route('/run_model', methods=['POST'])
def run_model():
	"""
	:return: final scene cluster
	"""
	print("############## run_model #################")
	# print(request.headers)
	# print(request.json)
	raw_data = request.json.get('plist_info')  # json type
	usr_id = request.json.get('user_id')  # user_id
	thr_id = request.json.get('thread_id')
	count = request.json.get('count')
	model_type = request.json.get('model_type')
	# temporarily save plist to a file on server
	file_path = os.path.join(save_path, usr_id + '_plist.json')
	with open(file_path, 'w') as file:
		json.dump(raw_data, file)
	file.close()
	### start a thread
	print("File path:", file_path)
	print("User id:", usr_id)
	print("Model type:", model_type)
	thread1 = JobsThreadWithReturn(thr_id, "Thread-%d" %thr_id, count, usr_id, file_path, model_type)
	async_result = thread1.apply_async(thread1.run)
	final_result = async_result.get()
	print('result:\n', final_result)
	return Response(json.dumps(final_result), mimetype='application/json')


###
class JobsThreadWithReturn(ThreadPool):
	print("############## JobsThread #################")

	def __init__(self, threadID, name, counter, usr_id, file_path, model_type):
		ThreadPool.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter
		self.usr_id = usr_id
		self.file_path = file_path
		self.model_type = model_type # timeonly or timegps

	def run(self):
		print("%s started..." %self.name)
		print(self.usr_id, self.file_path)
		res = subprocess.check_output(
			"python3 combine_album_server.py --usr_nm=%s --working_path=%s --plist_json=%s --model_type=%s --vis_idx_final=False" % (
			self.usr_id, FLAGS.working_path, self.file_path, self.model_type),
			shell=True)
		# get final results and return in json format
		# pdb.set_trace()
		fname = str(res).split('Final cluster results is saved in ')[1].split("\\n")[0]
		tmp_file = open(fname, 'r')
		json_output = json.loads(json.loads(json.load(tmp_file))['res_final'])
		print("%s ended..." %self.name)
		return json_output


if __name__ == '__main__':
	# app.run(host='192.168.1.4', port=5000, threaded=True)
	app.run(host='0.0.0.0', port=9000, threaded=True) # the port should be consistent with the port forwarded from the host server to docker
