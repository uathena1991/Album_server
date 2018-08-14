import os, subprocess
import json
from multiprocessing.pool import ThreadPool

from flask import Flask, request, Response



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


working_path = '/project/album_project/' # in docker (on server), where the model is
save_path = os.path.join(working_path, 'tmp') # path to temporarily save plist on server




@app.route('/run_model', methods=['POST'])
def run_model():
	print("############## run_model #################")
	# print(request.headers)
	# print(request.json)
	raw_data = request.json.get('plist_info')  # json type
	usr_id = request.json.get('user_id')  # user_id
	thr_id = request.json.get('thread_id')
	count = request.json.get('count')
	# temporarily save plist to a file on server
	file_path = os.path.join(save_path, usr_id + '.json')
	with open(file_path, 'w') as file:
		json.dump(raw_data, file)
	file.close()
	### start a thread
	print("File path", file_path)
	print("User id", usr_id)
	thread1 = JobsThreadWithReturn(thr_id, "Thread-%d" %thr_id, count, usr_id, file_path)
	async_result = thread1.apply_async(thread1.run)
	final_result = async_result.get()
	# print('result:', final_result)
	return Response(json.dumps(final_result), mimetype='application/json')


###
class JobsThreadWithReturn(ThreadPool):
	print("############## JobsThread #################")

	def __init__(self, threadID, name, counter, usr_id, file_path):
		ThreadPool.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter
		self.usr_id = usr_id
		self.file_path = file_path

	def run(self):
		print("%s started..." %self.name)
		print(self.usr_id, self.file_path)
		res = subprocess.check_output(
			"python3 combine_album_server.py --usr_nm=%s --plist_json=%s --vis_idx_final=False" % (
			self.usr_id, self.file_path),
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
