import requests
import json
import time
import cProfile
def main(thr_id, count,model_type,  user = 'hw', file_name = '/Volumes/working/album_project/serving_data/hw_plist.json'):
	"""

	:param user:  user id (string), should be unique for each user
	:param thr_id: thread id
	:param count: thread counter
	:param file_name: path of the sudo json file (should be from front-end in the real case)
	:return: model results from the server (list)
	"""
	headers = {'content-type': 'application/json'}

	t1 = time.time()
	file = open(file_name, 'r')
	raw_json = json.load(file)
	json_input = dict()
	json_input['plist_info'] = raw_json
	json_input['user_id'] = user
	json_input['thread_id'] = thr_id
	json_input['count'] = count
	json_input['model_type'] = model_type  # model type (either 'timeonly' or 'timegps')
	r_post = requests.post("http://10.128.34.1:9000/run_model", data = json.dumps(json_input), headers=headers)

	print(r_post.headers)
	print(r_post.text)
	t2 = time.time()
	tt = t2 - t1
	print("thread id: %d, counter: %d" %(thr_id, count))
	print("Total run time:", tt)

	# print(type(json.loads(r_post.text)))
	return json.loads(r_post.text)


cProfile.run("main(1,1, 'timegps','zd', '/Volumes/working/album_project/serving_data/zd_plist.json')", filename = 'result_client.out', sort ="cumulative")
