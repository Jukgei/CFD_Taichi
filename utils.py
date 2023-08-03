import json

def read_config(file_name):
	try:
		with open(file_name, 'r') as f:
			data = json.load(f)
			return data
	except Exception as e:
		print(e)
		print('Parsing config file error')
		exit(3)