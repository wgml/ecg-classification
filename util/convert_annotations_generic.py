import pprint
import os
import sys

pp = pprint.PrettyPrinter(indent=2)

directory = sys.argv[1]

files = os.listdir(directory)

anno_map = {}
id = 1
# get class ids
for fid in files:
	with open(directory + '/' + fid + '/annotations.txt') as f:
		annotations = [x.split()[2] for x in f if x.strip()]
		for a in set(annotations):
			if a not in anno_map:
				anno_map[a] = id
				id += 1
pp.pprint(anno_map)
pp.pprint({'Expected classes': len(anno_map)})

unmet_annotations = set(anno_map.keys())

for fid in files:
	anno_file = directory + '/' + fid + '/annotations.txt'
	data_file = directory + '/' + fid + '/ConvertedQRSRawData.txt'
	save_file = directory + '/' + fid + '/label.txt'
	data_save_file = directory + '/' + fid + '/data.txt'

	#load anno
	with open(anno_file) as f:
	    f=[x.strip() for x in f if x.strip()]
	    data=[tuple(map(str,x.split())) for x in f[:]]
	    annotations = {x[1]: x[2] for x in data}

	#load data
	with open(data_file) as f:
		data = {x.strip().split('.', 1)[0]:x.strip() for x in f if x.strip()}

	#remove unknown annotations
	annotations = {int(key):value for key, value in annotations.items() if key in data}
	data = {int(key):value for key, value in data.items() if int(key) in annotations}

	#translate anno
	for k, v in annotations.items():
		annotations[k] = anno_map[v]
		unmet_annotations.discard(v)

	#save in sorted order!
	with open(save_file, 'w') as f:
		for key, value in sorted(annotations.items()):
			f.write(str(value) + os.linesep)

	with open(data_save_file, 'w') as f:
		for key, value in sorted(data.items()):
			f.write(str(value) + os.linesep)

pp.pprint({'Unmet annotations': unmet_annotations})
