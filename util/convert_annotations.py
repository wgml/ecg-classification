import pprint
import os

pp = pprint.PrettyPrinter(indent=2)

files = os.listdir('ReferencyjneDane2')

# anno_map = {}
# id = 1
# # get class ids
# for fid in files:
# 	with open('./ReferencyjneDane2/' + fid + '/annotations.txt') as f:
# 		annotations = [x.split()[2] for x in f if x.strip()]
# 		for a in set(annotations):
# 			if a not in anno_map:
# 				anno_map[a] = id
# 				id += 1
# pp.pprint(anno_map)
# pp.pprint({'Expected classes': len(anno_map)})

for fid in files:
	anno_file = './ReferencyjneDane2/' + fid + '/annotations.txt'
	data_file = './ReferencyjneDane2/' + fid + '/ConvertedQRSRawData.txt'
	save_file = './ReferencyjneDane2/' + fid + '/Class_IDs_2.txt'
	data_save_file = './ReferencyjneDane2/' + fid + '/ConvertedQRSRawData_2.txt'

	#load anno
	with open(anno_file) as f:
	    f=[x.strip() for x in f if x.strip()]
	    data=[tuple(map(str,x.split())) for x in f[:]]
	    annotations = {x[1]: x[2] for x in data}

	#translate anno
	anno_map = {}
	id = 1
	for k, v in annotations.items():
		if v not in anno_map:
			anno_map[v] = id
			id += 1
		annotations[k] = anno_map[v]
	pp.pprint(anno_map)
	pp.pprint({'fid': fid, 'Expected classes': len(anno_map)})

	#load data
	with open(data_file) as f:
		data = {x.strip().split('.', 1)[0]:x.strip() for x in f if x.strip()}

	#remove unknown annotations
	annotations = {int(key):value for key, value in annotations.items() if key in data}
	data = {int(key):value for key, value in data.items() if int(key) in annotations}

	#save in sorted order!
	with open(save_file, 'w') as f:
		for key, value in sorted(annotations.items()):
			f.write(str(value) + os.linesep)

	with open(data_save_file, 'w') as f:
		for key, value in sorted(data.items()):
			f.write(str(value) + os.linesep)
