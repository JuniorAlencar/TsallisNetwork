# -*- coding: utf-8 -*-
import gzip
import os

def gml_reader(N, dim, alpha_a, alpha_g, cod_file):
	path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml/" 
	isExist = os.path.exists(path)
	if(isExist!=True):
		return "doesn't exist", "gml_folder"
	File = f"gml_{cod_file}.gml.gz"


	file_directory = path + File

	node = {"id": [],
		"position":[],
		"degree": []}
	edge = {"connections": [],
		  "distance": []}
	with gzip.open(file_directory) as file_in:
		String = file_in.readlines()
		Lines = [i.decode('utf-8') for i in String]
		for i in range(len(Lines)):
			if(Lines[i]=='node\n'):
				node["id"].append(int(Lines[i+2][4:-2]))
				node["position"].append([float(Lines[i+6][2:-1]),float(Lines[i+7][2:-1]),float(Lines[i+8][2:-1])])
				node["degree"].append(int(Lines[i+9][7:-1]))
			elif(Lines[i]=="edge\n"):
				edge["connections"].append([int(Lines[i+2][8:-2]),int(Lines[i+3][8:-2])])
				edge["distance"].append(float(Lines[i+4][9:-1]))
	return node, edge