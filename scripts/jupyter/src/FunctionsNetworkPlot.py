import glob
import os
import pandas as pd
import gzip
from collections import defaultdict
import networkx as nx

# Return 
# Edges (list of tuples with connections) and
# Positions (dictionary with tuple position 'node':(x,y,z)) 

def positions_GML(N, dim, alpha_a, alpha_g, filename):
    path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml/"
    file = path + filename
    
    x,y,z = [],[],[]
    source = []
    target = []
    
    index = 0
    with open(file, 'rb') as file_in:
        lines = []
        # decompress gzip
        with gzip.GzipFile(fileobj=file_in, mode='rb') as gzip_file:
            for line in gzip_file:
                # decode file
                decoded_line = line.decode('utf-8')
                
                # positions nodes
                if(decoded_line[0]=="x"):
                    x.append(float(line[1:-1]))
                elif(decoded_line[0]=="y"):
                    y.append(float(line[1:-1]))                    
                elif(decoded_line[0]=="z"):
                    z.append(float(line[1:-1])) 
                              
                
                # edges nodes

                # Append to buffer
                lines.append(decoded_line) 
                
                if len(lines) >= 5:  # 1 (current line) + 2 (2 lines below) + 3 (3 lines below)
                    # Check if the line 3 lines before the current line starts with 'edge'
                    if lines[-4].startswith('edge'):
                        # Lines 2 and 3 below the 'edge' line
                        line_2_below = lines[-2]
                        line_3_below = lines[-1]
                        source.append('id_' + line_2_below.strip()[8:-1])
                        target.append('id_' + line_3_below.strip()[8:-1])
                        
    
    
    positions = {}
    for i in range(len(x)):
        positions[f'id_{i}'] = (x[i],y[i],z[i])
    edges = [(i,j) for i,j in zip(source,target)]
    return edges, positions

# Select one file .gml in specific folder (N, dim, alpha_a, alpha_g)
# Return:
# filename
def select_first_gml_gz_file(N, dim, alpha_a, alpha_g):
    directory = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml/"
    # Itera sobre os arquivos na pasta fornecida
    for file in os.listdir(directory):
        if file.endswith('.gml.gz'):
            selected_file = file
            #print(selected_file)
            return selected_file
    
    print("Nenhum arquivo .gml.gz encontrado na pasta.")
    return None

# Função para desenhar o grafo
def draw_graph(ax, G, pos, title):
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color='lightblue', edgecolors='black')
    ax.set_title(title)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.grid(False)
    # Ajuste os limites dos eixos
    x_min, x_max = min(x for x, y in pos.values()) - 1, max(x for x, y in pos.values()) + 1
    y_min, y_max = min(y for x, y in pos.values()) - 1, max(y for x, y in pos.values()) + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))