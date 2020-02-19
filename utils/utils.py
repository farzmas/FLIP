import json
import numpy as np
import pandas as pd
import networkx as nx
import pickle as pk
from tqdm import tqdm
from scipy import sparse
from texttable import Texttable
import os

from sklearn.linear_model import LogisticRegression



def read_data(args):
    file_ = open(args.folder_path + args.file_name, 'rb')
    data = pk.load(file_)
    file_.close()
    return data

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def save_embd(embd, args):
    file = open(args.output_folder+args.file_name,'wb')
    pk.dump(embd, file )
    file.close()

def load_embd(args):
    file = open(args.output_folder+args.file_name,'rb')
    embd = pk.load(file )
    file.close()
    return embd


def load_embd_from_txt(args):
    embds = []
    for ct in range(10):
        file_name = args.file_name.split('.')[0] +'.txt'
        file_ = open(args.embedding_path+ str(ct)+ file_name, 'r')
        line = file_.readline().strip().split()
        embd = np.zeros((int(line[0]),int(line[1])))
        for line in file_:
            if len(line)>2:
                line = line.strip().split()
                node = int(line[0])
                line = [float(x) for x in line[1:]]
                embd[node,:] = np.array(line)
        embds.append(embd)
    return embds


def walk_exist(args, i):
    # only check one incident 
    path = args.walk_path+ str(i) +args.file_name.split('.')[0] +'.txt'
    #print(path)
    return os.path.isfile(path)

