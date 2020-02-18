import json
import numpy as np
import pandas as pd
import networkx as nx
import pickle as pk
from tqdm import tqdm
from scipy import sparse
from texttable import Texttable


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



