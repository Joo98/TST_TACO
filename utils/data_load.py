import os
import argparse
import torch
import numpy as np
from tqdm import tqdm 
import csv
from sklearn.model_selection import train_test_split

def data_load(dataset_path:str, huggingface_data:bool=True, seed:int=42, 
              mode:str="train") -> (dict, dict):
    
    
    src_list = []
    trg_list = []
    if mode == 'train':
        file_src = os.path.join(dataset_path, mode, 'informal_em_train.txt')
        with open(file_src, "r") as f:
            for line in f :
                src_list.append(line)
        file_trg = os.path.join(dataset_path, mode, 'formal_em_train.txt')        
        with open(file_trg, "r") as f:
            for line in f :
                trg_list.append(line)
    else : 
        file_src = os.path.join(dataset_path, mode, 'informal_em_test.txt')
        with open(file_src, "r") as f:
            for line in f :
                src_list.append(line)

        file_trg_list = [os.path.join(dataset_path, mode, 'formal.ref0_em_test.txt'),
                    os.path.join(dataset_path, mode, 'formal.ref1_em_test.txt'),
                    os.path.join(dataset_path, mode, 'formal.ref2_em_test.txt'),
                    os.path.join(dataset_path, mode, 'formal.ref3_em_test.txt')
                    ]
        trg_list = [[] for _ in range(len(src_list))]
        for file_trg in file_trg_list:
            with open(file_trg, "r") as f:
                file = list(f)
                for i in range(len(file)):
                    trg_list[i].append(file[i])

    assert len(src_list) == len(trg_list)
    print(f'example pair (informal / formal ) : \n\t {src_list[0]} \t {trg_list[0]}' )