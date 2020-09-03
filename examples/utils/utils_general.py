import torch
import torch.utils.data as data
import random
import math

from .dataloader_dst import *
from .dataloader_nlg import *
from .dataloader_nlu import *
from .dataloader_dm import *

def get_loader(args, mode, tokenizer, datasets, unified_meta, shuffle=False):
    task = args["task"]
    batch_size = args["batch_size"]
    
    combined_ds = []
    for ds in datasets:
        combined_ds += datasets[ds][mode]
    
    # do not consider empty system responses
    if (args["task_name"] == "rs") or (args["task"] == "dm"): 
        print("[Info] Remove turns with empty system response...")
        combined_ds = [d for d in combined_ds if d["turn_sys"]!=""]
    
    if (args["task_name"] == "rs"):
        print("[Info] Remove turn=0 system response...")
        combined_ds = [d for d in combined_ds if d["turn_id"]!=0]
        
    data_info = {k: [] for k in combined_ds[0].keys()}
    for d in combined_ds:
        for k in combined_ds[0].keys():
            data_info[k].append(d[k])

    dataset = globals()["Dataset_"+task](data_info, tokenizer, args, unified_meta, mode, args["max_seq_length"])
    
    bool_shuffle = (mode=="train" or shuffle)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=bool_shuffle,
                                              collate_fn=globals()["collate_fn_{}_{}".format(task, args["example_type"])])
    return data_loader

def get_unified_meta(datasets):
    unified_meta = {"others":None}
    for ds in datasets:
        for key, value in datasets[ds]["meta"].items():
            if key not in unified_meta.keys():
                unified_meta[key] = {}
            if type(value) == list:
                for v in value:
                    if v not in unified_meta[key].keys():
                        unified_meta[key][v] = len(unified_meta[key])  
            else:
                unified_meta[key] = value
                
    return unified_meta
