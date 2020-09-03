import json
import ast
import collections
import os

from .utils_function import get_input_example


def read_langs_turn(args, dial_files, max_line = None, ds_name=""):
    print(("Reading from {} for read_langs_turn".format(ds_name)))
    
    data = []
    
    cnt_lin = 1
        
    dials = open(dial_files, 'r').readlines()

    turn_sys = ""
    turn_usr = ""

    for line in dials:

        turns = line.split("__eou__")[:-1]

        dialog_history = []
        for ti, turn in enumerate(turns):
            if ti%2 == 1:
                turn_usr = turn.lower().strip()
                data_detail = get_input_example("turn")
                data_detail["ID"] = "{}-{}".format(ds_name, cnt_lin)
                data_detail["turn_id"] = ti // 2
                data_detail["turn_usr"] = turn_usr
                data_detail["turn_sys"] = turn_sys
                data_detail["dialog_history"] = list(dialog_history)

                if (not args["only_last_turn"]):
                    data.append(data_detail)

                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)

            else:
                turn_sys = turn.lower().strip()

        if args["only_last_turn"]:
            data.append(data_detail)

        cnt_lin += 1
        if(max_line and cnt_lin >= max_line):
            break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(("Reading from {} for read_langs_dial".format(file_name)))
    
    raise NotImplementedError



def prepare_data_dailydialog(args):
    ds_name = "dailydialog"
    
    example_type = args["example_type"]
    max_line = args["max_line"]
    
    onlyfiles_trn = os.path.join(args["data_path"], 'ijcnlp_dailydialog/train/dialogues_train.txt')
    onlyfiles_dev = os.path.join(args["data_path"], 'ijcnlp_dailydialog/validation/dialogues_validation.txt')
    onlyfiles_tst = os.path.join(args["data_path"], 'ijcnlp_dailydialog/test/dialogues_test.txt')
    
    _example_type = "dial" if "dial" in example_type else example_type
    pair_trn = globals()["read_langs_{}".format(_example_type)](args, onlyfiles_trn, max_line, ds_name)
    pair_dev = globals()["read_langs_{}".format(_example_type)](args, onlyfiles_dev, max_line, ds_name)
    pair_tst = globals()["read_langs_{}".format(_example_type)](args, onlyfiles_tst, max_line, ds_name)

    print("Read {} pairs train from {}".format(len(pair_trn), ds_name))
    print("Read {} pairs valid from {}".format(len(pair_dev), ds_name))
    print("Read {} pairs test  from {}".format(len(pair_tst), ds_name))  
    
    meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

