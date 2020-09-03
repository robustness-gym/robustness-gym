import json
import ast
import collections
import os

from .utils_function import get_input_example
from .multiwoz.fix_label import *

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"] #, "hospital", "police"]


def read_langs_turn(args, file_name, ontology, dialog_act, max_line = None, update_ont_flag=False):
    print(("Reading from {} for read_langs_turn".format(file_name)))
    
    data = []
    SLOTS = [k for k in ontology.keys()]
    max_resp_len, max_value_len = 0, 0
    domain_counter = {} 
    response_candidates = set()
    add_slot_values = set()
    
    with open(file_name) as f:
        dials = json.load(f)
        
        cnt_lin = 1
        for dial_dict in dials:
            dialog_history, dialog_history_delex = [], []
            
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):

                belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS, args["ontology_version"])
                belief_list = [str(k)+'-'+str(v) for k, v in belief_dict.items()]
                turn_slot_dict = fix_general_label_error(turn["turn_label"], True, SLOTS, args["ontology_version"])
                turn_slot_list = [str(k)+'-'+str(v) for k, v in turn_slot_dict.items()]
                turn_slot = list(set([k.split("-")[1] for k, v in turn_slot_dict.items()]))

                slot_values, gates  = [], []
                for slot in SLOTS:
                    if slot in belief_dict.keys(): 
                        
                        # update ontology
                        if args["ontology_version"] != "" and "the {}".format(belief_dict[slot]) in ontology[slot].keys(): 
                            belief_dict[slot] = "the {}".format(belief_dict[slot])
                        
                        if belief_dict[slot] not in ontology[slot].keys() and update_ont_flag:
                            if slot+"-"+belief_dict[slot] not in add_slot_values  \
                                and "|" not in belief_dict[slot] \
                                and ":" not in belief_dict[slot]: 
                                print("[Info] Adding Slot: {} with value: [{}]".format(slot, belief_dict[slot]))
                                add_slot_values.add(slot+"-"+belief_dict[slot])
                            
                            ontology[slot][belief_dict[slot]] = len(ontology[slot])
                        
                        slot_values.append(belief_dict[slot])
                        
                        if belief_dict[slot] == "none":
                            gates.append(0)
                        else:
                            gates.append(1)
                    else:
                        slot_values.append("none")
                        gates.append(0)

                # dialgoue act (exclude domain)
                if turn["turn_idx"] == 0 and turn["system_transcript"] == "":
                    cur_sys_acts = set()
                elif str(turn["turn_idx"]) not in dialog_act[dial_dict["dialogue_idx"].replace(".json", "")].keys():
                    cur_sys_acts = set()
                elif dialog_act[dial_dict["dialogue_idx"].replace(".json", "")][str(turn["turn_idx"])] == "No Annotation":
                    cur_sys_acts = set()
                else:
                    cur_sys_acts = dialog_act[dial_dict["dialogue_idx"].replace(".json", "")][str(turn["turn_idx"])]
                    
                    cur_sys_acts = set([key.split("-")[1].lower() for key in cur_sys_acts.keys()])
                
                data_detail = get_input_example("turn")
                data_detail["slots"] = SLOTS
                data_detail["ID"] = dial_dict["dialogue_idx"]
                data_detail["turn_id"] = turn["turn_idx"]
                data_detail["domains"] = dial_dict["domains"]
                data_detail["turn_domain"] = turn["domain"]
                data_detail["turn_usr"] = turn["transcript"].strip()
                data_detail["turn_sys"] = turn["system_transcript"].strip()
                data_detail["turn_usr_delex"] = turn["transcript_delex"].strip()
                data_detail["turn_sys_delex"] = turn["system_transcript_delex"].strip()
                data_detail["belief_state_vec"] = ast.literal_eval(turn["belief_state_vec"])
                data_detail["db_pointer"] = ast.literal_eval(turn["db_pointer"])
                data_detail["dialog_history"] = list(dialog_history)
                data_detail["dialog_history_delex"] = list(dialog_history_delex)
                data_detail["belief"] = belief_dict
                data_detail["del_belief"] = turn_slot_dict
                data_detail["slot_gate"] = gates
                data_detail["slot_values"] = slot_values
                data_detail["sys_act"] = cur_sys_acts
                data_detail["turn_slot"] = turn_slot
                
                if not args["only_last_turn"]:
                    data.append(data_detail)

                dialog_history.append(turn["system_transcript"])
                dialog_history.append(turn["transcript"])
                dialog_history_delex.append(turn["system_transcript_delex"])
                dialog_history_delex.append(turn["transcript_delex"])
                response_candidates.add(str(data_detail["turn_sys"]))
            
            if args["only_last_turn"]:
                data.append(data_detail)
            
            cnt_lin += 1
            if(max_line and cnt_lin >= max_line):
                break
    return data, ontology, response_candidates


def get_slot_information(args, ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    ontology_new = collections.OrderedDict()
    for k, v in ontology_domains.items():
        name = k.replace(" ","").lower() if ("book" not in k) else k.lower() 
        
        if args["ontology_version"] != "":
            v = clean_original_ontology(v)
        
        ontology_new[name] = {"none":0, "do n't care":1}
        for vv in v:
            if vv not in ontology_new[name].keys():
                ontology_new[name][vv] = len(ontology_new[name])
    return ontology_new


def prepare_data_multiwoz(args):
    max_line = args["max_line"]
    
    version = "2.1"
    print("[Info] Using Version", version)
    
    file_trn = os.path.join(args["data_path"], 'MultiWOZ-{}/train_dials.json'.format(version))
    file_dev = os.path.join(args["data_path"], 'MultiWOZ-{}/dev_dials.json'.format(version))
    file_tst = os.path.join(args["data_path"], 'MultiWOZ-{}/test_dials.json'.format(version))

    path_to_ontology_mapping = os.path.join(args["data_path"], 
                                            "MultiWOZ-{}/ontology-mapping{}.json".format(version, args["ontology_version"]))
    
    if os.path.exists(path_to_ontology_mapping):
        print("[Info] Load from old complete ontology from version {}...".format(args["ontology_version"]))
        ontology_mapping = json.load(open(path_to_ontology_mapping, 'r'))
        update_ont_flag = False
    else:
        print("[Info] Creating new ontology for version {}...".format(args["ontology_version"]))
        ontology = json.load(open(os.path.join(args["data_path"], "MultiWOZ-{}/ontology.json".format(version)), 'r'))
        ontology_mapping = get_slot_information(args, ontology)
        update_ont_flag = True

    dialog_act = json.load(open(os.path.join(args["data_path"], "MultiWOZ-{}/dialogue_acts.json".format(version)), 'r'))
    
    _example_type = "turn"
    
    pair_trn, ontology_mapping, resp_cand_trn = globals()["read_langs_{}".format(_example_type)](args, 
                                                                                  file_trn, 
                                                                                  ontology_mapping, 
                                                                                  dialog_act, 
                                                                                  max_line, 
                                                                                  update_ont_flag)
    
    pair_dev, ontology_mapping, resp_cand_dev = globals()["read_langs_{}".format(_example_type)](args, 
                                                                                  file_dev, 
                                                                                  ontology_mapping, 
                                                                                  dialog_act, 
                                                                                  max_line, 
                                                                                  update_ont_flag)
    
    pair_tst, ontology_mapping, resp_cand_tst = globals()["read_langs_{}".format(_example_type)](args, 
                                                                                  file_tst, 
                                                                                  ontology_mapping, 
                                                                                  dialog_act, 
                                                                                  max_line, 
                                                                                  update_ont_flag)

    
    if not os.path.exists(path_to_ontology_mapping):
        print("[Info] Dumping complete ontology...")
        json.dump(ontology_mapping, open(path_to_ontology_mapping, "w"), indent=4)
    
    print("Read %s pairs train from MultiWOZ" % len(pair_trn))
    print("Read %s pairs valid from MultiWOZ" % len(pair_dev))
    print("Read %s pairs test from MultiWOZ"  % len(pair_tst))  
    
    print('args["task_name"]', args["task_name"])
    
    if args["task_name"] == "dst":
        meta_data = {"slots":ontology_mapping, "num_labels": len(ontology_mapping)}
    elif args["task_name"] == "turn_domain":
        domain_set = set([d["turn_domain"] for d in pair_trn])
        domain_dict = {d:i for i, d in enumerate(domain_set)}
        meta_data = {"turn_domain":domain_dict, "num_labels": len(domain_dict)}
    elif args["task_name"] == "turn_slot":
        turn_slot_list = []
        for d in pair_trn:
            turn_slot_list += d["turn_slot"]
        turn_slot_list = list(set(turn_slot_list))
        turn_slot_mapping = {d:i for i, d in enumerate(turn_slot_list)}
        meta_data = {"turn_slot":turn_slot_mapping, "num_labels": len(turn_slot_mapping)}
    elif args["task_name"] == "sysact":
        act_set = set()
        for pair in [pair_tst, pair_dev, pair_trn]:
            for p in pair:
                if type(p["sys_act"]) == list:
                    for sysact in p["sys_act"]:
                        act_set.update(sysact)
                else:
                    act_set.update(p["sys_act"])
        print("act_set", len(act_set), act_set)
        sysact_lookup = {sysact:i for i, sysact in enumerate(act_set)}
        meta_data = {"sysact":sysact_lookup, "num_labels":len(act_set)}
    elif args["task_name"] == "rs":
        print("resp_cand_trn", len(resp_cand_trn))
        print("resp_cand_dev", len(resp_cand_dev))
        print("resp_cand_tst", len(resp_cand_tst))
        meta_data = {"num_labels":0, "resp_cand_trn": resp_cand_trn}
    else:
        meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

