import torch
import numpy as np

PAD_token = 0

def to_cuda(x):
    if torch.cuda.is_available(): x = x.cuda()
    return x


def merge(sequences, ignore_idx=None):
    '''
    merge from batch * sent_len to batch * max_len 
    '''
    pad_token = PAD_token if type(ignore_idx)==type(None) else ignore_idx
    lengths = [len(seq) for seq in sequences]
    max_len = 1 if max(lengths)==0 else max(lengths)
    padded_seqs = torch.ones(len(sequences), max_len).long() * pad_token 
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
    return padded_seqs, lengths

def merge_multi_response(sequences, ignore_idx=None):
    '''
    merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
    '''
    pad_token = PAD_token if type(ignore_idx)==type(None) else ignore_idx
    lengths = []
    for bsz_seq in sequences:
        length = [len(v) for v in bsz_seq]
        lengths.append(length)
    max_len = max([max(l) for l in lengths])
    padded_seqs = []
    for bsz_seq in sequences:
        pad_seq = []
        for v in bsz_seq:
            v = v + [pad_token] * (max_len-len(v))
            pad_seq.append(v)
        padded_seqs.append(pad_seq)
    padded_seqs = torch.tensor(padded_seqs).long()
    lengths = torch.tensor(lengths)
    return padded_seqs, lengths

def merge_sent_and_word(sequences, ignore_idx=None):
    '''
    merge from batch * nb_sent * nb_word to batch * max_nb_sent * max_nb_word
    '''

    max_nb_sent = max([len(seq) for seq in sequences])
    max_nb_word, lengths = [], []
    for seq in sequences:
        length = [len(sent) for sent in seq]
        max_nb_word += length
        lengths.append(length)
    max_nb_word = max(max_nb_word)
    
    pad_token = PAD_token if type(ignore_idx)==type(None) else ignore_idx
    padded_seqs = np.ones((len(sequences), max_nb_sent, max_nb_word)) * pad_token 
    
    for i, seq in enumerate(sequences):
        for ii, sent in enumerate(seq):
            padded_seqs[i, ii, :len(sent)] = np.array(sent)
    padded_seqs = torch.LongTensor(padded_seqs)
    padded_seqs = padded_seqs.detach() 
    return padded_seqs, lengths


def get_input_example(example_type):
    
    if example_type == "turn":
        
        data_detail = {
            "ID":"", 
            "turn_id":0, 
            "domains":[], 
            "turn_domain":[],
            "turn_usr":"",
            "turn_sys":"",
            "turn_usr_delex":"",
            "turn_sys_delex":"",
            "belief_state_vec":[],
            "db_pointer":[],
            "dialog_history":[], 
            "dialog_history_delex":[], 
            "belief":{},
            "del_belief":{},
            "slot_gate":[],
            "slot_values":[],
            "slots":[],
            "sys_act":[], 
            "usr_act":[], 
            "intent":"",
            "turn_slot":[]}
        
    elif example_type == "dial":

        data_detail = {
            "ID":"", 
            "turn_id":[], 
            "domains":[], 
            "turn_domain":[],
            "turn_usr":[],
            "turn_sys":[],
            "turn_usr_delex":[],
            "turn_sys_delex":[],
            "belief_state_vec":[],
            "db_pointer":[],
            "dialog_history":[], 
            "dialog_history_delex":[], 
            "belief":[],
            "del_belief":[],
            "slot_gate":[],
            "slot_values":[],
            "slots":[],
            "sys_act":[], 
            "usr_act":[], 
            "intent":[],
            "turn_slot":[]}
        
    return data_detail
