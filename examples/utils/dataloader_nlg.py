import torch
import torch.utils.data as data
import random

from .utils_function import to_cuda, merge
# from .config import *


class Dataset_nlg(torch.utils.data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, tokenizer, args, unified_meta, mode, max_length=512, max_sys_resp_len=50):
        """Reads source and target sequences from txt files."""
        self.data = data_info
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_total_seqs = len(data_info["ID"])
        self.usr_token = args["usr_token"]
        self.sys_token = args["sys_token"]
        self.unified_meta = unified_meta
        self.args = args
        self.mode = mode
        self.max_sys_resp_len = max_sys_resp_len
        self.others = unified_meta["others"]
        
        if "bert" in self.args["model_type"] or "electra" in self.args["model_type"]:
            self.start_token = self.tokenizer.cls_token  
            self.sep_token = self.tokenizer.sep_token
        else:
            self.start_token = self.tokenizer.bos_token
            self.sep_token = self.tokenizer.eos_token
        
        if self.args["nb_neg_sample_rs"] != 0:
            self.resp_cand_trn = list(self.unified_meta["resp_cand_trn"])
            random.shuffle(self.resp_cand_trn)
        
        
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        
        if self.args["example_type"] == "turn":
            context_plain = self.get_concat_context(self.data["dialog_history"][index])
            context_plain_delex = self.get_concat_context(self.data["dialog_history_delex"][index])
            context = self.preprocess(context_plain)
            context_delex = self.preprocess(context_plain_delex)
            response_plain = "{} ".format(self.sys_token) + self.data["turn_sys"][index]
            response = self.preprocess(response_plain)[:self.max_sys_resp_len]
            response_plain_delex = "{} ".format(self.sys_token) + self.data["turn_sys_delex"][index]
            response_delex = self.preprocess(response_plain_delex)
            utterance_plain = "{} ".format(self.usr_token) + self.data["turn_usr"][index]
            utterance = self.preprocess(utterance_plain)
            utterance_plain_delex = "{} ".format(self.usr_token) + self.data["turn_usr_delex"][index]
            utterance_delex = self.preprocess(utterance_plain_delex)
        else:
            raise NotImplementedError
        
        item_info = {
            "ID":self.data["ID"][index], 
            "turn_id":self.data["turn_id"][index], 
            "context":context,
            "context_plain":context_plain,
            "context_delex":context_delex,
            "context_plain_delex":context_plain_delex,
            "response":response,
            "response_plain":response_plain,
            "response_delex":response_delex,
            "response_plain_delex":response_plain_delex,
            "utterance":utterance,
            "utterance_plain":utterance_plain,
            "utterance_delex":utterance_delex,
            "utterance_plain_delex":utterance_plain_delex}
        
        if self.args["nb_neg_sample_rs"] != 0 and self.mode == "train":
            #random.shuffle(self.resp_cand_trn)
            
            if self.args["sample_negative_by_kmeans"]:
                try:
                    cur_cluster = self.others["ToD_BERT_SYS_UTTR_KMEANS"][self.data["turn_sys"][index]]
                    candidates = self.others["KMEANS_to_SENTS"][cur_cluster]
                    nb_selected = min(self.args["nb_neg_sample_rs"], len(candidates))
                    try:
                        start_pos = random.randint(0, len(candidates)-nb_selected-1)
                    except:
                        start_pos = 0
                    sampled_neg_resps = candidates[start_pos:start_pos+nb_selected]
                
                except:
                    start_pos = random.randint(0, len(self.resp_cand_trn)-self.args["nb_neg_sample_rs"]-1)
                    sampled_neg_resps = self.resp_cand_trn[start_pos:start_pos+self.args["nb_neg_sample_rs"]]  
            
            elif self.args["sample_negative_by_distance"]:
                sampled_neg_resps = self.others["sampled_neg_responses"][self.data["turn_sys"][index]]
            else:
                start_pos = random.randint(0, len(self.resp_cand_trn)-self.args["nb_neg_sample_rs"]-1)
                sampled_neg_resps = self.resp_cand_trn[start_pos:start_pos+self.args["nb_neg_sample_rs"]]
            
            neg_resp_arr, neg_resp_idx_arr = [], []
            for neg_resp in sampled_neg_resps:
                neg_resp_plain = "{} ".format(self.sys_token) + neg_resp
                neg_resp_idx = self.preprocess(neg_resp_plain)[:self.max_sys_resp_len]
                neg_resp_idx_arr.append(neg_resp_idx)
                neg_resp_arr.append(neg_resp_plain)
            item_info["neg_resp_idx_arr"] = neg_resp_idx_arr
            item_info["neg_resp_arr"] = neg_resp_arr

        return item_info

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence):
        """Converts words to ids."""
        #story = torch.Tensor(self.tokenizer.encode(sequence[-self.max_length:])) #, add_special_tokens=False, return_tensors="pt")
        tokens = self.tokenizer.tokenize(self.start_token) + self.tokenizer.tokenize(sequence)[-self.max_length+1:]
        story = torch.Tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        return story
    
    def get_concat_context(self, dialog_history):
        dialog_history_str = ""
        for ui, uttr in enumerate(dialog_history):
            if ui%2 == 0:
                dialog_history_str += "{} {} ".format(self.sys_token, uttr)
            else:
                dialog_history_str += "{} {} ".format(self.usr_token, uttr)
        dialog_history_str = dialog_history_str.strip()
        return dialog_history_str
        

def collate_fn_nlg_turn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    # augment negative samples
    if "neg_resp_idx_arr" in item_info.keys():
        neg_resp_idx_arr = []
        for arr in item_info['neg_resp_idx_arr']:
            neg_resp_idx_arr += arr
        
        # remove neg samples that are the same as one of the gold responses
        #print('item_info["response"]', item_info["response"])
        #print('neg_resp_idx_arr', neg_resp_idx_arr)
        
        for bi, arr in enumerate(item_info['neg_resp_arr']):
            for ri, neg_resp in enumerate(arr):
                if neg_resp not in item_info["response_plain"]:
                    item_info["response"] += [item_info['neg_resp_idx_arr'][bi][ri]]
                
        #neg_resp_idx_arr = [ng for ng in neg_resp_idx_arr if ng not in item_info["response"]] 
        #item_info["response"] += neg_resp_idx_arr
        
    # merge sequences    
    context, context_lengths = merge(item_info['context'])
    context_delex, context_delex_lengths = merge(item_info['context_delex'])
    response, response_lengths = merge(item_info["response"])
    response_delex, response_delex_lengths = merge(item_info["response_delex"])
    utterance, utterance_lengths = merge(item_info["utterance"])
    utterance_delex, utterance_delex_lengths = merge(item_info["utterance_delex"])
    
    #print("context", context.size())
    #print("response", response.size())
    
    item_info["context"] = to_cuda(context)
    item_info["context_lengths"] = context_lengths
    #item_info["context_delex"] = to_cuda(context_delex)
    #item_info["context_delex_lengths"] = context_delex_lengths
    item_info["response"] = to_cuda(response)
    item_info["response_lengths"] = response_lengths
    #item_info["response_delex"] = to_cuda(response_delex)
    #item_info["response_delex_lengths"] = response_delex_lengths
    item_info["utterance"] = to_cuda(utterance)
    item_info["utterance_lengths"] = response_lengths
    #item_info["utterance_delex"] = to_cuda(utterance_delex)
    #item_info["utterance_delex_lengths"] = utterance_delex_lengths
    
    return item_info



