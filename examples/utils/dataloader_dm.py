import torch
import torch.utils.data as data
# from .config import *
from .utils_function import to_cuda, merge, merge_multi_response, merge_sent_and_word

class Dataset_dm(torch.utils.data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, tokenizer, args, unified_meta, mode, max_length=512):
        """Reads source and target sequences from txt files."""
        self.data = data_info
        self.tokenizer = tokenizer
        self.num_total_seqs = len(data_info["ID"])
        self.usr_token = args["usr_token"]
        self.sys_token = args["sys_token"]
        self.max_length = max_length
        self.args = args
        self.unified_meta = unified_meta
        
        if "bert" in self.args["model_type"] or "electra" in self.args["model_type"]:
            self.start_token = self.tokenizer.cls_token  
            self.sep_token = self.tokenizer.sep_token
        else:
            self.start_token = self.tokenizer.bos_token
            self.sep_token = self.tokenizer.eos_token
        
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        
        if self.args["example_type"] == "turn":
            dialog_history_str = self.get_concat_context(self.data["dialog_history"][index])
            context_plain = self.concat_dh_sys_usr(dialog_history_str, self.data["turn_sys"][index], self.data["turn_usr"][index])
            context = self.preprocess(context_plain)
            act_plain = self.data["sys_act"][index]
            
            turn_sys_plain = "{} {}".format(self.sys_token, self.data["turn_sys"][index])
            turn_sys = self.preprocess(turn_sys_plain)
            
            act_one_hot = [0] * len(self.unified_meta["sysact"])
            for act in act_plain:
                act_one_hot[self.unified_meta["sysact"][act]] = 1
            
        elif self.args["example_type"] == "dial":
            #TODO
            print("Not Implemented dial for nlu yet...")
            
        item_info = {
            "ID":self.data["ID"][index], 
            "turn_id":self.data["turn_id"][index], 
            "context":context, 
            "context_plain":context_plain,
            "sysact":act_one_hot,
            "sysact_plain":act_plain, 
            "turn_sys":turn_sys}
            
        return item_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence):
        """Converts words to ids."""
        tokens = self.tokenizer.tokenize(self.start_token) + self.tokenizer.tokenize(sequence)[-self.max_length+1:]
        story = torch.Tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        return story
    
    def concat_dh_sys_usr(self, dialog_history, sys, usr):
        return dialog_history + " {} ".format(self.sys_token) + " {} ".format(self.sep_token) + sys + " {} ".format(self.usr_token) + usr

    def get_concat_context(self, dialog_history):
        dialog_history_str = ""
        for ui, uttr in enumerate(dialog_history):
            if ui%2 == 0:
                dialog_history_str += "{} {} ".format(self.sys_token, uttr)
            else:
                dialog_history_str += "{} {} ".format(self.usr_token, uttr)
        dialog_history_str = dialog_history_str.strip()
        return dialog_history_str


def collate_fn_dm_turn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    turn_sys, _ = merge(item_info["turn_sys"])
    sysact = torch.tensor(item_info["sysact"]).float()

    item_info["context"] = to_cuda(src_seqs)
    item_info["context_len"] = src_lengths
    item_info["sysact"] = to_cuda(sysact)
    item_info["turn_sys"] = to_cuda(turn_sys)
    
    return item_info


def collate_fn_nlu_dial(data):
    # TODO
    return 

