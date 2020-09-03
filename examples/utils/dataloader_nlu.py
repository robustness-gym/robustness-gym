import torch
import torch.utils.data as data
from .utils_function import to_cuda, merge, merge_multi_response, merge_sent_and_word

class Dataset_nlu(torch.utils.data.Dataset):
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

        context_plain = "{} {} {} {} {}".format(self.start_token, 
                                                self.sys_token, 
                                                self.data["turn_sys"][index], 
                                                self.usr_token, 
                                                self.data["turn_usr"][index])
        context = self.preprocess(context_plain)
        intent_plain = self.data["intent"][index]

        turn_sys_plain = "{} {}".format(self.sys_token, self.data["turn_sys"][index])
        turn_sys = self.preprocess(turn_sys_plain)

        try:
            intent_idx = self.unified_meta["intent"][intent_plain]
        except:
            intent_idx = -100

        try:
            domain_idx = self.unified_meta["turn_domain"][self.data["turn_domain"][index]]
        except:
            domain_idx = -100

        try:
            turn_slot_one_hot = [0] * len(self.unified_meta["turn_slot"])
            for ts in self.data["turn_slot"][index]:
                turn_slot_one_hot[self.unified_meta["turn_slot"][ts]] = 1
        except:
            turn_slot_one_hot = -100
            
        item_info = {
            "ID":self.data["ID"][index], 
            "turn_id":self.data["turn_id"][index], 
            "turn_domain":self.data["turn_domain"][index],
            "context":context, 
            "context_plain":context_plain,
            "intent":intent_idx,
            "intent_plain":intent_plain,
            "domain_plain":self.data["turn_domain"][index],
            "turn_domain": domain_idx,
            "turn_sys":turn_sys,
            "turn_slot":turn_slot_one_hot,
            "turn_sys_plain":turn_sys_plain
        }
            
        return item_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence):
        """Converts words to ids."""
        tokens = self.tokenizer.tokenize(self.start_token) + self.tokenizer.tokenize(sequence)[-self.max_length+1:]
        story = torch.Tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        return story


def collate_fn_nlu_turn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    turn_sys, _ = merge(item_info["turn_sys"])
    intent = torch.tensor(item_info["intent"])
    turn_domain = torch.tensor(item_info["turn_domain"])
    turn_slot = torch.tensor(item_info["turn_slot"]).float()
    
    item_info["context"] = to_cuda(src_seqs)
    item_info["context_len"] = src_lengths
    item_info["intent"] = to_cuda(intent)
    item_info["turn_domain"] = to_cuda(turn_domain)
    item_info["turn_sys"] = to_cuda(turn_sys)
    item_info["turn_slot"] = to_cuda(turn_slot)
    
    return item_info


def collate_fn_nlu_dial(data):
    # TODO
    return 

