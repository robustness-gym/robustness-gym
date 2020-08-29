import os.path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
import numpy as np

from transformers import *

def _gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BeliefTracker(nn.Module):
    def __init__(self, args, fix_encoder=False, learning_rate=5e-5, grad_clip=1.0):
        super(BeliefTracker, self).__init__()
        
        self.args = args
        self.n_gpu = 1
        self.num_labels = [len(v) for k, v in args["unified_meta"]["slots"].items()]
        self.num_slots = len(self.num_labels)
        self.tokenizer = args["tokenizer"]
        self.grad_clip = grad_clip
        
        self.slots = [k for k, v in self.args["unified_meta"]["slots"].items()]
        self.slot_value2id_dict = self.args["unified_meta"]["slots"]
        self.slot_id2value_dict = {}
        for k, v in self.slot_value2id_dict.items():
            self.slot_id2value_dict[k] = {vv: kk for kk, vv in v.items()}

        ### Utterance Encoder
        self.utterance_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"])
        self.bert_output_dim = args["config"].hidden_size
        if fix_encoder:
            print("[Info] Utterance Encoder does not requires grad...")
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"])
        print("[Info] SV Encoder does not requires grad...")
        for p in self.sv_encoder.parameters():
            p.requires_grad = False

        #self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in self.num_labels])
        
        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)
        
        ### My Add
        self.project_W_1 = nn.ModuleList([nn.Linear(self.bert_output_dim, self.bert_output_dim) \
                                          for _ in range(self.num_slots)])
        self.project_W_2 = nn.ModuleList([nn.Linear(2*self.bert_output_dim, self.bert_output_dim) \
                                          for _ in range(self.num_slots)])
        self.project_W_3 = nn.ModuleList([nn.Linear(self.bert_output_dim, 1) \
                                      for _ in range(self.num_slots)])
        self.linear = nn.Linear(400, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)
        
        self.start_token = self.tokenizer.cls_token if "bert" in self.args["model_type"] else self.tokenizer.bos_token
        self.sep_token = self.tokenizer.sep_token if "bert" in self.args["model_type"] else self.tokenizer.eos_token
        
        ## Prepare Optimizer
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': learning_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': learning_rate},
            ]
            return optimizer_grouped_parameters

        if self.n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.module)

        
        self.optimizer = AdamW(optimizer_grouped_parameters,
                                 lr=learning_rate)
                                 #warmup=args["warmup_proportion"],
                                 #t_total=t_total)
    
        self.initialize_slot_value_lookup()
        
    def optimize(self):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()
        
    
    def initialize_slot_value_lookup(self, max_seq_length=32):

        self.sv_encoder.eval()
        
        label_ids = []
        for dslot, value_dict in self.args["unified_meta"]["slots"].items():
            label_id = []
            value_dict_rev = {v:k for k, v in value_dict.items()}
            for i in range(len(value_dict)):
                label = value_dict_rev[i]
                label = " ".join([i for i in label.split(" ") if i != ""])

                label_tokens = [self.start_token] + self.tokenizer.tokenize(label) + [self.sep_token]
                label_token_ids = self.tokenizer.convert_tokens_to_ids(label_tokens)
                label_len = len(label_token_ids)

                label_padding = [0] * (max_seq_length - len(label_token_ids))
                label_token_ids += label_padding
                assert len(label_token_ids) == max_seq_length
                label_id.append(label_token_ids)
                
            label_id = torch.tensor(label_id).long()
            label_ids.append(label_id)

        for s, label_id in enumerate(label_ids):
            inputs = {"input_ids":label_id, "attention_mask":(label_id > 0).long()}
            
            if "bert" in self.args["model_type"]:
                hid_label = self.sv_encoder(**inputs)[0]
                hid_label = hid_label[:, 0, :]
            elif self.args["model_type"] == "gpt2":
                hid_label = self.sv_encoder(**inputs)[0]
                hid_label = hid_label.mean(1)
            elif self.args["model_type"] == "dialogpt":
                transformer_outputs = self.sv_encoder.transformer(**inputs)[0]
                hid_label = transformer_outputs.mean(1)
            
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")

    def forward(self, data):#input_ids, input_len, labels, gate_label, n_gpu=1, target_slot=None):
        batch_size = data["context"].size(0)
        labels = data["belief_ontology"]
        
        # Utterance encoding
        inputs = {"input_ids": data["context"], "attention_mask":(data["context"] > 0).long()}

        if "bert" in self.args["model_type"]:
            hidden = self.utterance_encoder(**inputs)[0]
            hidden_rep = hidden[:, 0, :]
        elif self.args["model_type"] == "gpt2":
            hidden = self.utterance_encoder(**inputs)[0]
            hidden_rep = hidden.mean(1)
        elif self.args["model_type"] == "dialogpt":
            transformer_outputs = self.utterance_encoder.transformer(
                data["context"],
                attention_mask=(data["context"] > 0).long()
            )
            hidden = transformer_outputs[0]
            hidden_rep = hidden.mean(1)

        # Label (slot-value) encoding
        loss = 0
        pred_slot = []
        
        for slot_id in range(self.num_slots): ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight # v * d
            num_slot_labels = hid_label.size(0)

            _hidden = _gelu(self.project_W_1[slot_id](hidden_rep))
            _hidden = torch.cat([hid_label.unsqueeze(0).repeat(batch_size, 1, 1), _hidden.unsqueeze(1).repeat(1, num_slot_labels, 1)], dim=2)
            _hidden = _gelu(self.project_W_2[slot_id](_hidden))
            _hidden = self.project_W_3[slot_id](_hidden)
            _dist = _hidden.squeeze(2) # b * 1 * num_slot_labels

            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.unsqueeze(1))
            #output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist, labels[:, slot_id])
                loss += _loss

        predictions = torch.cat(pred_slot, 1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        if self.training: 
            self.loss_grad = loss
            self.optimize()
        
        outputs = {"loss":loss.item(), "pred":predictions, "label":labels} 
        
        return outputs

    def evaluation(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)

        slot_acc, joint_acc, slot_acc_total, joint_acc_total = 0, 0, 0, 0
        for pred, label in zip(preds, labels):
            joint = 0
            
            assert len(pred) == len(label)
            
            for i, p in enumerate(pred):
                pred_str = self.slot_id2value_dict[self.slots[i]][p]
                gold_str = self.slot_id2value_dict[self.slots[i]][label[i]]
                
                if pred_str == gold_str or pred_str in gold_str.split("|"):
                    slot_acc += 1
                    joint += 1
                slot_acc_total += 1
            
            if joint == len(pred):
                joint_acc += 1
                
            joint_acc_total += 1
        
        joint_acc = joint_acc / joint_acc_total
        slot_acc = slot_acc / slot_acc_total
        results = {"joint_acc":joint_acc, "slot_acc":slot_acc}
        print("Results 1: ", results)
                
        return results

