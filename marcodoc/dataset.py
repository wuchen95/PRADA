import os
import math
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from collections import namedtuple, defaultdict
from transformers import BertTokenizer
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)


class CollectionDataset:
    def __init__(self, collection_memmap_dir):
        self.pids = np.memmap(f"{collection_memmap_dir}/pids.memmap", dtype='int32',)
        self.lengths = np.memmap(f"{collection_memmap_dir}/lengths.memmap", dtype='int32',)
        self.collection_size = len(self.pids)
        self.token_ids = np.memmap(f"{collection_memmap_dir}/token_ids.memmap", 
                dtype='int32', shape=(self.collection_size, 512))
    
    def __len__(self):
        return self.collection_size

    def __getitem__(self, item):

        '''
        :param item: pid
        :return: doc content
        '''
        index = np.argwhere(self.pids == item)[0][0]

        return self.token_ids[index, :self.lengths[index]].tolist()


def load_queries(tokenize_dir, mode):
    queries = dict()
    for line in tqdm(open(f"{tokenize_dir}/msmarco-doc{mode}-queries.tokenized.json"), desc="queries"):
        data = json.loads(line)
        queries[int(data['id'])] = data['ids']
    return queries


def load_queries_surr_model(tokenize_dir, mode):
    queries = dict()
    for line in tqdm(open(f"{tokenize_dir}/msmarco-docdev-queries.tokenized.json"), desc="queries"):
        data = json.loads(line)
        queries[int(data['id'])] = data['ids']
    return queries


def load_querydoc_pairs(msmarco_dir, mode):
    qrels = defaultdict(set)
    qids, pids, labels = [], [], []
    if mode == "train":
        for line in tqdm(open(f"{msmarco_dir}/train_triples_ids_10neg"),
                desc="load train triples"):
            qid, pos_pid, neg_pid = line.split("\t")
            qid, pos_pid, neg_pid = int(qid), int(pos_pid[1:]), int(neg_pid[1:])
            qids.append(qid)
            pids.append(pos_pid)
            labels.append(1)
            qids.append(qid)
            pids.append(neg_pid)
            labels.append(0)

    else:
        for line in open(f"{msmarco_dir}/msmarco-doc{mode}-top100.tsv"):  ### mode = dev
            qid, _, pid, _, _, _ = line.split(" ")
            qids.append(int(qid))
            pids.append(int(pid[1:]))
    # qrels = dict(qrels)
    if not mode == "train":
        labels, qrels = None, None
    return qids, pids, labels   #, qrels


def load_querydoc_pairs_surr_model(msmarco_dir, mode):
    qids, pids, labels = [], [], []
    if mode == "train":
        for line in tqdm(open(f"{msmarco_dir}/surrogate_model_training_triples"),
                desc="load train triples"):
            qid, pos_pid, neg_pid = line.split("\t")
            qid, pos_pid, neg_pid = int(qid), int(pos_pid), int(neg_pid)
            qids.append(qid)
            pids.append(pos_pid)
            labels.append(1)
            qids.append(qid)
            pids.append(neg_pid)
            labels.append(0)
    else:
        for line in open(f"{msmarco_dir}/msmarco-doc{mode}-top100.tsv"):  ### mode = dev
            qid, _, pid, _, _, _ = line.split(" ")
            qids.append(int(qid))
            pids.append(int(pid[1:]))
    if not mode == "train":
        labels, qrels = None, None
    return qids, pids, labels   #, qrels


def load_querydoc_pairs_by_file_path(qd_file_path):
    qids, pids, labels = [], [], []

    # file_format: qid \t did \t rank
    for line in open(qd_file_path):  # mode = dev
        qid, pid, _ = line.split("\t")
        qids.append(int(qid))
        pids.append(int(pid))
    return qids, pids, labels   #, qrels


class MSMARCODataset(Dataset):
    def __init__(self, mode, msmarco_dir,
            collection_memmap_dir, tokenize_dir, tokenizer_dir,
            max_query_length=64, max_doc_length=445):

        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.labels = load_querydoc_pairs(msmarco_dir, mode)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        '''
        '''
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
   
            label = self.labels[item]
            ret_val["label"] = label
        return ret_val


class MSMARCOSURRMODELDataset(Dataset):
    def __init__(self, mode, msmarco_dir,
            collection_memmap_dir, tokenize_dir, tokenizer_dir,
            max_query_length=64, max_doc_length=445):

        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries_surr_model(tokenize_dir, mode)
        self.qids, self.pids, self.labels = load_querydoc_pairs_surr_model(msmarco_dir, mode)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        '''
        :param item:
        :return:
        '''
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
            label = self.labels[item]
            ret_val["label"] = label
        return ret_val


class MSMARCODataset_file_qd(Dataset):
    def __init__(self, mode, q_d_file_path,
            collection_memmap_dir, tokenize_dir, tokenizer_dir,
            max_query_length=64, max_doc_length=445):

        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.labels = load_querydoc_pairs_by_file_path(q_d_file_path)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        '''
        :param item:
        :return:
        '''
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
            #
            label = self.labels[item]
            ret_val["label"] = label
        return ret_val


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def get_collate_function(mode):
    def collate_function(batch):
        input_ids_lst = [x["query_input_ids"] + x["doc_input_ids"] for x in batch]
        token_type_ids_lst = [[0]*len(x["query_input_ids"]) + [1]*len(x["doc_input_ids"]) 
            for x in batch]
        #
        position_ids_lst = [list(range(len(x["query_input_ids"]) + len(x["doc_input_ids"]))) for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
        }
        qid_lst = [x['qid'] for x in batch]
        docid_lst = [x['docid'] for x in batch]
        if mode == "train":

            data["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.int64)
        return data, qid_lst, docid_lst
    return collate_function
