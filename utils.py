import os
import re
import random
from collections import defaultdict
import subprocess


def generate_rank(input_path, output_path):
    score_dict = defaultdict(list)
    for line in open(input_path):
        query_id, para_id, score = line.split("\t")
        score_dict[int(query_id)].append((float(score), int(para_id)))
    with open(output_path, "w") as outFile:
        for query_id, para_lst in score_dict.items():
            random.shuffle(para_lst)
            para_lst = sorted(para_lst, key=lambda x:x[0], reverse=True)
            for rank_idx, (score, para_id) in enumerate(para_lst):
                outFile.write("{}\t{}\t{}\n".format(query_id, para_id, rank_idx+1))


def eval_results(run_file_path,
        eval_script="./ms_marco_eval.py", 
        qrels="./data/msmarco_doc/msmarco-docdev-qrels.tsv" ):
    assert os.path.exists(eval_script) and os.path.exists(qrels)
    dataset = 'marcodoc'
    result = subprocess.check_output(['python', eval_script, qrels, run_file_path, dataset])
    match = re.search('MRR @100: ([\d.]+)', result.decode('utf-8'))
    mrr = float(match.group(1))
    return mrr


def eval_passage_results(run_file_path,
        eval_script="./ms_marco_eval.py",
        qrels="./data/msmarco_pas/qrels.dev.small.tsv" ):
    assert os.path.exists(eval_script) and os.path.exists(qrels)
    dataset = 'marcopassage'
    result = subprocess.check_output(['python', eval_script, qrels, run_file_path, dataset])
    match = re.search('MRR @100: ([\d.]+)', result.decode('utf-8'))
    mrr = float(match.group(1))
    return mrr


def read_qd_pairs(QD_FILE_PATH):
    qds = {}
    with open(QD_FILE_PATH, 'r', encoding='utf-8') as qdsf:
        for line in qdsf:
            ss = line.strip().split('\t')
            qid = int(ss[0])
            did = int(ss[1])
            if qid not in qds:
                qds[qid] = []
            qds[qid].append(did)
    return qds


def read_qds_pairs(QD_FILE_PATH):
    qds = {}
    with open(QD_FILE_PATH, 'r', encoding='utf-8') as qdsf:
        for line in qdsf:
            ss = line.strip().split('\t')
            qid = int(ss[0])
            did = int(ss[1])
            score = float(ss[2])
            if qid not in qds:
                qds[qid] = {}
            qds[qid][did] = score
    return qds