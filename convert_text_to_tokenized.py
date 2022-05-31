import os, gc
import json
import argparse
from tqdm import tqdm
from transformers import BertTokenizer
from multiprocessing import Pool

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_doc_content(input_args):
    i = 0
    doc_list = input_args
    res_lines = []
    for doc in doc_list:
        text = doc['content']
        doc_id = doc['docid']
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        res_json = json.dumps(
            {"id": doc_id, "ids": ids}
        )
        res_lines.append(res_json)
        i += 1
        if i % 10000 == 0:
            print('finished', i, 'lines')
    print('finished a subprocess!')
    del doc_list, input_args
    gc.collect()

    return res_lines


def tokenize_file(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))
    with open(output_file, 'w') as outFile:
        for line in tqdm(open(input_file), total=total_size,
                desc=f"Tokenize: {os.path.basename(input_file)}"):
            seq_id, text = line.split("\t")
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            outFile.write(json.dumps(
                {"id":seq_id, "ids":ids}
            ))
            outFile.write("\n")


def tokenize_MSMARCOdocument_ranking_collection(input_file, output_file):
    '''
    :param tokenizer:
    :param input_file:
    :param output_file:
    :return:
    '''
    total_size = sum(1 for _ in open(input_file))
    docs_list = []

    # read all the docs
    for line in tqdm(open(input_file), total=total_size,
            desc=f"Read contents: {os.path.basename(input_file)}"):
        doc_id, url, title, body = line.split("\t")
        text = title + body
        doc_dict = {}
        doc_dict['docid'] = doc_id
        doc_dict['content'] = text
        docs_list.append(doc_dict)

    new_docs_list = docs_list

    # seperate the data
    docs_list_seps = []
    docs_list_s = []
    for j in range(len(new_docs_list)):
        if (j != 0) and (j % 20000 == 0):
            docs_list_seps.append(docs_list_s)
            docs_list_s = []
            print("finished given", j, "docs")
        docs_list_s.append(new_docs_list[j])
        if j == (len(new_docs_list) - 1):
            docs_list_seps.append(docs_list_s)
            docs_list_s = []

    del docs_list, new_docs_list
    gc.collect()

    # multi-process the data
    arg_list = [docs_list_ss for docs_list_ss in docs_list_seps]

    pool = Pool(20)
    res = pool.map(tokenize_doc_content, arg_list)

    del arg_list
    gc.collect()

    # write to the file
    write_index = 0
    outFile = open(output_file, 'w')
    for doc_list in res:
        for doc_json in doc_list:
            outFile.write(doc_json)
            outFile.write("\n")
            write_index += 1
            if write_index % 30000 == 0:
                print('finished writing', write_index, 'lines')


def tokenize_queries(args, tokenizer):
    for mode in ["train", "dev"]:#, "eval.small", "dev", "eval", "train"]:
        query_input = f"{args.output_dir}/queries.{mode}.tsv"
        query_output = f"{args.output_dir}/tokenized_query_collection/msmarco-passage{mode}-queries.tokenized.json"
        tokenize_file(tokenizer, query_input, query_output)


def tokenize_collection(args, tokenizer):
    collection_output = f"{args.output_dir}/tokenized_query_collection/collection.tokenized.json"
    tokenize_file(tokenizer, f"{args.msmarco_dir}/collection.tsv", collection_output)


def tokenize_collection_document_ranking(args, tokenizer):
    collection_input = f"{args.msmarco_dir}/corpus/msmarco-docs.tsv"
    collection_output = f"{args.output_dir}/tokenized_query_collection/collection.tokenized.json_26tofinal"
    tokenize_MSMARCOdocument_ranking_collection(tokenizer, collection_input, collection_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument("--output_dir", type=str, default="./data/tokenize")
    parser.add_argument("--tokenize_queries", action="store_true")
    parser.add_argument("--tokenize_collection", action="store_true")
    parser.add_argument("--dataset_type", type=str, default="document_ranking")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.tokenize_queries:
        tokenize_queries(args, tokenizer)  
    if args.tokenize_collection:
        if args.dataset_type == 'document_ranking':
            tokenize_collection_document_ranking(args, tokenizer)
        else:
            tokenize_collection(args, tokenizer)
