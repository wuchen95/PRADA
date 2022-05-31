import copy


import logging, argparse, os, time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import (BertConfig, BertTokenizer, AdamW,
                          get_linear_schedule_with_warmup)
from transformers import BertTokenizer

from myutils.semantic_helper import SemanticHelper
from utils import read_qds_pairs
from modeling import RankingBERT_Train
from marcodoc.dataset import MSMARCODataset_file_qd, get_collate_function, CollectionDataset
from myutils.word_recover.Bert_word_recover import BERTWordRecover
from myutils.attacker.attacker import Attacker
from mem_helper import occumpy_mem

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)


def read_resources(embed_path, embed_cos_matrix_path, embed_name,
                   attack_qd_path, black_model_ranked_list_score_path, bert_tokenizer,
                   bert_vocab_path, max_query_length, max_doc_length, collection_memmap_dir):
    '''
    :return:
    '''
    synonym_helper = SemanticHelper(embed_path, embed_cos_matrix_path)
    synonym_helper.build_vocab()
    synonym_helper.load_embedding_cos_sim_matrix()

    word_re = BERTWordRecover(embed_name, bert_tokenizer, bert_vocab_path, max_query_length, max_doc_length)

    attack_qds = read_qds_pairs(attack_qd_path)

    ori_ranked_list_qds = read_qds_pairs(black_model_ranked_list_score_path)

    collection = CollectionDataset(collection_memmap_dir)

    return synonym_helper, word_re, attack_qds, ori_ranked_list_qds, collection


def run_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=f"./data/train")

    parser.add_argument("--msmarco_dir", type=str,
                        default=f"./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str,
                        default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_doc_length", type=int, default=445)

    parser.add_argument("--batch_size", default=20, type=int)

    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--data_num_workers", default=0, type=int)

    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--bert_tokenizer_path",
                        default='',
                        type=str)
    parser.add_argument("--attack_qd_path", type=str,
                        default="")
    parser.add_argument("--attack_save_dir", type=str,
                        default="")
    parser.add_argument("--doc_list_length", type=int,
                        default=100)

    parser.add_argument("--eps", type=float, default=45)

    parser.add_argument("--max_iter", type=int, default=3)

    parser.add_argument("--simi_candi_topk", type=int, default=50)

    parser.add_argument("--simi_threshod", type=float, default=0.7)

    parser.add_argument("--previous_done", type=int, default=0)

    parser.add_argument("--max_attack_word_number", type=int, default=10)

    parser.add_argument("--find_word_search", action="store_true")

    parser.add_argument("--black_model_ranked_list_path", type=str,
                        default='')

    parser.add_argument("--black_model_ranked_list_score_path", type=str,
                        default='')

    parser.add_argument("--save_doc_tokens_path", type=str,
                        default='')

    parser.add_argument("--ori_model_path", type=str,
                        default='')

    parser.add_argument("--surrogate_model_path", type=str,
                        default='')

    parser.add_argument("--embed_path", type=str,
                        default='')

    parser.add_argument("--embed_cos_matrix_path", type=str,
                        default='')

    args = parser.parse_args()

    time_stamp = time.strftime("%b-%d_%H:%M:%S", time.localtime())
    args.log_dir = f"{args.output_dir}/log/{time_stamp}"
    args.model_save_dir = f"{args.output_dir}/models"
    args.bert_vocab_path = args.bert_tokenizer_path + '/vocab.txt'

    assert 100 % args.batch_size == 0
    return args


def remove_input_tensor_column(input_tensor, delete_index):

    res_tensor = input_tensor[
        torch.arange(input_tensor.size(0)) != delete_index]
    return res_tensor


def find_attack_doc_input_and_remove(batch_list, docid_list, attack_doc_id):

    res_batch_list = copy.deepcopy(batch_list)
    attack_doc_input = {}
    assert len(res_batch_list) == len(docid_list)
    for i in range(len(docid_list)):
        docids = docid_list[i]
        for j in range(len(docids)):
            did = docids[j]
            if did == attack_doc_id:
                attack_doc_input["input_ids"] = res_batch_list[i]['input_ids'][
                    j]
                attack_doc_input["token_type_ids"] = \
                res_batch_list[i]['token_type_ids'][j]
                attack_doc_input["position_ids"] = \
                res_batch_list[i]['position_ids'][j]

                res_batch_list[i]['input_ids'] = remove_input_tensor_column(
                    res_batch_list[i]['input_ids'], j)
                res_batch_list[i][
                    'token_type_ids'] = remove_input_tensor_column(
                    res_batch_list[i]['token_type_ids'], j)
                res_batch_list[i]['position_ids'] = remove_input_tensor_column(
                    res_batch_list[i]['position_ids'], j)

    return attack_doc_input, res_batch_list


def attack_by_testing_blackbox(ori_model, surr_model, batch_list, docid_list,
                               attack_doc_id, args, attack_word_number,
                               bert_tokenizer, semantic_helper, word_re,
                               ori_score, ori_qds, qid):

    attack_doc_input, batch_list = find_attack_doc_input_and_remove(batch_list,
                                                                    docid_list,
                                                                    attack_doc_id)

    attack_input_ids_list = attack_doc_input['input_ids'].tolist()
    sep_token_id = bert_tokenizer.sep_token_id

    query_token_id = attack_input_ids_list[1:attack_input_ids_list.index(
                                           sep_token_id)]

    with_last_sep_doc_token_ids_list = attack_input_ids_list[
                                       attack_input_ids_list.index(
                                           sep_token_id) + 1:]
    ori_doc_token_ids_list = with_last_sep_doc_token_ids_list[
                             :len(with_last_sep_doc_token_ids_list) - 1]

    doc_token_ids_list = list({}.fromkeys(ori_doc_token_ids_list).keys())

    word_embedding_matrix = word_re.get_word_embedding(surr_model)
    ori_we_matrix = word_embedding_matrix.clone().detach()

    attacker = Attacker()

    attacker.get_model_gradient(surr_model, batch_list, attack_doc_input,
                                args.device)

    gradient_norm_topk_word, gradient_topk_word_idx_list = word_re.get_highest_gradient_words(
        surr_model, attack_word_number, doc_token_ids_list)

    gradient_topk_words = []
    for word_idx in gradient_topk_word_idx_list:
        gradient_topk_words.append(word_re.idx2word[word_idx])

    attacker.attack(surr_model, batch_list, attack_doc_input,
                    attack_word_idx=doc_token_ids_list,
                    args=args, eps=args.eps, max_iter=args.max_iter)

    attacked_we_matrix = word_re.get_word_embedding(surr_model)

    sim_word_ids_dict, sim_values, sub_word_dict = semantic_helper.pick_most_similar_words_batch(
        gradient_topk_words, ori_doc_token_ids_list, word_re,
        args.simi_candi_topk, args.simi_threshod)

    new_doc_token_id_list, score = \
        word_re.recover_document_greedy_rank_pos(ori_doc_token_ids_list, ori_we_matrix,
                                                 attacked_we_matrix, sim_word_ids_dict, ori_score,
                                                 ori_model, query_token_id, args,
                                                 sub_word_dict, ori_qds, attack_doc_id, qid)

    return new_doc_token_id_list, score


def main():
    args = run_parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    doc_list_length = args.doc_list_length
    assert doc_list_length % args.batch_size == 0
    read_num = int(doc_list_length / args.batch_size)

    # load ori_model
    ori_model_path = args.ori_model_path

    ori_config = BertConfig.from_pretrained(ori_model_path)
    ori_model = RankingBERT_Train.from_pretrained(ori_model_path, config=ori_config)
    ori_model.to(args.device)
    # multi-gpu
    if args.n_gpu > 1:
        ori_model = torch.nn.DataParallel(ori_model)

    # load surrogate_model
    surr_model_path = args.surrogate_model_path
    surr_config = BertConfig.from_pretrained(surr_model_path)
    surr_model = RankingBERT_Train.from_pretrained(surr_model_path, config=surr_config)
    surr_model.to(args.device)
    # multi-gpu
    if args.n_gpu > 1:
        surr_model = torch.nn.DataParallel(surr_model)
    # restore the model's state
    surr_model_state_dict = surr_model.state_dict()
    surr_model_state_dict = copy.deepcopy(surr_model_state_dict)

    # get the embedding layer's name
    for name, param in surr_model.named_parameters():
        args.embed_name = name
        break
    print(args.embed_name)

    logger.info("evaluation parameters %s", args)

    # create global resources
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    synonym_helper, word_recover, attack_qds, ori_qds, collection = read_resources \
        (args.embed_path, args.embed_cos_matrix_path, args.embed_name,
         args.attack_qd_path, args.black_model_ranked_list_score_path, bert_tokenizer, args.bert_vocab_path,
         args.max_query_length, args.max_doc_length, args.collection_memmap_dir)

    max_attack_word_number = args.max_attack_word_number

    attack_save_path = args.save_doc_tokens_path
    attacked_docs_dict = {}
    attacked_docs_score_dict = {}

    # create dataset
    mode = 'dev'
    dev_dataset = MSMARCODataset_file_qd(mode, args.black_model_ranked_list_path,
                                 args.collection_memmap_dir, args.tokenize_dir,
                                 args.bert_tokenizer_path,
                                 args.max_query_length, args.max_doc_length)
    collate_fn = get_collate_function(mode=mode)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                num_workers=args.data_num_workers,
                                collate_fn=collate_fn)
    dataloader_iter = enumerate(dev_dataloader)

    save_attacked_docs_f = open(attack_save_path, 'a')
    previous_done = args.previous_done
    # skip some queries
    for j in range(previous_done * read_num):
        dataloader_iter.__next__()
    qid_list_t = tqdm(list(attack_qds.keys())[previous_done:])

    tested_q_num = 0
    for qid in qid_list_t:
        tested_q_num += 1

        attack_docid_list = list(attack_qds[qid].keys())
        batch_list = []
        docid_list = []
        for i in range(read_num):
            batch_index, data = dataloader_iter.__next__()
            batch, qids, docids = data
            batch_list.append(batch)
            docid_list.append(docids)

        attack_docid_list_t = tqdm(attack_docid_list)
        for attack_docid in attack_docid_list_t:

            surr_model.load_state_dict(surr_model_state_dict)
            ori_score = attack_qds[qid][attack_docid]
            new_doc_token_id_list, score = attack_by_testing_blackbox(ori_model, surr_model,
                                                                      batch_list,
                                                                      docid_list,
                                                                      attack_docid, args,
                                                                      max_attack_word_number,
                                                                      bert_tokenizer,
                                                                      synonym_helper,
                                                                      word_recover,
                                                                      ori_score,
                                                                      ori_qds, qid)
            attack_doc_key = str(qid) + '_' + str(attack_docid)
            attacked_docs_dict[attack_doc_key] = new_doc_token_id_list
            attacked_docs_score_dict[attack_doc_key] = score

        for qid_docid in attacked_docs_dict:
            # attacked_doc_tokens = ' '.join(
            #     str(i) for i in attacked_docs_dict[qid_docid])
            attacked_doc = word_recover.recover_doc(qid_docid.split('_')[1], attacked_docs_dict[qid_docid],
                                                    collection, args.max_doc_length)

            to_write = qid_docid + '\t' + attacked_doc \
                       + '\t' + str(attacked_docs_score_dict[qid_docid])
            save_attacked_docs_f.write(to_write + '\n')
        attacked_docs_dict = {}
        occumpy_mem('1')


if __name__ == "__main__":
    main()