import argparse


def read_score_file(scorefilepath):
    q_d_s = {}
    with open(scorefilepath, 'r') as sf:
        for line in sf:
            ss = line.strip().split('\t')
            qid = int(ss[0])
            docid = int(ss[1])
            score = float(ss[2])

            if qid not in q_d_s:
                q_d_s[qid] = {}

            q_d_s[qid][docid] = score

    return q_d_s


def read_attacked_doc_file(file_path):
    qds_dict, qd_dict = {}, {}
    with open(file_path, 'r') as f:
        for line in f:
            qid_docid, content, score = line.strip().split('\t')
            qid, docid = qid_docid.split('_')
            if int(qid) not in qds_dict:
                qds_dict[int(qid)] = {}
            qds_dict[int(qid)][int(docid)] = float(score)
            qd_dict[qid_docid] = content

    return qds_dict, qd_dict


def get_rank_pos(d_s_dict, wanted_docid):
    ranked_docs = sorted(d_s_dict, key=d_s_dict.get, reverse=True)
    index = 1
    rank_pos = -1
    for docid in ranked_docs:
        if docid == wanted_docid:
            rank_pos = index
            break
        index += 1
    assert rank_pos != -1
    return rank_pos


def get_use_single_doc(ori_doc_text, attacked_doc_text, use):
    text1 = []
    text1.append(ori_doc_text)
    text2 = []
    text2.append(attacked_doc_text)
    semantic_sim = use.semantic_sim(text1, text2)
    # print(semantic_sim)
    return semantic_sim


def evaluate_results(args):

    adversarial_doc_score_path = args.adversarial_doc_tokens_path
    qrel_path = args.qrel_path
    ori_score_path = args.black_model_ranked_list_path
    collection_memmap_dir = args.collection_memmap_dir
    bert_tokenizer_dir = args.bert_tokenizer_dir

    evaluate_query_number = 200

    qrels = {}
    with open(qrel_path
            , encoding='utf-8') as qrel:
        for line in qrel:
            l = line.split(" ")
            qid = int(l[0])
            docNo = int(l[2][1:])
            label = int(l[3])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docNo] = label

    ori_score_dict = read_score_file(ori_score_path)
    adv_doc_score_dict, qd_dict = read_attacked_doc_file(adversarial_doc_score_path)

    ori_pos_dict, att_pos_dict = {}, {}
    import copy
    index = 0
    for qid in adv_doc_score_dict:
        index += 1
        if index > evaluate_query_number:
            break

        ori_pos_dict[qid], att_pos_dict[qid] = [], []

        ori_d_s_dict = ori_score_dict[qid]
        for adv_docid in adv_doc_score_dict[qid]:
            changed_d_s_dict = copy.deepcopy(ori_d_s_dict)

            ori_pos = get_rank_pos(ori_d_s_dict, adv_docid)
            ori_pos_dict[qid].append(ori_pos)

            changed_d_s_dict[adv_docid] = adv_doc_score_dict[qid][adv_docid]
            adv_pos = get_rank_pos(changed_d_s_dict, adv_docid)
            att_pos_dict[qid].append(adv_pos)

    print(ori_pos_dict)
    print(att_pos_dict)

    from myutils.evaluate_metrics import success_rate, perturb_percent
    success_rate(ori_pos_dict, att_pos_dict, print_every_query=False)
    print(adversarial_doc_score_path)

    print('now computing the PP...')
    from marcodoc.dataset import CollectionDataset
    from transformers import BertTokenizer
    collection = CollectionDataset(collection_memmap_dir)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_dir)
    perturb_percent_list = []
    for qid_docid in qd_dict:
        doc_id = int(qid_docid.split('_')[1])
        doc_token_ids = collection[doc_id]
        ori_doc = bert_tokenizer.decode(doc_token_ids)
        adv_doc = qd_dict[qid_docid]
        # print(doc_id)
        # print(ori_doc)
        # print(adv_doc)
        perturb_percent_list.append(perturb_percent(ori_doc, adv_doc))
    from numpy import mean
    print('PP:', mean(perturb_percent_list))

    print('now computing the SS_{doc}...')
    # please use cpu
    import os
    from myutils.Use import USE
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    use = USE(args.use_cache_path)
    doc_sim_score_list = []
    for qid_docid in qd_dict:
        doc_id = int(qid_docid.split('_')[1])
        doc_token_ids = collection[doc_id]
        ori_doc = bert_tokenizer.decode(doc_token_ids)
        adv_doc = qd_dict[qid_docid]
        # print(doc_id)
        # print(ori_doc)
        # print(adv_doc)
        doc_use_score = get_use_single_doc(ori_doc, adv_doc, use)
        doc_sim_score_list.append(doc_use_score)
    print('SS_{doc}:', mean(doc_sim_score_list))

    print('now computing the SS_{sent}...')
    from nltk import sent_tokenize

    sent_sim_score_list = []
    for qid_docid in qd_dict:
        doc_id = int(qid_docid.split('_')[1])
        doc_token_ids = collection[doc_id]
        ori_doc = bert_tokenizer.decode(doc_token_ids)
        adv_doc = qd_dict[qid_docid]

        ori_doc_sentence_list = sent_tokenize(ori_doc)
        att_doc_sentence_list = sent_tokenize(adv_doc)

        try:
            assert len(ori_doc_sentence_list) == len(att_doc_sentence_list)
        except AssertionError:
            # print(ori_doc_sentence_list)
            # print(att_doc_sentence_list)
            ori_sentence = ori_doc.strip()
            att_sentence = adv_doc.strip()
            use_score = get_use_single_doc(ori_sentence, att_sentence, use)
            sent_sim_score_list.append(use_score[0][0])
            continue

        single_doc_use_score_list = []
        for i in range(len(ori_doc_sentence_list)):
            ori_sentence = ori_doc_sentence_list[i].strip()
            att_sentence = att_doc_sentence_list[i].strip()
            sentence_use_score = get_use_single_doc(ori_sentence, att_sentence, use)
            single_doc_use_score_list.append(sentence_use_score)

        use_score = mean(single_doc_use_score_list)
        # print(use_score)

        sent_sim_score_list.append(use_score)

    print('SS_{sent}:', mean(sent_sim_score_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adversarial_doc_tokens_path", type=str, default="")
    parser.add_argument("--black_model_ranked_list_path", type=str, default=" ")
    parser.add_argument("--qrel_path", type=str, default="msmarco/document_ranking/dev/msmarco-docdev-qrels.tsv")
    parser.add_argument("--collection_memmap_dir", type=str, default='msmarco/document_ranking/tokenized_collection_memmap')
    parser.add_argument("--bert_tokenizer_dir", type=str, default='bert_pretrained_model/bert-base-uncased')
    parser.add_argument("--use_cache_path", type=str, default='USE_DAN')
    args = parser.parse_args()
    print(args)
    evaluate_results(args)