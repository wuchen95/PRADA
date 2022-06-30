import numpy as np


class SemanticHelper:

    def __init__(self, sim_embedding_path, sim_embedding_npy_path):
        self.sim_embedding_path = sim_embedding_path
        self.sim_embedding_npy_path = sim_embedding_npy_path

    def build_vocab(self):
        idx2word = {}
        word2idx = {}

        print("Building vocab...")
        with open(self.sim_embedding_path, 'r') as sef:
            for line in sef:
                word = line.split()[0]
                if word not in word2idx:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1

        self.idx2word = idx2word
        self.word2idx = word2idx

    def load_embedding_cos_sim_matrix(self):
        print('Load pre-computed cosine similarity matrix from {}'.format(self.sim_embedding_npy_path))
        cos_sim = np.load(self.sim_embedding_npy_path)
        print("Cos sim import finished!")
        self.cos_sim_matrix = cos_sim

    def is_number(self, input):
        try:
            t = float(input)
            return True
        except:
            return False

    def recover_whole_word(self, sub_word, ori_doc_token_ids_list, word_re):
        word_index = ori_doc_token_ids_list.index(word_re.word2idx[sub_word])
        right_word_index = word_index + 1
        while (right_word_index < len(ori_doc_token_ids_list)) and\
                (word_re.idx2word[ori_doc_token_ids_list[right_word_index]].
                        startswith("##")):
            word_index = word_index + 1
            right_word_index = word_index + 1
        subword_list = []
        subword_token_id_list = []
        now_sub_word = sub_word
        now_word_index = word_index
        subword_list.append(now_sub_word)
        subword_token_id_list.append(word_re.word2idx[now_sub_word])
        while (now_sub_word.startswith("##")) and (now_word_index > 0):
            now_word_index = now_word_index - 1
            now_sub_word = word_re.idx2word[
                ori_doc_token_ids_list[now_word_index]]

            subword_list.append(now_sub_word)
            subword_token_id_list.append(word_re.word2idx[now_sub_word])

        whole_word = subword_list[-1]

        assert len(subword_list) > 1

        for i in range(len(subword_list) - 2, -1, -1):
            current_sub_word = subword_list[i][2:]
            whole_word += current_sub_word

        first_word = subword_list[-1]
        tail_word = whole_word[len(first_word):]
        if self.is_number(first_word) and (tail_word in self.word2idx):
            whole_word = tail_word
            subword_token_id_list.pop(0)
        self.subword_neighbor_dict[whole_word] = subword_token_id_list
        return whole_word

    def pick_most_similar_words_batch(self, src_words, ori_doc_token_ids_list,
                                      word_re, ret_count=10, threshold=0.):
        in_words_idx = []
        in_words = []
        out_words = []

        self.subword_neighbor_dict = {}
        for src_word in src_words:
            if src_word.startswith('##'):
                src_word = self.recover_whole_word(src_word, ori_doc_token_ids_list, word_re)

            if src_word in self.word2idx:
                in_words_idx.append(self.word2idx[src_word])
                in_words.append(src_word)
            else:
                out_words.append(src_word)
        sim_order = np.argsort(-self.cos_sim_matrix[in_words_idx, :])[:, 1:1 + ret_count]
        sim_words, sim_values = {}, []
        for idx, in_word_idx in enumerate(in_words_idx):
            sim_value = self.cos_sim_matrix[in_word_idx][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [self.idx2word[id] for id in sim_word]
            sim_words[in_words[idx]] = sim_word
            sim_values.append(sim_value)
        return sim_words, sim_values, self.subword_neighbor_dict

    def pick_nearest_word_with_embedding(self, importance_tokens, importance_token_id_list,
                                         ori_doc_token_ids_list,
                                         ori_embedding, att_embedding, word_re, attacker_word_number):
        new_doc_token_ids_list = ori_doc_token_ids_list.copy()
        assert len(importance_tokens) == len(importance_token_id_list)
        m = attacker_word_number
        ti_huan_num = 0
        for i in range(len(importance_tokens)):
            word = importance_tokens[i]
            word_id = importance_token_id_list[i]
            for j in range(len(ori_doc_token_ids_list)):
                ori_word_id = ori_doc_token_ids_list[j]
                if ori_word_id == word_id:
                    word_vecor = att_embedding[word_id]
                    max_sim, max_id = word_re.get_max_sim_word(word_vecor, ori_embedding)
                    replace_word_id = max_id.cpu()[0].item()
                    new_doc_token_ids_list[j] = replace_word_id
                    ti_huan_num += 1
                    if ti_huan_num == m:
                        break
            if ti_huan_num == m:
                break
        return new_doc_token_ids_list

    def pick_most_similar_words_batch_with_itself(self, src_words, ret_count=10, threshold=0.):

        in_words_idx = []
        in_words = []
        out_words = []
        for src_word in src_words:
            if src_word in self.word2idx:
                in_words_idx.append(self.word2idx[src_word])
                in_words.append(src_word)
            else:
                out_words.append(src_word)

        sim_order = np.argsort(-self.cos_sim_matrix[in_words_idx, :])[:, 1:1 + ret_count]
        sim_words, sim_values = {}, []
        for idx, in_word_idx in enumerate(in_words_idx):
            sim_value = self.cos_sim_matrix[in_word_idx][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [self.idx2word[id] for id in sim_word]
            sim_word.append(in_words[idx])
            sim_words[in_words[idx]] = sim_word
            sim_values.append(sim_value)
        return sim_words, sim_values

