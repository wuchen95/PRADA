import torch


class WordRecover:
    def __init__(self, embed_name):
        self.embed_name = embed_name
        pass

    def get_word_embedding(self, model):
        emb_name = self.embed_name
        for name, param in model.named_parameters():
            if emb_name in name:
               embedding_matrix = param
               return embedding_matrix

    def get_max_sim_word(self, word_tensors, ori_word_embedding_matrix):
        cossim_maxtrix = self.cos_similar(word_tensors, ori_word_embedding_matrix)
        max_sim, max_idx = torch.max(cossim_maxtrix, dim=-1)

        return max_sim, max_idx

    def cos_similar(self, p:torch.Tensor, q:torch.Tensor):
        sim_maxtrix = p.matmul(q.transpose(-2, -1))
        a = torch.norm(p, p=2, dim=-1)
        b = torch.norm(q, p=2, dim=-1)
        sim_maxtrix /= a.unsqueeze(-1)
        sim_maxtrix /= b.unsqueeze(-2)
        return sim_maxtrix

    def get_sim_word_matrix(self, simwords_list, ori_word_embedding_matrix):
        candidate_word_matrix_list = {}
        candidate_word_no_dict = {}
        for ori_word in simwords_list:
            idx_list = []
            simwords = simwords_list[ori_word]
            for simword in simwords:
                simword_idx = self.tokenizer.encode(simword)
                if len(simword_idx) > 1:
                    continue
                idx_list.append(simword_idx[0])
            simword_matrix = ori_word_embedding_matrix[idx_list, :]
            candidate_word_matrix_list[ori_word] = simword_matrix
            candidate_word_no_dict[ori_word] = idx_list
        return candidate_word_matrix_list, candidate_word_no_dict

    def get_sim_word_semantic(self, candidate_word_matrix_list, now_word_embedding_matrix):
        max_sim_list = {}
        max_id_list = {}

        for ori_word in candidate_word_matrix_list:
            ori_word_idx = self.tokenizer.encode(ori_word)
            ori_word_tensor = now_word_embedding_matrix[ori_word_idx, :]
            if len(candidate_word_matrix_list[ori_word]) == 0:
                continue
            max_sim, max_idx = self.get_max_sim_word(ori_word_tensor, candidate_word_matrix_list[ori_word])
            max_sim_list[ori_word] = max_sim
            max_id_list[ori_word] = max_idx
        return max_sim_list, max_id_list

    def recover_document_semantic(self, old_doc, ori_wem, now_wem, simwords_list):

        max_word_no_dict = {}
        candidate_word_matrix_dict, candidate_word_no_dict = self.get_sim_word_matrix(simwords_list, ori_wem)
        max_sim_dict, max_id_dict = self.get_sim_word_semantic(candidate_word_matrix_dict, now_wem)
        for word in max_id_dict:
            max_word_no_dict[word] = [candidate_word_no_dict[word][max_id_dict[word][0]]]

        new_doc_list = []
        for token in old_doc.split(' '):
            if token not in max_word_no_dict:
                new_doc_list.append(token)
            else:
                # print(max_word_no_dict[token])
                new_token = self.tokenizer.decode(max_word_no_dict[token])
                # print(new_token)
                new_doc_list.append(new_token)
        new_doc = ' '.join(new_doc_list)
        print(new_doc)
        return new_doc

    def get_highest_gradient_words(self, model, attack_num=50, attack_word_idx = []):
        embed_name = self.embed_name
        word_embedding = self.get_word_embedding(model, embed_name)
        gradient_matrix = word_embedding.grad[attack_word_idx]

        row_norm = torch.norm(gradient_matrix, p=2, dim=-1)

        attack_num = min(attack_num, row_norm.shape[0])

        topk_norm, topk_idx = torch.topk(row_norm, k=attack_num, dim=-1)
        print(topk_norm, topk_idx)
        model.zero_grad()

        word_topk_list = []
        for idx in topk_idx:
            idx = idx.item()
            true_idx = attack_word_idx[idx]
            word_topk_list.append(true_idx)

        return topk_norm, word_topk_list



