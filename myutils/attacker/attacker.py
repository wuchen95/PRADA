import torch
import torch.nn as nn

from myutils.attacker.PGD import PGD


class Attacker:
    def __init__(self):
        pass

    def rank_attack_loss(self, pos_doc_s: torch.Tensor, neg_docs_s: torch.Tensor):
        '''

        :param pos_doc_s: shape = [1, 1]
        :param neg_docs_s: shape = [B, 1]
        :return:
        '''

        margin = 1.0

        reduction = 'sum'
        loss_fct = nn.MarginRankingLoss(margin=margin, reduction=reduction)
        num_neg = neg_docs_s.shape[0]
        pos_doc_ss = pos_doc_s.repeat(num_neg, 1)
        labels = torch.ones_like(pos_doc_ss)
        computed_loss = loss_fct(pos_doc_ss, neg_docs_s, labels)
        return computed_loss

    def get_model_gradient(self, model, batch_list, attack_doc_input, device):

        model.train()
        model.zero_grad()

        for batch in batch_list:

            batch = {k: torch.cat((attack_doc_input[k].unsqueeze(dim=0), v), dim=0)
                     for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            # scores, shape = [B, 1]
            attack_loss = self.rank_attack_loss(outputs[0], outputs[1:])

            if attack_loss is not None:
                attack_loss.backward()

    def attack(self, model, batch_list, attack_doc_input, attack_word_idx, args,
               eps=0.009, max_iter=3):

        model.train()

        for batch in batch_list:
            pgd_attacker = PGD(model, attack_word_idx, args.device)

            batch = {k: torch.cat((attack_doc_input[k].unsqueeze(dim=0), v), dim=0)
                     for k, v in batch.items()}
            batch = {k: v.to(args.device) for k, v in batch.items()}

            alpha = eps / max(1, max_iter//2)
            for t in range(max_iter):

                outputs = model(**batch)
                attack_loss = self.rank_attack_loss(outputs[0], outputs[1:])

                if attack_loss is not None:
                    attack_loss.backward()

                pgd_attacker.attack(is_first_attack=(t == 0), epsilon=eps,
                                    alpha=alpha, emb_name=args.embed_name)

                model.zero_grad()

    def random_attack(self, model, args):

        emb_name = args.embed_name
        # change the emb_name to the embedding parameter name of your model
        for name, param in model.named_parameters():
            if emb_name in name:
                param_avg = torch.mean(param)
                r_raodong = ((torch.rand(param.shape)-0.5) * 2).to(args.device) * param_avg
                # print(r_raodong)
                param.data.add_(r_raodong)

    def attack_with_momentum(self, model, batch_list, attack_doc_input, attack_word_idx, args,
               eps=0.009, max_iter=3):

        model.train()
        pgd_attacker = PGD(model, attack_word_idx, args.device)
        for batch in batch_list:

            batch = {k: torch.cat((attack_doc_input[k].unsqueeze(dim=0), v), dim=0)
                     for k, v in batch.items()}
            batch = {k: v.to(args.device) for k, v in batch.items()}

            alpha = eps / max(1, max_iter//2)
            for t in range(max_iter):

                outputs = model(**batch)
                attack_loss = self.rank_attack_loss(outputs[0], outputs[1:])

                if attack_loss is not None:
                    attack_loss.backward()

                pgd_attacker.attack_with_momentum(is_first_attack=(t == 0),
                                                  epsilon=eps, alpha=alpha,
                                                  emb_name=args.embed_name)

                model.zero_grad()


if __name__ == '__main__':

    print('begin')



