import torch
import torch.nn.functional as F
from train_utils import log
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def consistency_loss(logits, logits_t, log_file, lbd, y, epoch, dis, eta=0.5):
    """
    Consistency regularization for certified robustness.

    Parameters
    ----------
    logits : List[torch.Tensor]
        A list of logit batches of the same shape, where each
        is sampled from f(x + noise) with i.i.d. noises.
        len(logits) determines the number of noises, i.e., m > 1.
    lbd : float
        Hyperparameter that controls the strength of the regularization.
    eta : float (default: 0.5)
        Hyperparameter that controls the strength of the entropy term.
        Currently used only when loss='default'.
    loss : {'default', 'xent', 'kl', 'mse'} (optional)
        Which loss to minimize to obtain consistency.
        - 'default': The default form of loss.
            All the values in the paper are reproducible with this option.
            The form is equivalent to 'xent' when eta = lbd, but allows
            a larger lbd (e.g., lbd = 20) when eta is smaller (e.g., eta < 1).
        - 'xent': The cross-entropy loss.
            A special case of loss='default' when eta = lbd. One should use
            a lower lbd (e.g., lbd = 3) for better results.
        - 'kl': The KL-divergence between each predictions and their average.
        - 'mse': The mean-squared error between the first two predictions.

    """

    m = len(logits)
    softmax = [F.softmax(logit, dim=1) for logit in logits]


    avg_logits = sum(logits) / m
    tmp1 = torch.argsort(avg_logits, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y[:avg_logits.size(0)], tmp1[:, -2], tmp1[:, -1])

    logit_margin_y = torch.gather(avg_logits, 1, (y[:avg_logits.size(0)].unsqueeze(1)).long()).squeeze() 
    logit_margin_k = torch.gather(avg_logits, 1, (new_y.unsqueeze(1)).long()).squeeze()
    logit_margin = logit_margin_k - logit_margin_y
    logit_margin_sort = torch.sort(logit_margin)[0]



    self_index = torch.nonzero(logit_margin<1, as_tuple=False).squeeze()
    logit_0_self = logits[0][self_index]
    logit_1_self = logits[1][self_index]

    



    teacher_index = torch.nonzero(logit_margin>=1, as_tuple=False).squeeze()
    if teacher_index.numel() == 0:
        loss_distillation = torch.tensor([0., 0.]).to(device)
    else:
        logit_0_student = logits[0][teacher_index]
        logit_1_student = logits[1][teacher_index]
        logit_0_teacher = logits_t[0][teacher_index]
        logit_1_teacher = logits_t[1][teacher_index]

        distillation_loss_0 = torch.norm(logit_0_teacher - logit_0_student, dim=-1)

        distillation_loss_1 = torch.norm(logit_1_teacher - logit_1_student, dim=-1)
        loss_distillation = (distillation_loss_0 + distillation_loss_1) / m



    avg_softmax = sum(softmax) / m
    avg_softmax_self = avg_softmax[self_index]

    if epoch >= -1:#inspired from MART MAIL MMA
        loss_kl_0 = kl_div(logit_0_self, avg_softmax_self)
        loss_kl_1 = kl_div(logit_1_self, avg_softmax_self)
        loss_kl_tmp = (loss_kl_0 + loss_kl_1) / m



        with torch.no_grad():
            abs_margin = torch.abs(logit_margin[self_index])
            ada_w = torch.sigmoid(0.1*(abs_margin))
            ada_w_normalize = ada_w.size(0) * (ada_w / ada_w.sum())
        loss_kl = loss_kl_tmp * ada_w_normalize
        loss_ent = entropy(avg_softmax)
        consistency = lbd * loss_kl.mean() + dis * loss_distillation.mean() + eta * loss_ent.mean()
    else:
        loss_kl = [kl_div(logit, avg_softmax) for logit in logits]
        loss_kl = sum(loss_kl) / m
        loss_ent = entropy(avg_softmax)
        if loss_distillation.mean().item()==0:
            consistency = lbd * loss_kl.mean() + eta * loss_ent.mean()
        else:
            consistency = lbd * loss_kl.mean() + dis * loss_distillation.mean() + eta * loss_ent.mean()

    
    
    log(log_file, "{}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, logit_margin.max().item(), logit_margin.min().item(),
            (logit_margin >= 0).sum().item(), (logit_margin >= 1).sum().item(),
            (logit_margin >= 2).sum().item(), logit_margin_sort[int(0.9*torch.sort(logit_margin)[0].size(0))],
            logit_margin_sort[int(0.8*torch.sort(logit_margin)[0].size(0))], logit_margin_sort[int(0.7*torch.sort(logit_margin)[0].size(0))],
            self_index.size(0), loss_distillation.mean().item(), loss_distillation.min().item(), loss_distillation.max().item()))


    return consistency


def kl_div(input, targets):
    return F.kl_div(F.log_softmax(input, dim=1), targets, reduction='none').sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent
