import torch


def calc_acc(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    return torch.sum(targets == preds)
