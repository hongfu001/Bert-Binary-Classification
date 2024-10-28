import torch
import random
import numpy as np
from transformers import AutoTokenizer

def collate_fn(batch):
    text, target = zip(*batch)
    text, target = list(text), list(target)

    # src是要送给bert的，所以不需要特殊处理，直接用tokenizer的结果即可
    # padding='max_length' 不够长度的进行填充
    # truncation=True 长度过长的进行裁剪
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    src = tokenizer(text, padding='max_length', max_length=500, return_tensors='pt', truncation=True)

    return src, torch.LongTensor(target)

def ErrorRateAt95Recall1(labels, scores):
    recall_point = 0.95
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]    #降序排列
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True