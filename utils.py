from transformers import AutoTokenizer


def get_tokenizer(pretrain, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


# 统计 dataset 中正样本和负样本的数量
def compute_positive_negative_samples(dataset, label_key="value", positive_token="[POS]", negative_token="[NEG]"):
    positive_cnt = 0
    negative_cnt = 0

    for data in dataset:
        label = data[label_key]
        if label == positive_token:
            positive_cnt += 1
        elif label == negative_token:
            negative_cnt += 1

    return positive_cnt, negative_cnt


# 给定 dataset, 基于 value 和 soft_label 修正 soft_label 的结果
def refine_soft_labels(dataset, label_key="value", softlabel_key="softlabel", positive_token="[POS]", negative_token="[NEG]", threshold=0.1):
    refined_softlabels = []
    # 如果 label == positive_token, 并且 soft_label 和 1.0 的差值小于 threshold, 则将 soft_label 设为 1.0
    # 如果 label == negative_token, 并且 soft_label 和 0.0 的差值小于 threshold, 则将 soft_label 设为 0.0
    # 否则保持 soft_label 不变
    for data in dataset:
        labels = data[label_key]
        softlabels = data[softlabel_key]
        refined_softlabel = []
        for label, softlabel in zip(labels, softlabels):
            if label == positive_token and abs(softlabel - 1.0) < threshold:
                refined_softlabel.append(1.0)
            elif label == negative_token and abs(softlabel - 0.0) < threshold:
                refined_softlabel.append(0.0)
            else:
                refined_softlabel.append(softlabel)
        refined_softlabels.append(refined_softlabel)

    # TypeError: 'Dataset' object does not support item assignment
    dataset = dataset.remove_columns([softlabel_key])
    dataset = dataset.add_column(softlabel_key, refined_softlabels)
    # print(refined_softlabels)
    # print(dataset[0][softlabel_key])
    return dataset