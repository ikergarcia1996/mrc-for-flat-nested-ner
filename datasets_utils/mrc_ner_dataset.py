# encoding: utf-8


import json
import torch
from transformers import AutoTokenizer, BertTokenizer, XLMRobertaTokenizer
from torch.utils.data import Dataset
from typing import Union, List, Tuple


def token2words(offsets: str) -> List[int]:
    current_word = 0
    t2w: List[int] = []
    prev_offset = -1

    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            t2w.append(None)
            current_word = 0
            prev_offset = -1
            continue
        if offset[0] != prev_offset:
            current_word += 1
        t2w.append(current_word)
        prev_offset = offset[1]

    return t2w


def get_offsets(sentence: str, tokenizer) -> List[Tuple[int, int]]:
    tokens = [
        (tokenizer.convert_ids_to_tokens(tokenizer.encode(w, add_special_tokens=False)))
        for w in sentence.split(" ")
    ]

    offsets = []
    start_id = 0

    for word in tokens:
        if len(word) > 0:
            for token in word:
                token = token.replace("#", "")
                offsets.append((start_id, start_id + len(token)))
                start_id += len(token)

        start_id += 1

    return offsets


def get_token_type_ids(ids: List[int], sep: int) -> List[int]:
    first_sep = ids.index(sep)
    return [0] * (first_sep + 1) + [1] * (len(ids) - first_sep - 1)


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """

    def __init__(
        self,
        json_path,
        tokenizer: Union[BertTokenizer, XLMRobertaTokenizer],
        max_length: int = 128,
        possible_only=False,
        is_chinese=False,
        pad_to_maxlen=False,
    ):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenzier = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [x for x in self.all_data if x["start_position"]]
        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        data = self.all_data[item]
        tokenizer = self.tokenzier

        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]

        if self.is_chinese:
            context = "".join(context.split())
            end_positions = [x + 1 for x in end_positions]
        else:
            # add space offsets
            words = context.split()
            start_positions = [
                x + sum([len(w) for w in words[:x]]) for x in start_positions
            ]
            end_positions = [
                x + sum([len(w) for w in words[: x + 1]]) for x in end_positions
            ]

        encode_plus = tokenizer.encode_plus(
            query.replace("#", ""),
            context.replace("#", ""),
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        tokens = encode_plus["input_ids"]
        type_ids = get_token_type_ids(ids=tokens, sep=tokenizer.sep_token_id)

        # print()
        # print(f"Query: {query}")
        # print(f"Context: {context}")
        # print(f"Token ids: {tokens}")
        # print(f"type_ids: {type_ids}")

        offsets = encode_plus["offset_mapping"]

        # print(f"offsets: {offsets}")

        word_ids = token2words(offsets=offsets)

        # print(f"word_ids: {word_ids}")

        assert len(word_ids) == len(tokens) == len(type_ids) == len(offsets), (
            f"word_ids: {len(word_ids)} "
            f"tokens: {len(tokens)} "
            f"type_ids: {len(type_ids)} "
            f"offsets: {len(offsets)} "
        )

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = [
            origin_offset2token_idx_start[start] for start in start_positions
        ]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # the start/end position must be whole word
        if not self.is_chinese:
            for token_idx in range(len(tokens)):
                current_word_idx = word_ids[token_idx]
                next_word_idx = (
                    word_ids[token_idx + 1] if token_idx + 1 < len(tokens) else None
                )
                prev_word_idx = word_ids[token_idx - 1] if token_idx - 1 > 0 else None
                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert (
            len(new_start_positions) == len(new_end_positions) == len(start_positions)
        )
        assert len(label_mask) == len(tokens)
        start_labels = [
            (1 if idx in new_start_positions else 0) for idx in range(len(tokens))
        ]
        end_labels = [
            (1 if idx in new_end_positions else 0) for idx in range(len(tokens))
        ]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # make sure last token is [SEP]
        sep_token = tokenizer.sep_token_id
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[:-1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        print(
            f"tokens: {tokens}\n"
            f"type_ids: {type_ids}\n"
            f"start_labels: {start_labels}\n"
            f"end_labels: {end_labels}\n"
            f"start_label_mask:{start_label_mask}\n"
            f"end_label_mask:{end_label_mask}\n"
            f"match_labels:{match_labels}\n"
            f"sample_idx:{sample_idx}\n"
            f"label_idx:{label_idx}\n"
            f"==================================================0"
        )

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            sample_idx,
            label_idx,
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def run_dataset():
    """test dataset"""
    import os
    from datasets_utils.collate_functions import collate_to_max_length
    from torch.utils.data import DataLoader

    # zh datasets_utils
    # bert_path = "/mnt/mrc/chinese_L-12_H-768_A-12"
    # json_path = "/mnt/mrc/zh_msra/mrc-ner.test"
    # # json_path = "/mnt/mrc/zh_onto4/mrc-ner.train"
    # is_chinese = True

    # en datasets_utils
    bert_path = "/mnt/mrc/bert-base-uncased"
    json_path = "/mnt/mrc/ace2004/mrc-ner.train"
    # json_path = "/mnt/mrc/genia/mrc-ner.train"
    is_chinese = False

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    dataset = MRCNERDataset(
        json_path=json_path, tokenizer=tokenizer, is_chinese=is_chinese
    )

    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_to_max_length)

    for batch in dataloader:
        for (
            tokens,
            token_type_ids,
            start_labels,
            end_labels,
            start_label_mask,
            end_label_mask,
            match_labels,
            sample_idx,
            label_idx,
        ) in zip(*batch):
            tokens = tokens.tolist()
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            if not start_positions:
                continue
            print("=" * 20)
            print(
                f"len: {len(tokens)}",
                tokenizer.decode(tokens, skip_special_tokens=False),
            )
            for start, end in zip(start_positions, end_positions):
                print(
                    str(sample_idx.item()),
                    str(label_idx.item())
                    + "\t"
                    + tokenizer.decode(tokens[start : end + 1]),
                )


if __name__ == "__main__":
    run_dataset()
