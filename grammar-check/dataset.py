import csv
import itertools
import os
import random

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as flow
import torch.utils.data
from transformers import AutoTokenizer

MAX_LENGTH = 250


class Dataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_dir: str,
        model_name: str,
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        files = os.listdir(data_dir)
        self.files = []
        for i in range(rank, len(files), world_size):
            file_path = os.path.join(data_dir, files[i])
            self.files.append(file_path)
        self.aug_methods = [
            nac.OcrAug(aug_char_max=2, aug_word_max=2),
            nac.KeyboardAug(aug_char_max=2, aug_word_max=2),
            nac.RandomCharAug(action="insert", aug_char_max=2, aug_word_max=2),
            nac.RandomCharAug(action="substitute", aug_char_max=2, aug_word_max=2),
            nac.RandomCharAug(action="swap", aug_char_max=2, aug_word_max=2),
            nac.RandomCharAug(action="delete", aug_char_max=2, aug_word_max=2),
            naw.SpellingAug(aug_max=3),
        ]

    def __iter__(self):
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id
        readers = []
        for file_path in self.files:
            file_handler = open(file_path, encoding="utf-8")
            reader = csv.reader(
                (x.replace("\0", "") for x in file_handler), delimiter="\t"
            )
            mapped_iter = map(self.row_process, reader)
            iter_reader = itertools.islice(
                mapped_iter, worker_id, None, worker_total_num
            )
            readers.append(iter_reader)
        group_iter = itertools.chain.from_iterable(readers)
        return group_iter

    def row_process(self, row):
        if len(row) == 1:
            row.append(row[0])
        aug_method_idx = random.randrange(0, len(self.aug_methods))
        aug_input = self.aug_methods[aug_method_idx].augment(row[0])
        input_tokens = self.tokenizer(
            aug_input[0],
            max_length=MAX_LENGTH,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        output_tokens = self.tokenizer(
            row[1],
            max_length=MAX_LENGTH,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return (
            input_tokens["input_ids"][0],
            input_tokens["attention_mask"][0],
            output_tokens["input_ids"][0],
        )
