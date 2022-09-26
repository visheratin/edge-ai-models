import csv
import io
import tarfile
import uuid

import torch
from transformers import AutoTokenizer

MAX_LENGTH = 250


def process_file(file_path: str, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tar_file = tarfile.open(f"{file_path}.tar", "w")
    with open(file_path) as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            input_tokens = tokenizer(
                row[0],
                max_length=MAX_LENGTH,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            output_tokens = tokenizer(
                row[1],
                max_length=MAX_LENGTH,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            diff = torch.sum(
                input_tokens["input_ids"][0] - output_tokens["input_ids"][0]
            )
            if diff == 0:
                continue
            res = torch.cat(
                (
                    input_tokens["input_ids"][0].unsqueeze(0),
                    input_tokens["attention_mask"][0].unsqueeze(0),
                    output_tokens["input_ids"][0].unsqueeze(0),
                ),
                dim=0,
            )
            buff = io.BytesIO()
            file_name = f"{str(uuid.uuid4())}.pt"
            torch.save(res, buff)
            buff.seek(0)
            info = tarfile.TarInfo(name=file_name)
            info.size = buff.getbuffer().nbytes
            tar_file.addfile(info, buff)
    tar_file.close()


process_file("./result.tsv-00000-of-00010", "google/t5-efficient-tiny")
