# coding=utf-8

# Reference: https://github.com/huggingface/pytorch-pretrained-BERT

"""Convert BERT checkpoint."""

from __future__ import absolute_import, division, print_function

import argparse
import re

import numpy as np
import tensorflow as tf
import torch
from modeling import BertConfig, BertModel

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument(
    "--tf_checkpoint_path",
    default="uncased_L-12_H-768_A-12/bert_model.ckpt",
    type=str,
    required=False,
    help="Path the TensorFlow checkpoint path.",
)
parser.add_argument(
    "--bert_config_file",
    default="uncased_L-12_H-768_A-12/bert_config.json",
    type=str,
    required=False,
    help="The config json file corresponding to the pre-trained BERT model. \n"
    "This specifies the model architecture.",
)
parser.add_argument(
    "--pytorch_dump_path",
    default="uncased_L-12_H-768_A-12/pytorch_model.bin",
    type=str,
    required=False,
    help="Path to the output PyTorch model.",
)

args = parser.parse_args()


def convert():
    # Initialise PyTorch model
    config = BertConfig.from_json_file(args.bert_config_file)
    model = BertModel(config)

    # Load weights from TF model
    path = args.tf_checkpoint_path
    print("Converting TensorFlow checkpoint from {}".format(path))

    init_vars = tf.train.list_variables(path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading {} with shape {}".format(name, shape))
        array = tf.train.load_variable(path, name)
        print("Numpy array shape {}".format(array.shape))
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "bert/"
        print("Loading {}".format(name))
        name = name.split("/")
        if any(n in ["adam_v", "adam_m", "l_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        if name[0] in ["redictions", "eq_relationship"]:
            print("Skipping")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    torch.save(model.state_dict(), args.pytorch_dump_path)


if __name__ == "__main__":
    convert()
