# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.nn as nn

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG
import model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers


class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length=20, mode='x'):
        super().__init__()
        self.max_seq_length = args.max_seq_length
        set_visual_config(args)
        self.action_embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, int(args.hidden_size/2)),
            nn.Tanh()
        )
        self.action_drop = nn.Dropout(p=args.dropout)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased",
            mode=mode
        )
        # self.model = VisualBertForLXRFeature(args, mode=mode)
        self.candidate_att_layer = model.SoftDotAttention(args.hidden_size+args.hidden_size, args.feature_size+args.angle_feat_size)
        # if args.from_scratch:
        #     print("initializing all the weights")
        #     self.model.apply(self.model.init_bert_weights)
        #     # TODO init candidate_att_layer
        # model_path = "./pytorch_pretrained_bert/pytorch_model.bin"
        # state_dict = torch.load(model_path)
        # load_state_dict = {}
        # for k,v in self.model.named_parameters():
        #     if k in state_dict and k.startswith("bert.embeddings"):
        #         print(k)
        #         load_state_dict[k]=state_dict[k]
        # self.model.load_state_dict(load_state_dict, strict=False) # shape mismatch

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)
        self.candidate_att_layer = nn.DataParallel(self.candidate_att_layer)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, candidate_feat, visual_attention_mask=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        h = self.model(input_ids, segment_ids, input_mask,
                            visual_feats=feats,
                            visual_attention_mask=visual_attention_mask)

        # h_drop = self.drop(h) # TODO
        _, logit = self.candidate_att_layer(h, candidate_feat, output_prob=False)
        return logit, h

    def forward_enc(self, sents):
        train_features = convert_sents_to_features(
                sents, self.max_seq_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        return input_ids, input_mask, segment_ids

    def forward_dec(self, input_ids, input_mask, segment_ids, feats, candidate_feat, action, visual_attention_mask=None):
        feat, pos = feats
        h = self.model(input_ids, segment_ids, input_mask,
                visual_feats=feats,
                visual_attention_mask=visual_attention_mask)
        action_embeds = self.action_embedding(action)
        action_embeds = self.action_drop(action_embeds)
        cat_h = torch.cat([h, action_embeds], dim=1)
        # h_drop = self.drop(h) # TODO
        _, logit = self.candidate_att_layer(cat_h, candidate_feat, output_prob=False)
        return logit, h

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)




