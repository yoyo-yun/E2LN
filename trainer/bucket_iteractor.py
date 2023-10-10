import math
import torch
import random

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('pretrained_models/bert-base-uncased')

class BucketIteratorForBert(object):
    def __init__(self, data, batch_size, tokenizer=bert_tokenizer, sort_index=0, shuffle=True, sort=True, device='cpu'):
        self.shuffle = shuffle
        self.sort = sort
        self.tokenizer = tokenizer
        self.sort_key = sort_index
        self.device = device
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_usr_indices = []
        batch_prd_indices = []
        batch_labels = []
        batch_lengths = []

        max_len_sentence = max([len(t[0]) for t in batch_data])

        for item in batch_data:
            tokens_index, label, user_index, product_index = item
            batch_lengths.append(len(tokens_index))
            tokens_padding = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.tokenizer.pad_token)) * (max_len_sentence - len(tokens_index))
            tokens_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.tokenizer.cls_token)) + \
                           tokens_index + \
                           self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.tokenizer.sep_token)) + \
                           tokens_padding

            batch_text_indices.append(tokens_index)
            batch_labels.append(label)
            batch_usr_indices.append(user_index)
            batch_prd_indices.append(product_index)
        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)
        batch_labels = torch.tensor(batch_labels, device=self.device)

        return {'batch_text_indices': batch_text_indices,
                'batch_usr_indices': batch_usr_indices,
                'batch_labels': batch_labels,
                'batch_prd_indices': batch_prd_indices,
                'batch_lengths': batch_lengths}

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.batch_len