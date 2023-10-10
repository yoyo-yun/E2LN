import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from models.attention_zoo import LN_based_attention, LN_in_Transformer

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers import BertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
]


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LN_in_Transformer(config.hidden_size, eps=config.layer_norm_eps, up=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        # if token_type_ids is None:
        #     if hasattr(self, "token_type_ids"):
        #         buffered_token_type_ids = self.token_type_ids[:, :seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings, _ = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def extend_word_embedding(self):
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings * 10, self.config.hidden_size).from_pretrained(
            torch.cat([self.position_embeddings.weight.data for _ in range(10)], dim=0),
            freeze=False
        )
        self.position_ids = torch.arange(self.config.max_position_embeddings * 10).expand((1, -1))


class UserItemEmbeddings(nn.Module):
    def __init__(self, cus_config):
        super().__init__()
        self.cus_config = cus_config
        self.user_embeddings = nn.Embedding(cus_config.usr_size, cus_config.usr_dim)
        self.user_embeddings.weight.requires_grad = True

        self.item_embeddings = nn.Embedding(cus_config.prd_size, cus_config.prd_dim)
        self.item_embeddings.weight.requires_grad = True
        self.reset_parameters()

    def forward(self, user_ids, item_ids):
        if len(user_ids.shape) == 1:
            user_ids = user_ids.unsqueeze(1)
            item_ids = item_ids.unsqueeze(1)
        return self.user_embeddings(user_ids), self.item_embeddings(item_ids)

    def reset_parameters(self):
        self.user_embeddings.weight.data.copy_(torch.zeros(self.user_embeddings.weight.size(0), self.user_embeddings.weight.size(1)))
        self.item_embeddings.weight.data.copy_(torch.zeros(self.item_embeddings.weight.size(0), self.item_embeddings.weight.size(1)))


class BertSelfAttention(nn.Module):
    def __init__(self, config, cus_config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.layer_id = layer_id
        attention_window = cus_config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2
        self.attention_window = attention_window

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        self_memory=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # query_vectors = self.query(hidden_states)
        # key_vectors = self.key(hidden_states)
        # value_vectors = self.value(hidden_states)
        #
        # batch_size, seq_len, embed_dim = hidden_states.size()
        # assert (
        #         embed_dim == self.embed_dim
        # ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"
        #
        # # normalize query
        # query_vectors /= math.sqrt(self.head_dim)
        #
        # query_vectors = query_vectors.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # key_vectors = key_vectors.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # value_vectors = value_vectors.view(batch_size, seq_len, self.num_heads, self.head_dim)
        #
        # if self_memory is not None:
        #     key_memory_vectors = self.key(self_memory)
        #     value_memory_vectors = self.value(self_memory)
        #     key_memory_vectors = key_memory_vectors.view(batch_size, -1, self.num_heads, self.head_dim)
        #     value_memory_vectors = value_memory_vectors.view(batch_size, -1, self.num_heads, self.head_dim)
        #     outputs, attn_probs = self._sliding_chunks_multi_head_attention_w_memory(
        #         query_vectors,
        #         key_vectors,
        #         value_vectors,
        #         attention_mask,
        #         self.attention_window,
        #         key_memory_vectors,
        #         value_memory_vectors,
        #     )
        # else:
        #     outputs, attention_scores_, attn_probs = self._sliding_chunks_multi_head_attention(
        #         query_vectors,
        #         key_vectors,
        #         value_vectors,
        #         attention_mask,
        #         self.attention_window,
        #     )
        #
        # outputs = (outputs, attn_probs) if output_attentions else (outputs,)
        #
        # return outputs

        # hidden_state: bs, seq, dim
        # self_memory: bs, self_len, dim

        batch_size, seq_len, embed_dim = hidden_states.size()
        assert (
                embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"
        hidden_states = hidden_states.reshape(batch_size, seq_len // self.attention_window, self.attention_window,
                                              embed_dim)
        query_vectors = self.query(hidden_states)

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        # self_memory = None
        if self_memory is not None:
            _, self_len, _ = self_memory.size()
            memory_states = self_memory.unsqueeze(1).repeat([1, seq_len // self.attention_window, 1, 1])
            hidden_states = torch.cat((hidden_states, memory_states), dim=2)  # (bs, chunk, window + self_len, embed_dim)

        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        query_vectors = query_vectors.view(batch_size, seq_len // self.attention_window, -1, self.num_heads, self.head_dim)
        key_vectors = key_vectors.view(batch_size, seq_len // self.attention_window, -1, self.num_heads, self.head_dim)
        value_vectors = value_vectors.view(batch_size, seq_len // self.attention_window, -1, self.num_heads, self.head_dim)

        outputs, attention_scores_, attn_probs = self._sliding_chunks_attention_w_memory(
            query_vectors,
            key_vectors,
            value_vectors,
            attention_mask,
            self.attention_window,
        )

        outputs = (outputs, attn_probs) if output_attentions else (outputs,)

        return outputs

    def _cal_memory_mask_to_attention(
            self,
            attention_mask,  # (batch_size, seq_len)
            window_size,  # int
            ):
        batch_size, seq_len = attention_mask.size()
        chunks_count = seq_len // window_size
        attention_mask = attention_mask.reshape(batch_size, seq_len // window_size, window_size)
        self_memory_mask = (attention_mask == 0.).sum(2) > 0
        self_memory_mask = (~self_memory_mask) * - 10000.  # (batch_size, self_len)

        # self_memory_mask = self_memory_mask * 0. - 10000.
        # print(self_memory_mask)

        attention_mask = attention_mask.unsqueeze(2).repeat(
            [1, 1, window_size, 1])  # (batch_size, chunks_count, window_size, window_size)

        self_memory_mask = self_memory_mask[:, None, None, :]  # (batch_size, None, None, self_len)
        self_memory_mask = self_memory_mask.repeat([1, chunks_count, window_size, 1])
        # mask = attention_mask + self_memory_mask  # (batch_size, chunks_count, window_size, window_size+self_len)
        mask = torch.cat((attention_mask, self_memory_mask), dim=-1)
        mask = mask[:, None]  # (batch_size, None, chunks_count, window_size, window_size+self_len)
        return mask

    @staticmethod
    def _sliding_chunks_multi_head_attention_w_memory(
            query: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            key: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            value: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            attention_mask: torch.Tensor,  # (batch_size, seq_len)
            window_size: int,
            key_memory: torch.Tensor,  # (batch_size, self_len, num_heads, head_dim)
            value_memory: torch.Tensor,  # (batch_size, self_len, num_heads, head_dim)
    ):
        batch_size, seq_len, num_heads, head_dim = query.size()
        self_len = key_memory.size(1)  # self_len == seq_len // window_size == chunks_count
        assert (
                seq_len % window_size == 0
        ), f"Sequence length should be multiple of {window_size}. Given {seq_len}"
        assert query.size() == key.size()
        chunks_count = seq_len // window_size
        query = query.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)
        key = key.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)
        value = value.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)
        key_memory = key_memory.transpose(1, 2).reshape(batch_size, num_heads, 1, self_len, head_dim)\
            .repeat([1, 1, chunks_count, 1, 1])
        value_memory = value_memory.transpose(1, 2).reshape(batch_size, num_heads, 1, self_len, head_dim) \
            .repeat([1, 1, chunks_count, 1, 1])

        key = torch.cat((key, key_memory),
                        dim=3)  # (batch_size, num_heads, chunks_count, window_size + self_len, head_dim)
        value = torch.cat((value, value_memory),
                          dim=3)  # (batch_size, num_heads, chunks_count, window_size + self_len, head_dim)

        # (batch_size, num_heads, chunks_count, window_size, window_size + self_len)
        attention_scores = torch.einsum("bncxd,bncyd->bncxy", (query, key))  # multiply

        # generate (attention_masks, self_memory) corresponding to (attention_scores, memory tokens)
        attention_mask = attention_mask.reshape(batch_size, chunks_count, window_size)
        self_memory_mask = (attention_mask == 0).sum(2) > 0
        self_memory_mask = (~self_memory_mask) * - 10000.  # (batch_size, self_len)
        attention_mask = attention_mask.unsqueeze(-1).repeat(
            [1, 1, 1, window_size])  # (batch_size, chunks_count, window_size, window_size)
        self_memory_mask = self_memory_mask[:, None, None, :]  # (batch_size, None, None, self_len)
        self_memory_mask = self_memory_mask.repeat([1, chunks_count, window_size, 1])
        # print(attention_mask.shape)
        # print(self_memory_mask.shape)
        # mask = attention_mask + self_memory_mask  # (batch_size, chunks_count, window_size, window_size+self_len)
        mask = torch.cat((attention_mask, self_memory_mask), dim=-1)
        mask = mask[:, None]  # (batch_size, None, chunks_count, window_size, window_size+self_len)

        attention_scores = attention_scores + mask

        # (batch_size, num_heads, chunks_count, window_size, window_size + self_len)
        weighted_att = torch.softmax(attention_scores, dim=-1)

        # (batch_size, num_heads, chunks_count, window_size + self_len, head_dim)
        output = torch.einsum("bhcws, bhcsd->bhcwd", (weighted_att, value)).reshape(batch_size, num_heads, seq_len,
                                                                                  head_dim)

        # (batch_size, seq_len, hidden_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return output, weighted_att

    def _sliding_chunks_attention_w_memory(
            self,
            query: torch.Tensor,  # (batch_size, chunks_count, given_window_size, num_heads, head_dim)
            key: torch.Tensor,  # (batch_size, chunks_count, window_size, num_heads, head_dim)
            value: torch.Tensor,  # (batch_size, chunks_count, window_size, num_heads, head_dim)
            attention_mask: torch.Tensor,  # (batch_size, seq_len)
            given_window_size: int,
    ):
        batch_size, chunks_count, window_size, num_heads, head_dim = key.size()
        assert given_window_size == query.size(2)
        # window_size = given_window_size + self_len
        query = query.permute(0, 3, 1, 2, 4)  # batch_size, num_heads, chunks_count, given_window_size, head_dim
        key = key.permute(0, 3, 1, 2, 4)  # batch_size, num_heads, chunks_count, window_size, head_dim
        value = value.permute(0, 3, 1, 2, 4)  # batch_size, num_heads, chunks_count, window_size, head_dim

        # (batch_size, num_heads, chunks_count, given_window_size, window_size)
        attention_scores = torch.einsum("bncxd, bncyd->bncxy", (query, key))  # multiply

        # generate (attention_masks, self_memory) corresponding to (attention_scores, memory tokens)
        if given_window_size == window_size:
            attention_mask = attention_mask.reshape(batch_size, chunks_count, window_size)
            mask = attention_mask[:, None, :, None, :]  # (batch_size, None, chunks_count, None, window_size)
        else:
            mask = self._cal_memory_mask_to_attention(attention_mask, given_window_size)

        attention_scores = attention_scores + mask

        # (batch_size, num_heads, chunks_count, window_size, window_size)
        weighted_att = torch.softmax(attention_scores, dim=-1)
        # weighted_att = nn.functional.softmax(attention_scores, dim=-1)
        # weighted_att = self.dropout(weighted_att)

        # (batch_size, num_heads, chunks_count, window_size, head_dim)
        output = torch.einsum("bhcws, bhcsd->bhcwd", (weighted_att, value))\
            .reshape(batch_size, num_heads, chunks_count*given_window_size, head_dim)
        # (batch_size, seq_len, hidden_dim)
        output = output.transpose(1, 2).reshape(batch_size, chunks_count*given_window_size, -1)
        return output, attention_scores, weighted_att

    @staticmethod
    def _sliding_chunks_multi_head_attention(
            query: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            key: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            value: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            attention_mask: torch.Tensor,  # (batch_size, seq_len)
            window_size: int,
    ):
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
                seq_len % window_size == 0
        ), f"Sequence length should be multiple of {window_size}. Given {seq_len}"
        assert query.size() == key.size()
        chunks_count = seq_len // window_size
        query = query.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)
        key = key.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)
        value = value.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)

        # (batch_size, num_heads, chunks_count, window_size, window_size)
        attention_scores = torch.einsum("bncxd,bncyd->bncxy", (query, key))  # multiply

        # generate (attention_masks, self_memory) corresponding to (attention_scores, memory tokens)
        attention_mask = attention_mask.reshape(batch_size, chunks_count, window_size)
        mask = attention_mask[:, None, :, None, :]  # (batch_size, None, chunks_count, None, window_size)

        attention_scores = attention_scores + mask

        # (batch_size, num_heads, chunks_count, window_size, window_size)
        weighted_att = torch.softmax(attention_scores, dim=-1)
        # weighted_att = nn.functional.softmax(attention_scores, dim=-1)
        # weighted_att = self.dropout(weighted_att)

        # (batch_size, num_heads, chunks_count, window_size, head_dim)
        output = torch.einsum("bhcws, bhcsd->bhcwd", (weighted_att, value)).reshape(batch_size, num_heads, seq_len,
                                                                                    head_dim)

        # (batch_size, seq_len, hidden_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return output, attention_scores, weighted_att

    def _sliding_chunks_multi_head_attention_(
            self,
            query: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            key: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            value: torch.Tensor,  # (batch_size, seq_len, num_heads, head_dim)
            attention_mask: torch.Tensor,  # (batch_size, seq_len)
            window_size: int,
    ):
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
                seq_len % window_size == 0
        ), f"Sequence length should be multiple of {window_size}. Given {seq_len}"
        assert query.size() == key.size()
        chunks_count = seq_len // window_size
        query = query.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)
        key = key.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)
        value = value.transpose(1, 2).reshape(batch_size, num_heads, chunks_count, window_size, head_dim)

        query = self._chunk(query, window_size // 2)  # (bs, heads, chunks_count * 2 -1, window_size // 2, head_dim)
        key = self._chunk(key, window_size // 2)

        # (batch_size, num_heads, chunks_count * 2 - 1, window_size // 2, window_size // 2)
        attention_scores = torch.einsum("bncxd, bncyd->bncxy", (query, key))  # multiply

        diagonal_attention_scores = attention_scores.new_empty(
            (batch_size * num_heads, chunks_count * 2, window_size // 2, window_size // 2 * 3)
        )

        # # generate (attention_masks, self_memory) corresponding to (attention_scores, memory tokens)
        # attention_mask = attention_mask.reshape(batch_size, chunks_count, window_size)
        # mask = attention_mask[:, None, :, None, :]  # (batch_size, None, chunks_count, None, window_size)
        #
        # attention_scores = attention_scores + mask

        # (batch_size, num_heads, chunks_count, window_size, window_size)
        weighted_att = torch.softmax(attention_scores, dim=-1)
        # weighted_att = nn.functional.softmax(attention_scores, dim=-1)
        # weighted_att = self.dropout(weighted_att)

        # (batch_size, num_heads, chunks_count, window_size, head_dim)
        output = torch.einsum("bhcws, bhcsd->bhcwd", (weighted_att, value)).reshape(batch_size, num_heads, seq_len,
                                                                                    head_dim)

        # (batch_size, seq_len, hidden_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return output, attention_scores, weighted_att

    @staticmethod
    def _chunk(hidden_states, window_overlap):

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())  # (bs, head, chunk, 2*window_overlap, head_dim)
        chunk_size[2] = chunk_size[2] * 2 - 1

        chunk_stride = list(hidden_states.stride())
        chunk_stride[2] = chunk_stride[2] // 2

        # return (bs, head, chunk*2-1, window_overlap, head_dim)
        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


class BertSelfOutput(nn.Module):
    def __init__(self, config, cus_config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_scale = nn.Linear(cus_config.usr_dim + cus_config.prd_dim, config.hidden_size)
        self.dense_bias = nn.Linear(cus_config.usr_dim + cus_config.prd_dim, config.hidden_size)
        self.LayerNorm = LN_in_Transformer(config.hidden_size, eps=config.layer_norm_eps, up=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, user_embedding, item_embedding, mask):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if user_embedding is not None and item_embedding is not None:
            up = torch.cat((user_embedding, item_embedding), dim=-1)
            up_scale, up_bias = self.dense_scale(up), self.dense_bias(up)
        else:
            up_scale, up_bias = None, None
        hidden_states, _ = self.LayerNorm(hidden_states + input_tensor, up_scale, up_bias, mask)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, cus_config, layer_id):
        super().__init__()
        self.self = BertSelfAttention(config, cus_config, layer_id)
        self.output = BertSelfOutput(config, cus_config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        user_embedding=None,
        item_embedding=None,
        attention_mask=None,
        self_memory=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            self_memory,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states, user_embedding, item_embedding, attention_mask)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config, cus_config, layer_id):
        super().__init__()
        self.cus_config = cus_config
        self.layer_id = layer_id
        attention_window = cus_config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2
        self.attention_window = attention_window
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LN_in_Transformer(config.hidden_size, eps=config.layer_norm_eps, up=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, memory_states = self.LayerNorm(hidden_states + input_tensor, mask=mask, window_size=self.attention_window)
        return hidden_states, memory_states


class BertLayer(nn.Module):
    def __init__(self, config, cus_config, layer_id):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config, cus_config, layer_id)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config, cus_config, layer_id)

    def forward(
        self,
        hidden_states,
        user_embedding=None,
        item_embedding=None,
        attention_mask=None,
        self_memory=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            user_embedding,
            item_embedding,
            attention_mask,
            self_memory,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                user_embedding,
                item_embedding,
                attention_mask,
                self_memory,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, attention_mask
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output, attention_mask):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, attention_mask)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config, cus_config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, cus_config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        user_embedding=None,
        item_embedding=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        memory_states = None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    memory_states,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    user_embedding,
                    item_embedding,
                    attention_mask,
                    memory_states,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states, memory_states = layer_outputs[0]

            memory_states = None  # for ablation on structured text information

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = LN_in_Transformer(config.hidden_size, eps=config.layer_norm_eps, up=False)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states, _ = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LN_based_attention):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertModel(BertPreTrainedModel):
    def __init__(self, config, cus_config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.cus_config = cus_config

        if isinstance(cus_config.attention_window, int):
            assert cus_config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert cus_config.attention_window > 0, "`config.attention_window` has to be positive"
            cus_config.attention_window = [cus_config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(cus_config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(cus_config.attention_window)}"
            )

        self.user_item_embeddings = UserItemEmbeddings(cus_config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, cus_config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def init_personalized(self):
        self.user_item_embeddings.reset_parameters()

        # for layer in self.encoder.layer:
        #     layer.attention.self.p_q.reset_parameters()
        #     layer.attention.self.p_k.reset_parameters()
        #     layer.attention.self.p_v.reset_parameters()
        #
        # if self.cus_config.injection in ['inserting', 'stacking']:
        #     for layer in self.encoder.more_layer:
        #         layer.attention.self.p_q.reset_parameters()
        #         layer.attention.self.p_k.reset_parameters()
        #         layer.attention.self.p_v.reset_parameters()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.cus_config.attention_window
            if isinstance(self.cus_config.attention_window, int)
            else max(self.cus_config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens
            if token_type_ids is not None:
                token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds


    def forward(
        self,
        input_ids=None,
        user_ids=None,
        item_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # if token_type_ids is None:
        #     if hasattr(self.embeddings, "token_type_ids"):
        #         buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[
            :, 0, 0, :
        ]  # (bs, seq) # 0 indicates not masked tokens; -10000. indicates masked tokens

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_user, embedding_item = self.user_item_embeddings(user_ids, item_ids)

        # embedding_user = embedding_user.zero_()  # for user ablation
        # embedding_item = embedding_item.zero_()  # for product ablation

        encoder_outputs = self.encoder(
            embedding_output,
            embedding_user,
            embedding_item,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # sequence_output, memory_states = encoder_outputs[0]
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # undo padding
        if padding_len > 0:
            # unpad `sequence_output` because the calling function is expecting a length == input_ids.size(1)
            sequence_output = sequence_output[:, :-padding_len]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )