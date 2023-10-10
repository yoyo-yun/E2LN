import torch
import os
import torch.nn.functional as F
from dataset import *
from cfgs.constants import DATASET_PATH_MAP, PRE_TRAINED_VECTOR_PATH
from torch.utils.data import DataLoader

DATASET_MAP = {
    # "imdb": IMDB,
    "imdb": imdb.IMDBHierarchical,
    "yelp_13": yelp_13.YELP13Hierarchical,
    "yelp_14": yelp_14.YELP14Hierarchical,
}

DATASET_MAP_BERT = {
    "imdb": dataset_bert.IMDB,
    "yelp_13": dataset_bert.YELP_13,
    "yelp_14": dataset_bert.YELP_14
}

DATASET_MAP_LSTM = {
    "imdb": imdb.IMDB,
    "yelp_13": yelp_13.YELP13,
    "yelp_14": yelp_14.YELP14,
    "sst": dataset_from_torchtxt.SST2_,
    "sst5": dataset_from_torchtxt.SST5_,
    "emobank": dataset_from_torchtxt.EmoBank,
}

class MyVector(object):
    def __init__(self):
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None


def load_sentence_vectors(config, show_example=False, show_statics=False):
    import time
    start_time = time.time()
    print("====loading vectors...")
    train, val, test = DATASET_MAP_LSTM[config.dataset].iters(path=DATASET_PATH_MAP[config.dataset],
                                                         batch_size=config.TRAIN.batch_size, shuffle=True,
                                                         device=config.device)
    print("done!")
    print()
    end_time = time.time()

    # example an iterator
    if show_example:
        for i in train:
            print("====text size...")
            # text: (batch, sent, seq) length:(batch, sent)
            print("text:{}\tlength:{}".format(i.text[0].shape, i.text[1].shape))
            # label: (batch,)
            print("label size")
            print("text:{}".format(i.label.shape))

            print("====example...")
            print("text:    ", str(i.text[0][0]))
            print("length:  ", str(i.text[1][0]))
            print("label:   ", str(i.label[0]))
            break
    print("taking times: {:.2f}s.".format(end_time - start_time))
    print()

    # update configuration
    config.num_classes = train.dataset.NUM_CLASSES
    config.text_embedding = train.dataset.TEXT_FIELD.vocab.vectors
    config.pad_idx = train.dataset.TEXT_FIELD.vocab.stoi[train.dataset.TEXT_FIELD.pad_token]

    print("===Train size       : " + str(len(train.dataset)))
    print()
    print("===Validation size  : " + str(len(val.dataset)))
    print()
    print("===Test size        : " + str(len(test.dataset)))
    print()
    print("===common datasets information...")
    print("num_labels          : " + str(config.num_classes))
    print("pad_idx             : " + str(config.pad_idx))
    print("text vocabulary size: " + str(config.text_embedding.shape[0]))
    print()
    return train, val, test


def load_vectors(config, show_example=False, show_statics=False):
    import time
    start_time = time.time()
    print("====loading vectors...")
    train, val, test = DATASET_MAP[config.dataset].iters(path=DATASET_PATH_MAP[config.dataset],
                                                         batch_size=config.TRAIN.batch_size, shuffle=True,
                                                         device=config.device)
    print("done!")
    print()
    end_time = time.time()

    # example an iterator
    if show_example:
        for i in train:
            print("====text size...")
            # text: (batch, sent, seq) length:(batch, sent)
            print("text:{}\tlength:{}".format(i.text[0].shape, i.text[1].shape))
            # label: (batch,)
            print("label size")
            print("text:{}".format(i.label.shape))
            # user: (batch, 1)
            print("user size")
            print("usr:{}".format(i.usr.shape))
            # product: (batch, 1)
            print("product size")
            print("prd:{}".format(i.prd.shape))

            print("====example...")
            print("text:    ", str(i.text[0][0]))
            print("length:  ", str(i.text[1][0]))
            print("label:   ", str(i.label[0]))
            print("user:    ", str(i.usr[0]))
            print("product: ", str(i.prd[0]))
            break
    print("taking times: {:.2f}s.".format(end_time - start_time))
    print()

    # statistic
    if show_statics:
        words_per_setence = []
        sentences_per_document = []
        for batch in train:
            sentences_per_document.append(batch.text[1])
            words_per_setence.append(batch.text[2])
        for batch in val:
            sentences_per_document.append(batch.text[1])
            words_per_setence.append(batch.text[2])
        for batch in test:
            sentences_per_document.append(batch.text[1])
            words_per_setence.append(batch.text[2])

    # update configuration
    config.num_classes = train.dataset.NUM_CLASSES
    config.text_embedding = train.dataset.TEXT_FIELD.nesting_field.vocab.vectors
    config.usr_vocab = train.dataset.USR_FIELD.vocab
    config.prd_vocab = train.dataset.PRD_FIELD.vocab
    config.pad_idx = train.dataset.TEXT_FIELD.nesting_field.vocab.stoi[train.dataset.TEXT_FIELD.pad_token]

    print("===Train size       : " + str(len(train.dataset)))
    print()
    print("===Validation size  : " + str(len(val.dataset)))
    print()
    print("===Test size        : " + str(len(test.dataset)))
    print()
    print("===common datasets information...")
    print("num_labels          : " + str(config.num_classes))
    print("pad_idx             : " + str(config.pad_idx))
    print("text vocabulary size: " + str(config.text_embedding.shape[0]))
    print("usr  vocabulary size: " + str(len(train.dataset.USR_FIELD.vocab)))
    print("prd  vocabulary size: " + str(len(train.dataset.PRD_FIELD.vocab)))
    print()
    return train, val, test


def load_vectors_LSTM(config, show_example=False, show_statics=False):
    import time
    start_time = time.time()
    print("====loading vectors...")
    train, val, test = DATASET_MAP_LSTM[config.dataset].iters(path=DATASET_PATH_MAP[config.dataset],
                                                         batch_size=config.TRAIN.batch_size, shuffle=True,
                                                         device=config.device)
    print("done!")
    print()
    end_time = time.time()

    # example an iterator
    if show_example:
        for i in train:
            print("====text size...")
            # text: (batch, seq) length:(batch, sent)
            print("text:{}\tlength:{}".format(i.text[0].shape, i.text[1].shape))
            # label: (batch,)
            print("label size")
            print("text:{}".format(i.label.shape))
            # user: (batch, 1)
            print("user size")
            print("usr:{}".format(i.usr.shape))
            # product: (batch, 1)
            print("product size")
            print("prd:{}".format(i.prd.shape))

            print("====example...")
            print("text:    ", str(i.text[0][0]))
            print("length:  ", str(i.text[1][0]))
            print("label:   ", str(i.label[0]))
            print("user:    ", str(i.usr[0]))
            print("product: ", str(i.prd[0]))
            break
    print("taking times: {:.2f}s.".format(end_time - start_time))
    print()

    # statistic
    if show_statics:
        words_per_setence = []
        for batch in train:
            words_per_setence.append(batch.text[1])
        for batch in val:
            words_per_setence.append(batch.text[1])
        for batch in test:
            words_per_setence.append(batch.text[1])

    # update configuration
    config.num_classes = train.dataset.NUM_CLASSES
    config.text_embedding = train.dataset.TEXT_FIELD.vocab.vectors
    config.usr_embedding = train.dataset.USR_FIELD.vocab.vectors
    config.prd_embedding = train.dataset.PRD_FIELD.vocab.vectors
    config.pad_idx = train.dataset.TEXT_FIELD.vocab.stoi[train.dataset.TEXT_FIELD.pad_token]

    print("===Train size       : " + str(len(train.dataset)))
    print()
    print("===Validation size  : " + str(len(val.dataset)))
    print()
    print("===Test size        : " + str(len(test.dataset)))
    print()
    print("===common datasets information...")
    print("num_labels          : " + str(config.num_classes))
    print("pad_idx             : " + str(config.pad_idx))
    print("text vocabulary size: " + str(config.text_embedding.shape[0]))
    print("usr  vocabulary size: " + str(config.usr_embedding.shape[0]))
    print("prd  vocabulary size: " + str(config.prd_embedding.shape[0]))
    print()
    return train, val, test


def multi_acc(y, preds):
    """
    get accuracy

    preds: logits
    y: true label
    """
    preds = torch.argmax(F.softmax(preds), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

def multi_mse(y, preds):
    mse_loss = torch.nn.MSELoss()
    preds = torch.argmax(F.softmax(preds), dim=1)
    return mse_loss(y.float(), preds.float())


def load_baselines_datasets(path, field='train', strategy='tail'):
    return torch.load(os.path.join(path, '{}_{}.pt'.format(field, strategy)))


class Data(torch.utils.data.Dataset):
    sort_key = None
    def __init__(self, *data):
        assert all(len(data[0]) == len(d) for d in data)
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)


def load_document4baseline_from_local(config):
    try:
        train_input_ids, train_labels, train_users, train_products = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='train', strategy=config.strategy)
        dev_input_ids, dev_labels, dev_users, dev_products = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='dev', strategy=config.strategy)
        test_input_ids, test_labels, test_users, test_products = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='test', strategy=config.strategy)

        processor = DATASET_MAP_BERT[config.dataset]()
        config.num_labels = processor.NUM_CLASSES
        train_dataset = Data(train_input_ids, train_labels, train_users, train_products)
        dev_dataset = Data(dev_input_ids, dev_labels, dev_users, dev_products)
        test_dataset = Data(test_input_ids, test_labels, test_users, test_products)

        train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config.TRAIN.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=config.TRAIN.batch_size)

        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
        config.usr_size = len(usr_stoi)
        config.prd_size = len(prd_stoi)
        config.num_train_optimization_steps = int(
            len(
                train_dataset) / config.TRAIN.batch_size / config.gradient_accumulation_steps) * config.TRAIN.max_epoch
        print("===loading {} document from local...".format(config.strategy))
        print("Done!")
        return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi
    except:
        save_datasets(config)
        train_input_ids, train_labels, train_users, train_products = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='train', strategy=config.strategy)
        dev_input_ids, dev_labels, dev_users, dev_products = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='dev', strategy=config.strategy)
        test_input_ids, test_labels, test_users, test_products = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='test', strategy=config.strategy)

        processor = DATASET_MAP_BERT[config.dataset]()
        config.num_labels = processor.NUM_CLASSES

        train_dataset = Data(train_input_ids, train_labels, train_users, train_products)
        dev_dataset = Data(dev_input_ids, dev_labels, dev_users, dev_products)
        test_dataset = Data(test_input_ids, test_labels, test_users, test_products)

        train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config.TRAIN.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=config.TRAIN.batch_size)

        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
        config.usr_size = len(usr_stoi)
        config.prd_size = len(prd_stoi)
        config.num_train_optimization_steps = int(
            len(
                train_dataset) / config.TRAIN.batch_size / config.gradient_accumulation_steps) * config.TRAIN.max_epoch

        return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi


def save_datasets(config):
    from transformers import BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    processor = DATASET_MAP_BERT[config.dataset]()
    train_examples, dev_examples, test_examples = processor.get_documents()

    train_texts, train_labels, train_users, train_products = [], [], [], []
    dev_texts, dev_labels, dev_users, dev_products = [], [], [], []
    test_texts, test_labels, test_users, test_products = [], [], [], []

    print("==loading train datasets")
    for step, example in enumerate(train_examples):
        train_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
        train_labels.append(example.label)
        train_users.append(example.user)
        train_products.append(example.product)
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                          step / len(train_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading dev datasets")
    for step, example in enumerate(dev_examples):
        dev_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
        dev_labels.append(example.label)
        dev_users.append(example.user)
        dev_products.append(example.product)
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                          step / len(dev_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading test datasets")
    for step, example in enumerate(test_examples):
        test_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
        test_labels.append(example.label)
        test_users.append(example.user)
        test_products.append(example.product)
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                          step / len(test_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    train_data = train_texts, train_labels, train_users, train_products
    dev_data = dev_texts, dev_labels, dev_users, dev_products
    test_data = test_texts, test_labels, test_users, test_products
    torch.save(train_data,
               os.path.join(DATASET_PATH_MAP[config.dataset], 'train_{}.pt'.format(config.strategy)))
    torch.save(dev_data,
               os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_{}.pt'.format(config.strategy)))
    torch.save(test_data,
               os.path.join(DATASET_PATH_MAP[config.dataset], 'test_{}.pt'.format(config.strategy)))

    users, products = processor.get_attributes()
    usr_stoi, prd_stoi = load_attr_vocab(config.dataset, users, products)
    config.num_labels = processor.NUM_CLASSES
    config.num_usrs = len(usr_stoi)
    config.num_prds = len(prd_stoi)


def load_vocab(path, field='usr'):
    # itos, stoi, vectors, dim
    return torch.load(os.path.join(path, '{}.pt'.format(field)))

def load_attr_vocab(dataset, users, products):
    try:
        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='prd')
    except:
        usr_vocab = build_vocab(users)
        prd_vocab = build_vocab(products)
        save_vectors(DATASET_PATH_MAP[dataset], usr_vocab, field='usr')
        save_vectors(DATASET_PATH_MAP[dataset], prd_vocab, field='prd')
        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='prd')
    return usr_stoi, prd_stoi

def processor4baseline_over_one_example(text, tokenizer, config):
    # [PAD] id is 0
    tokens = tokenizer.tokenize(text)
    new_tokens = _truncate_and_pad(tokens, config.max_length - 2, config.strategy)
    input_id = tokenizer.convert_tokens_to_ids(new_tokens)
    return torch.tensor(input_id, dtype=torch.long)

def save_vectors(path, vocab, field='usr'):
    # itos, stoi, vectors, dim
    data = vocab.itos, vocab.stoi
    torch.save(data, os.path.join(path, '{}.pt'.format(field)))

def _truncate_and_pad(tokens, max_length=510, pad_strategy="head"):
    """
    Truncates a sequence in place to the maximum length
    :param tokens:
    :param max_length:
    :param pad_strategy: "head", "tail", "both"
    :return:
    """
    total_length = len(tokens)
    if total_length > max_length:
        if pad_strategy == 'head':
            return ['[CLS]'] + tokens[:max_length] + ['[SEP]']
        if pad_strategy == 'tail':
            return ['[CLS]'] + tokens[-max_length:]+ ['[SEP]']
        if pad_strategy == 'both':
            return ['[CLS]'] + tokens[:128] + tokens[-max_length+128:] + ['[SEP]']
        return
    else:
        return ['[CLS]'] + tokens + ['[SEP]'] + ['[PAD]'] * (max_length-total_length)

def build_vocab(counter):
    from torchtext.legacy.vocab import Vocab
    vocab = Vocab(counter=counter, specials=[], vectors=None)
    return vocab


def save_dataset_hierarchy(config):
    from transformers import BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    processor = DATASET_MAP_BERT[config.dataset]()
    train_examples, dev_examples, test_examples = processor.get_sent_doc()
    train_data, dev_data, test_data = [], [], []

    def over_one_example_hierarchy(text, tokenizer):
        tt = []
        for t in text:
            tokens = tokenizer.tokenize(t)
            tt.append(tokenizer.convert_tokens_to_ids(tokens))
        return tt

    print("==loading train datasets")
    for step, example in enumerate(train_examples):
        train_data.append(((over_one_example_hierarchy(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                          step / len(train_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading dev datasets")
    for step, example in enumerate(dev_examples):
        dev_data.append(((over_one_example_hierarchy(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                          step / len(dev_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading test datasets")
    for step, example in enumerate(test_examples):
        test_data.append(((over_one_example_hierarchy(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                          step / len(test_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    torch.save(train_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'train_hie.pt'))
    torch.save(dev_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_hie.pt'))
    torch.save(test_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'test_hie.pt'))

    users, products = processor.get_attributes()
    usr_stoi, prd_stoi = load_attr_vocab(config.dataset, users, products)
    config.num_labels = processor.NUM_CLASSES
    config.num_usrs = len(usr_stoi)
    config.num_prds = len(prd_stoi)


def load_dataset_hierarchy(config, from_sratch=True):
    processor = DATASET_MAP_BERT[config.dataset]()
    config.num_labels = processor.NUM_CLASSES
    try:
        if from_sratch: raise Exception
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'train_hie.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_hie.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'test_hie.pt'))
        print("===loading {} document from local...".format(config.strategy))
        print("Done!")
    except:
        save_dataset_hierarchy(config)
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'train_hie.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_hie.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'test_hie.pt'))

    from trainer.bucket_iteractor import BucketIterator
    train_dataloader = BucketIterator(train_data, config.TRAIN.batch_size, max_length=config.max_length, shuffle=True, device=config.device)
    dev_dataloader = BucketIterator(dev_data, config.TRAIN.batch_size, max_length=config.max_length, shuffle=True, device=config.device)
    test_dataloader = BucketIterator(test_data, config.TRAIN.batch_size, max_length=config.max_length, shuffle=True, device=config.device)

    usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
    prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
    config.usr_size = len(usr_stoi)
    config.prd_size = len(prd_stoi)
    config.num_train_optimization_steps = int(
        len(train_dataloader) / config.gradient_accumulation_steps) * config.TRAIN.max_epoch

    return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi


def saving_dataset_whole_document(config):
    from transformers import LongformerTokenizer
    pretrained_weights = 'allenai/longformer-base-4096'
    tokenizer = LongformerTokenizer.from_pretrained(pretrained_weights)
    processor = DATASET_MAP_BERT[config.dataset]()
    train_examples, dev_examples, test_examples = processor.get_documents()

    train_data, dev_data, test_data = [], [], []

    def over_one_example(text, tokenizer):
        tokens = tokenizer.tokenize(text)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        return input_id

    print("==loading train datasets")
    for step, example in enumerate(train_examples):
        train_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                          step / len(train_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading dev datasets")
    for step, example in enumerate(dev_examples):
        dev_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                          step / len(dev_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading test datasets")
    for step, example in enumerate(test_examples):
        test_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                          step / len(test_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    torch.save(train_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole.pt'))
    torch.save(dev_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole.pt'))
    torch.save(test_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole.pt'))

    users, products = processor.get_attributes()
    usr_stoi, prd_stoi = load_attr_vocab(config.dataset, users, products)
    config.num_labels = processor.NUM_CLASSES
    config.num_usrs = len(usr_stoi)
    config.num_prds = len(prd_stoi)


def saving_dataset_whole_document_bert(config):
    from transformers import BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    processor = DATASET_MAP_BERT[config.dataset]()
    train_examples, dev_examples, test_examples = processor.get_documents()

    train_data, dev_data, test_data = [], [], []

    def over_one_example(text, tokenizer):
        tokens = tokenizer.tokenize(text)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        return input_id

    print("==loading train datasets")
    for step, example in enumerate(train_examples):
        train_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                          step / len(train_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading dev datasets")
    for step, example in enumerate(dev_examples):
        dev_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                          step / len(dev_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading test datasets")
    for step, example in enumerate(test_examples):
        test_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                          step / len(test_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    torch.save(train_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole_bert.pt'))
    torch.save(dev_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole_bert.pt'))
    torch.save(test_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole_bert.pt'))

    users, products = processor.get_attributes()
    usr_stoi, prd_stoi = load_attr_vocab(config.dataset, users, products)
    config.num_labels = processor.NUM_CLASSES
    config.num_usrs = len(usr_stoi)
    config.num_prds = len(prd_stoi)


def load_dataset_whole_document(config, from_sratch=True):
    processor = DATASET_MAP_BERT[config.dataset]()
    config.num_labels = processor.NUM_CLASSES
    try:
        if from_sratch: raise Exception
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole.pt'))
        print("===loading {} document from local...".format(config.strategy))
        print("Done!")
    except:
        saving_dataset_whole_document(config)
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole.pt'))

    from trainer.bucket_iteractor import BucketIteratorForLongformer
    train_dataloader = BucketIteratorForLongformer(train_data, config.TRAIN.batch_size, shuffle=True, device=config.device)
    dev_dataloader = BucketIteratorForLongformer(dev_data, config.TRAIN.batch_size, shuffle=True, device=config.device)
    test_dataloader = BucketIteratorForLongformer(test_data, config.TRAIN.batch_size, shuffle=True, device=config.device)

    usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
    prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
    config.usr_size = len(usr_stoi)
    config.prd_size = len(prd_stoi)
    config.num_train_optimization_steps = int(
        len(train_dataloader) / config.gradient_accumulation_steps) * config.TRAIN.max_epoch

    return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi


def load_dataset_whole_document_bert(config, from_sratch=True):
    processor = DATASET_MAP_BERT[config.dataset]()
    config.num_labels = processor.NUM_CLASSES
    try:
        if from_sratch: raise Exception
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole_bert.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole_bert.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole_bert.pt'))
        print("===loading {} document from local...".format(config.strategy))
        print("Done!")
    except:
        saving_dataset_whole_document_bert(config)
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole_bert.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole_bert.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole_bert.pt'))

    from trainer.bucket_iteractor import BucketIteratorForBert
    train_dataloader = BucketIteratorForBert(train_data, config.TRAIN.batch_size, shuffle=True, device=config.device)
    dev_dataloader = BucketIteratorForBert(dev_data, config.TRAIN.batch_size, shuffle=True, device=config.device)
    test_dataloader = BucketIteratorForBert(test_data, config.TRAIN.batch_size, shuffle=True, device=config.device)

    usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
    prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
    config.usr_size = len(usr_stoi)
    config.prd_size = len(prd_stoi)
    config.num_train_optimization_steps = int(
        len(train_dataloader) / config.gradient_accumulation_steps) * config.TRAIN.max_epoch

    return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi


# utils for roberta
def saving_dataset_whole_document_roberta(config):
    from transformers import RobertaTokenizer
    pretrained_weights = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
    processor = DATASET_MAP_BERT[config.dataset]()
    train_examples, dev_examples, test_examples = processor.get_documents()

    train_data, dev_data, test_data = [], [], []

    def over_one_example(text, tokenizer):
        tokens = tokenizer.tokenize(text)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        return input_id

    print("==loading train datasets")
    for step, example in enumerate(train_examples):
        train_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                          step / len(train_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading dev datasets")
    for step, example in enumerate(dev_examples):
        dev_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                          step / len(dev_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading test datasets")
    for step, example in enumerate(test_examples):
        test_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                          step / len(test_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    torch.save(train_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole_roberta.pt'))
    torch.save(dev_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole_roberta.pt'))
    torch.save(test_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole_roberta.pt'))

    users, products = processor.get_attributes()
    usr_stoi, prd_stoi = load_attr_vocab(config.dataset, users, products)
    config.num_labels = processor.NUM_CLASSES
    config.num_usrs = len(usr_stoi)
    config.num_prds = len(prd_stoi)

def load_dataset_whole_document_roberta(config, from_sratch=True):
    processor = DATASET_MAP_BERT[config.dataset]()
    config.num_labels = processor.NUM_CLASSES
    try:
        if from_sratch: raise Exception
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole_roberta.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole_roberta.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole_roberta.pt'))
        print("===loading {} document from local...".format(config.strategy))
        print("Done!")
    except:
        saving_dataset_whole_document_roberta(config)
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'train_whole_roberta.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_whole_roberta.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], 'test_whole_roberta.pt'))

    from trainer.bucket_iteractor import BucketIteratorForRoBerta
    train_dataloader = BucketIteratorForRoBerta(train_data, config.TRAIN.batch_size, shuffle=True, device=config.device)
    dev_dataloader = BucketIteratorForRoBerta(dev_data, config.TRAIN.batch_size, shuffle=True, device=config.device)
    test_dataloader = BucketIteratorForRoBerta(test_data, config.TRAIN.batch_size, shuffle=True, device=config.device)

    usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
    prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
    config.usr_size = len(usr_stoi)
    config.prd_size = len(prd_stoi)
    config.num_train_optimization_steps = int(
        len(train_dataloader) / config.gradient_accumulation_steps) * config.TRAIN.max_epoch

    return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi