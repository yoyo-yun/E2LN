import netrc
import os
import time
import torch
import datetime
import numpy as np

from tqdm import tqdm
from models.get_optim import get_Adam_optim, get_Adam_optim_v2
from trainer.utils import multi_acc, multi_mse, load_vectors_LSTM, load_vectors, load_dataset_whole_document_bert
from models import *
from transformers import BertTokenizer, BertConfig

ALL_MODLES = {
    'bert': baselines.BertForSequenceClassification_doc,
}


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.step_count = 0
        self.losses = []
        self.losses_whole = []
        self.dev_acc_per_epoch = []
        self.best_dev_acc = 0

    def ensureDirs(self, *dir_paths):
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def train(self):
        pass

    def train_epoch(self):
        pass

    def eval(self, eval_itr):
        pass

    def empty_log(self, version):
        if (os.path.exists(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')):
            os.remove(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def get_logging(self, loss, acc, rmse, eval='training'):
        logs = \
            '==={} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
            "total_loss:{:>2.4f}\ttotal_acc:{:>2.4f}\ttotal_rmse:{:>2.4f}".format(loss, acc, rmse) + "\n"
        return logs


class BERTTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.train_itr, self.dev_itr, self.test_itr, self.usr_stoi, self.prd_stoi = \
            load_dataset_whole_document_bert(config, from_sratch=False)
        self.pad_token_id = self.tokenizer.pad_token_id

        net = ALL_MODLES[config.model].from_pretrained(pretrained_weights, num_labels=config.num_labels, cus_config=self.config)
        net.bert.embeddings.extend_word_embedding()
        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net).to(self.config.device)
        else:
            self.net = net.to(self.config.device)
        self.optim, self.scheduler = get_Adam_optim_v2(config, self.net)

        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.moniter_per_step = len(self.train_itr) // 10
        training_steps_per_epoch = len(self.train_itr) // (config.gradient_accumulation_steps)
        self.config.num_train_optimization_steps = self.config.TRAIN.max_epoch * training_steps_per_epoch

    def train(self):
        # Save log information
        logfile = open(
            self.config.log_path +
            '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
            'a+'
        )
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n'
        )
        logfile.close()
        for epoch in range(0, self.config.TRAIN.max_epoch):
            self.net.train()
            train_loss, train_acc, train_rmse, best_losses_per_epoch, best_acc_per_epoch, best_rmse_per_peoch = self.train_epoch()


            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, train_rmse, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         logs)

            eval_logs = self.get_logging(best_losses_per_epoch, best_acc_per_epoch, best_rmse_per_peoch, eval="evaluating")
            print("\r" + eval_logs)

            # logging evaluating logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         eval_logs)

            # early stopping
            if best_acc_per_epoch > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = best_acc_per_epoch
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                    early_stop_logs = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt' + "\n" + \
                                      "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc)
                    print(early_stop_logs)
                    self.logging(
                        self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                        early_stop_logs)
                    break

    def train_epoch(self, epoch=1):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        eval_best_loss = 0.
        eval_best_acc = 0.
        eval_best_rmse = 0.
        epoch_tqdm = tqdm(self.train_itr)
        epoch_tqdm.set_description_str("Processing Epoch: {}".format(epoch))
        self.optim.zero_grad()
        for step, batch in enumerate(epoch_tqdm):
            self.net.train()
            # input_ids, label, usr, prd = batch
            input_ids, label, usr, prd, lengths = batch['batch_text_indices'], batch['batch_labels'], batch['batch_usr_indices'], \
                                         batch['batch_prd_indices'], batch['batch_lengths']
            input_ids = input_ids.to(self.config.device)
            attention_mask = (input_ids != self.pad_token_id).long().to(self.config.device)  # id of [PAD] is 0
            labels = label.long().to(self.config.device)
            usr = torch.Tensor([self.usr_stoi[x] for x in usr]).long().to(self.config.device)
            prd = torch.Tensor([self.prd_stoi[x] for x in prd]).long().to(self.config.device)
            logits = self.net(input_ids=input_ids,
                              user_ids=usr,
                              item_ids=prd,
                              attention_mask=attention_mask
                              )[0]
            loss = loss_fn(logits, labels)
            metric_acc = acc_fn(labels, logits)
            metric_mse = mse_fn(labels, logits)

            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optim.step()
                if self.scheduler is not None: self.scheduler.step()
                self.optim.zero_grad()

            total_loss.append(loss.item())
            total_acc.append(metric_acc.item())
            total_mse.append(metric_mse.item())

            if step % self.moniter_per_step == 0 and step != 0:
                self.net.eval()
                with torch.no_grad():
                    eval_loss, eval_acc, eval_rmse = self.eval(self.dev_itr)

                # monitoring eval metrices
                if eval_acc > eval_best_acc:
                    eval_best_loss = eval_loss
                    eval_best_acc = eval_acc
                    eval_best_rmse = eval_rmse

                if eval_acc > self.best_dev_acc:
                    # saving models
                    self.saving_model()

        return np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean()),\
               eval_best_loss, eval_best_acc, eval_best_rmse

    def saving_model(self):
        SAVED_MODEL_PATH = self.config.ckpts_path
        self.ensureDirs(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        self.tokenizer.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        if self.config.n_gpu > 1:
            self.net.module.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        else:
            self.net.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))

    def load_state(self):
        SAVED_MODEL_PATH = self.config.ckpts_path
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        net = ALL_MODLES[self.config.model].from_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset), cus_config = self.config)
        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net).to(self.config.device)
        else:
            self.net = net.to(self.config.device)

    def eval(self, eval_itr):
        loss_fn = torch.nn.CrossEntropyLoss()
        metric_fn = multi_acc
        mse_fn = multi_mse
        total_logit = []
        total_labels = []
        total_loss = []
        self.net.eval()
        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            # input_ids, label, usr, prd = batch
            input_ids, label, usr, prd, lengths = batch['batch_text_indices'], batch['batch_labels'], batch['batch_usr_indices'], \
                                         batch['batch_prd_indices'], batch['batch_lengths']
            input_ids = input_ids.to(self.config.device)
            attention_mask = (input_ids != self.pad_token_id).long().to(self.config.device)  # id of [PAD] is 0
            labels = label.long().to(self.config.device) # (bs, )
            usr = torch.Tensor([self.usr_stoi[x] for x in usr]).long().to(self.config.device)
            prd = torch.Tensor([self.prd_stoi[x] for x in prd]).long().to(self.config.device)
            logits = self.net(input_ids=input_ids,
                              user_ids=usr,
                              item_ids=prd,
                              attention_mask=attention_mask
                              )[0]
            loss = loss_fn(logits, labels)
            total_loss.append(loss.item())
            total_logit.extend(logits.cpu().detach().tolist())
            total_labels.extend(labels.cpu().detach().tolist())

            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    # int(len(eval_itr.dataset) / self.config.TRAIN.batch_size)
                    int(len(eval_itr) / self.config.TRAIN.batch_size)
                    - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)   -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    # step, int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    step, int(len(eval_itr) / self.config.TRAIN.batch_size),
                    # 100 * (step) / int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    100 * (step) / int(len(eval_itr) / self.config.TRAIN.batch_size),
                    int(h), int(m), int(s)),
                end="")

        return np.array(total_loss).mean(0), \
               metric_fn(torch.tensor(total_labels), torch.tensor(total_logit)), \
               mse_fn(torch.tensor(total_labels), torch.tensor(total_logit)).sqrt()

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.config.version)
            self.train()
        elif run_mode == 'val':
            eval_loss, eval_acc, eval_rmse = self.eval(self.dev_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, "evaluating")
            print("\r" + eval_logs)
        elif run_mode == 'test':
            self.load_state()
            eval_loss, eval_acc, eval_rmse = self.eval(self.test_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, "evaluating")
            print("\r" + eval_logs)
        else:
            exit(-1)

