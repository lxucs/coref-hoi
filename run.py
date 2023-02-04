import logging
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam
from tensorize import CorefDataProcessor
import util
import time
from os.path import join
from metrics import CorefEvaluator
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from model import CorefModel
import conll
import sys
import csv

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class Runner:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = CorefDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None):
        model = CorefModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                # Forward pass
                model.train()
                example_gpu = [d.to(self.device) for d in example]
                _, loss = model(*example_gpu)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                        if f1 > max_f1:
                            max_f1 = f1
                            self.save_model_checkpoint(model, len(loss_history))
                        logger.info('Eval max f1: %.2f' % max_f1)
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Wrap up
        tb_writer.close()
        return loss_history

    def k_best_antecedents_logging(self, doc_number, doc_key, span_starts, span_ends, k_best_antecedent_idx, k_best_antecedent_scores):
        """
        k_best_antecedent logging for the document n° *doc_key* into evaluation/{doc_key}-k_best_antecedents.csv with comma separator
        """
        with open(f"evaluation/{doc_number}-k_best_ant_gold_bound.csv", "w") as file:
            file.write("doc_key,span_idx,span_start,span_end,antecedent_rank,antecedent_score,antecedent_idx,antecedent_start,antecedent_end\n")
            assert len(span_starts) == k_best_antecedent_idx.shape[0]
            nb_spans = len(span_starts)
            k = k_best_antecedent_idx.shape[1] if nb_spans > 0 else 0
            for span_idx in range(nb_spans):
                for rank in range(k):
                    antecedent_idx = k_best_antecedent_idx[span_idx, rank]
                    antecedent_score = k_best_antecedent_scores[span_idx, rank]
                    #if antecedent_score > 0: # else, dummy antecedent : so do not log
                    file.write(f"{doc_key},{span_idx},{span_starts[span_idx]},{span_ends[span_idx]},{rank+1},{antecedent_score},{antecedent_idx},{span_starts[antecedent_idx]},{span_ends[antecedent_idx]}\n")

    def gold_antecedents_logging(self, doc_number, doc_key, tensor_example_gold):
        """
        gold antecedents logging for the document n° *doc_key* into evaluation/{doc_key}-gold_antecedents.csv with comma separator
        """
        with open(f"evaluation/{doc_number}-gold_antecedents.csv", "w") as file:
            file.write("doc_key,anaphor_idx,anaphor_start,anaphor_end,antecedent_idx,antecedent_start,antecedent_end,cluster_id\n")
            (gold_starts, gold_ends, gold_mention_cluster_map) = tensor_example_gold
            nb_spans = gold_starts.shape[0]
            seen_cluster_ids = []
            for span_idx in range(nb_spans):
                cluster_id = gold_mention_cluster_map[span_idx]
                if cluster_id not in seen_cluster_ids: # first mention
                    seen_cluster_ids.append(cluster_id)
                else: # anaphor
                    antecedent_idx = (gold_mention_cluster_map[:span_idx] == cluster_id).nonzero(as_tuple=True)[0][-1].item() # gives the index of the last previous mention that belongs to the same cluster (i.e. the antecedent)
                    file.write(f"{doc_key},{span_idx},{gold_starts[span_idx]},{gold_ends[span_idx]},{antecedent_idx},{gold_starts[antecedent_idx]},{gold_ends[antecedent_idx]},{cluster_id}\n")

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example_gold = tensor_example[7:]
            tensor_example = tensor_example[:7]  # Strip out gold
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                output = model(*example_gpu)
                if output is None: # no candidate
                    span_starts, span_ends, antecedent_idx, antecedent_scores = [], [], [], []
                else:
                    _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = output
                    span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
                    antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()

            ## uncomment the following lines to log k best antecedents for each span of each test document
            # k_best_antecedent_idx, k_best_antecedent_scores = CorefModel.get_k_best_predicted_antecedents(antecedent_idx, antecedent_scores, k=50)
            # self.k_best_antecedents_logging(i, doc_key, span_starts, span_ends, k_best_antecedent_idx, k_best_antecedent_scores)
            # nb_examples = len(tensor_examples)
            # logger.info(f"k best ant. (gold boundaries) logging ... {i+1}/{nb_examples}")
            
            ## uncomment the following line to log the gold anaphor-antecedent pairs
            # self.gold_antecedents_logging(i, doc_key, tensor_example_gold)
            # nb_examples = len(tensor_examples)
            # logger.info(f"gold_antecedents_logging ... {i+1}/{nb_examples}")

            predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
            doc_to_prediction[doc_key] = predicted_clusters

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return f * 100, metrics
    
    def evaluate_from_csv(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None, gold_boundaries=True):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}
        pred_filename = "k_best_ant_gold_bound" if gold_boundaries else "k_best_antecedents"

        model.eval()
        for k in range(1, self.config["max_top_antecedents"] + 1):
            for i, (doc_key, tensor_example) in enumerate(tensor_examples):
                gold_clusters = stored_info['gold'][doc_key]

                # extract span_starts, ends, predicted_antecedent_idx from csv files
                gold_file = open(f"evaluation/{i}-gold_antecedents.csv", "r")
                pred_file = open(f"evaluation/{i}-{pred_filename}.csv", "r")
                gold_reader = list(csv.DictReader(gold_file))
                pred_reader = list(csv.DictReader(pred_file))
                span_starts, span_ends, predicted_antecedent_idx = [], [], []
                visited_idx = []
                for pred_row in pred_reader:
                    idx = pred_row["span_idx"]
                    if idx not in visited_idx:
                    # new span, this row is the top 1 antecedent's
                        visited_idx.append(idx)
                        predicted_antecedent_idx.append(pred_row["antecedent_idx"]) # predict the top 1 by default, will be corrected later if the gold is found
                        pred_start = pred_row["span_start"]
                        pred_end = pred_row["span_end"]
                        span_starts.append(pred_start)
                        span_ends.append(pred_end)
                        gold_antecedent_start = None
                        gold_antecedent_end = None
                        for gold_row in gold_reader:
                            gold_start = gold_row["anaphor_start"]
                            gold_end = gold_row["anaphor_end"]
                            if (pred_start, pred_end) == (gold_start, gold_end):
                                gold_antecedent_start = gold_row["antecedent_start"]
                                gold_antecedent_end = gold_row["antecedent_end"]
                                break
                    
                    if float(pred_row["antecedent_score"]) < 0 or int(pred_row["antecedent_rank"]) > k:
                        continue
                    
                    pred_antecedent_start = pred_row["antecedent_start"]
                    pred_antecedent_end = pred_row["antecedent_end"]
                    if (pred_start, pred_end, pred_antecedent_start, pred_antecedent_end) == (gold_start, gold_end, gold_antecedent_start, gold_antecedent_end):
                        predicted_antecedent_idx[-1] = pred_row["antecedent_idx"] # if gold is found, we correct the current predicted antecedent

                predicted_clusters = model.update_evaluator_v2(span_starts, span_ends, predicted_antecedent_idx, gold_clusters, evaluator)
                doc_to_prediction[doc_key] = predicted_clusters

            p, r, f = evaluator.get_prf()
            metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
            for name, score in metrics.items():
                logger.info('%s: %.2f' % (name, score))
                if tb_writer:
                    tb_writer.add_scalar(name, score, step)

            if official:
                conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
                official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
                logger.info('Official avg F1: %.4f' % official_f1)


    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            tensor_example = tensor_example[:7]
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

        return predicted_clusters, predicted_spans, predicted_antecedents

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'], weight_decay=0)
        ]
        return optimizers
        # grouped_parameters = [
        #     {
        #         'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': 0.0
        #     }, {
        #         'params': [p for n, p in task_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in task_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': 0.0
        #     }
        # ]
        # optimizer = AdamW(grouped_parameters, lr=self.config['task_learning_rate'], eps=self.config['adam_eps'])
        # return optimizer

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step):
        if step < 30000:
            return  # Debug
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)


if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model()

    runner.train(model)
