"""BitFit evaluation on GLUE Benchmark. Link to paper: https://arxiv.org/abs/2106.10199 """

import os
import re
import logging
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap
from functools import reduce
from torch.optim import Adam
from datasets import load_dataset
from transformers.optimization import AdamW
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets.arrow_dataset import Dataset
from utils import setup_logging

setup_logging()
LOGGER = logging.getLogger(__file__)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASK_TO_METRICS = {
    "cola": ["MCC"],
    "mnli": ["Accuracy"],
    "mrpc": ["Accuracy", "F1"],
    "qnli": ["Accuracy"],
    "qqp": ["Accuracy", "F1"],
    "rte": ["Accuracy"],
    "sst2": ["Accuracy"],
    "stsb": ["Spearman", "Pearson"],
    "wnli": ["Accuracy"],
}

METRIC_NAME_TO_FUNCTION = {
    "MCC": matthews_corrcoef,
    "Accuracy": accuracy_score,
    "F1": f1_score,
    "Spearman": spearmanr,
    "Pearson": pearsonr,
}

BIAS_TERMS_DICT = {
    'intermediate': 'intermediate.dense.bias',
    'key': 'attention.self.key.bias',
    'query': 'attention.self.query.bias',
    'value': 'attention.self.value.bias',
    'output': 'output.dense.bias',
    'output_layernorm': 'output.LayerNorm.bias',
    'attention_layernorm': 'attention.output.LayerNorm.bias',
    'all': 'bias',
}

TASK_NAME_TO_SUBMISSION_FILE_NAME = {
    "cola": "CoLA.tsv",
    "mnli": ("MNLI-m.tsv", "MNLI-mm.tsv"),
    "mrpc": "MRPC.tsv",
    "qnli": "QNLI.tsv",
    "qqp": "QQP.tsv",
    "rte": "RTE.tsv",
    "sst2": "SST-2.tsv",
    "stsb": "STS-B.tsv",
    "wnli": "WNLI.tsv",
}

TASK_IS_BINARY = {
    "cola": True,
    "mnli": False,
    "mrpc": True,
    "qnli": False,
    "qqp": True,
    "rte": False,
    "sst2": True,
    "stsb": True,
    "wnli": True,
}

BIAS_LAYER_NAME_TO_LATEX = {
    'attention.self.query.bias': '$\mathbf{b}_{q}^{\ell}$',
    'attention.self.key.bias': '$\mathbf{b}_{k}^{\ell}$',
    'attention.self.value.bias': '$\mathbf{b}_{v}^{\ell}$',
    'attention.output.dense.bias': '$\mathbf{b}_{m_1}^{\ell}$',
    'attention.output.LayerNorm.bias': '$\mathbf{b}_{LN_1}^{\ell}$',
    'intermediate.dense.bias': '$\mathbf{b}_{m_2}^{\ell}$',
    'output.dense.bias': '$\mathbf{b}_{m_3}^{\ell}$',
    'output.LayerNorm.bias': '$\mathbf{b}_{LN_2}^{\ell}$',
}


class GLUEvaluator:
    """This class contains the functionality for BitFit evaluation on GLUE benchmark.

    This class expose an API for evaluating BitFit on GLUE Benchmark (https://arxiv.org/abs/1804.07461).

    Attributes:
        task_name (str): task name, e.g. 'rte'.
        model_name (str): model name, e.g. 'bert-base-uncased'.
        device (str): GPU device to run on, if None will run on CPU.

    """

    def __init__(self, task_name, model_name, device):
        self.task_name = task_name
        self.model_name = model_name
        self.device = device

        # initialization
        self.is_regression = task_name == 'stsb'
        self.num_labels = None
        self.data_loaders = None
        self.batch_size = None
        self.model = None
        self.optimizer = None
        self.learning_rate = None
        self.evaluations = None
        self.encoder_trainable = None
        self.masks = None
        self.idx_to_label = None

    @staticmethod
    def convert_dataset_to_data_loader(dataset, model_name, batch_size, random_sampler, test=False):
        """Convert a Dataset to torch DataLoader.

        Args:
            dataset (datasets.arrow_dataset.Dataset): the dataset to convert to torch DataLoader.
            model_name (str): model name (e.g. bert-base-uncased).
            batch_size (int): batch size for training and evaluation.
            random_sampler (bool): if True, DataLoader will sample randomly else sequentially.
            test (bool): if True, dataset contains test samples.

        """
        if test:
            keys = ['input_ids', 'attention_mask', 'token_type_ids']
        else:
            keys = ['input_ids', 'attention_mask', 'token_type_ids', 'label']

        if 'roberta' in model_name:
            keys.remove('token_type_ids')

        data = {key: list() for key in keys}
        for sample in dataset:
            for key in keys:
                data[key].append(sample[key])

        for k, v in data.items():
            data[k] = torch.tensor(v)

        tensor_dataset = TensorDataset(*[data[key] for key in keys])
        data_sampler = RandomSampler(tensor_dataset) if random_sampler else SequentialSampler(tensor_dataset)
        return DataLoader(tensor_dataset, sampler=data_sampler, batch_size=batch_size)

    @staticmethod
    def convert_to_actual_components(components):
        return [BIAS_TERMS_DICT[component] for component in components]

    def preprocess_dataset(self, padding, max_sequence_len, batch_size, train_size=None):
        """Preprocess the train and validation datasets.

        Args:
            padding (str): padding method (currently 'max_length' is the suggested method)
            max_sequence_len (int): the maximum sequence length
            batch_size (int): training and evaluating batch size
            train_size (int): clip the train dataset size, if None will use all available samples

        """
        LOGGER.info(f'Downloading dataset: {self.task_name}')
        datasets = load_dataset('glue', self.task_name)

        self.batch_size = batch_size
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        is_regression = self.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            self.idx_to_label = {k: v for k, v in enumerate(datasets['train'].features['label'].__dict__['_int2str'])}
            self.num_labels = len(label_list)
        else:
            self.num_labels = 1

        sentence1_key, sentence2_key = TASK_TO_KEYS[self.task_name]

        def _preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                    examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_sequence_len, truncation=True)
            return result

        datasets = datasets.map(_preprocess_function, batched=True, load_from_cache_file=False)

        self.data_loaders = dict()

        if train_size:
            perm = np.random.permutation(len(datasets['train']))[:train_size]
            self.data_loaders['train'] = Dataset.from_dict(datasets['train'][perm])
        else:
            self.data_loaders['train'] = datasets['train']

        if self.task_name == 'mnli':
            self.data_loaders['validation_matched'] = datasets['validation_matched']
            self.data_loaders['validation_mismatched'] = datasets['validation_mismatched']
            self.data_loaders['test_matched'] = datasets['test_matched']
            self.data_loaders['test_mismatched'] = datasets['test_mismatched']
        else:
            self.data_loaders['validation'] = datasets['validation']
            self.data_loaders['test'] = datasets['test']

        for dataset_name, dataset in self.data_loaders.items():
            self.data_loaders[dataset_name] = self.convert_dataset_to_data_loader(dataset=dataset,
                                                                                  model_name=self.model_name,
                                                                                  batch_size=self.batch_size,
                                                                                  random_sampler=dataset_name == 'train',
                                                                                  test='test' in dataset_name)

    def _train(self, train_dataloader, epoch, max_grad_norm=1.0):
        # train the model
        self.model.train()
        trained_samples, loss_sum = 0, 0
        criteria = torch.nn.MSELoss() if self.is_regression else torch.nn.CrossEntropyLoss()
        n = len(train_dataloader.dataset)

        for step, batch in enumerate(train_dataloader):
            # move batch to gpu
            if self.device is not None:
                batch = tuple(obj.cuda(self.device) for obj in batch)

            if 'roberta' in self.model_name:
                input_ids, attention_mask, labels = batch
                token_type_ids = None
            else:
                input_ids, attention_mask, token_type_ids, labels = batch

            # forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            outputs = outputs.logits

            # loss calculation
            labels = labels.view(-1)
            outputs = outputs.view(-1) if self.is_regression else outputs.view(-1, self.num_labels)

            loss = criteria(outputs, labels)

            # backward pass
            loss.backward()

            # masking the gradients (if mask exists)
            if self.masks:
                if 'roberta' in self.model_name:
                    for name, param in self.model.roberta.named_parameters():
                        param.grad[~self.masks[name]] = 0
                else:
                    for name, param in self.model.bert.named_parameters():
                        param.grad[~self.masks[name]] = 0

            # track train loss
            loss_sum += loss.item()
            trained_samples += len(labels)

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)

            # update parameters
            self.optimizer.step()
            self.model.zero_grad()

            print(f'EPOCH: {epoch}   TRAIN: {trained_samples}/{n}   LOSS: {round(loss_sum / (step + 1), 3)}\r', end='')
        print('')

    def _evaluate(self, dataloader, dataloader_type):
        # evaluate model on validation set
        self.model.eval()
        evaluated_samples, accuracy_sum = 0, 0
        all_preds, all_labels = [], []

        for step, batch in enumerate(dataloader):
            # move batch to gpu
            if self.device is not None:
                batch = tuple(obj.cuda(self.device) for obj in batch)
            if 'roberta' in self.model_name:
                input_ids, attention_mask, labels = batch
                token_type_ids = None
            else:
                input_ids, attention_mask, token_type_ids, labels = batch

            # forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                outputs = outputs.logits

            # reshaping
            labels = labels.view(-1)
            outputs = outputs.view(-1) if self.is_regression else outputs.view(-1, self.num_labels)

            outputs = outputs.detach().cpu().numpy()
            labels = labels.cpu().numpy()

            evaluated_samples += len(labels)

            if not self.is_regression:
                outputs = np.argmax(outputs, axis=1)
                # accuracy calculation
                accuracy_sum += accuracy_score(labels, outputs) * len(labels)
                print(f'{dataloader_type} ACC: {round(accuracy_sum / evaluated_samples, 5)}\r', end='')

            all_preds.extend(list(outputs))
            all_labels.extend(list(labels))

        print('')
        results = {}
        for metric_name in TASK_TO_METRICS[self.task_name]:
            metric = METRIC_NAME_TO_FUNCTION[metric_name]
            result = metric(all_labels, all_preds)
            result = result[0] if self.is_regression else result
            results[metric_name] = result

        return results

    def _deactivate_relevant_gradients(self, trainable_components):
        # turns off the model parameters requires_grad except the trainable bias terms.
        for param in self.model.parameters():
            param.requires_grad = False
        if trainable_components:
            trainable_components = trainable_components + ['pooler.dense.bias']
        trainable_components = trainable_components + ['classifier']
        for name, param in self.model.named_parameters():
            for component in trainable_components:
                if component in name:
                    param.requires_grad = True
                    break

    def training_preparation(self, learning_rate, optimizer, encoder_trainable, trainable_components=None,
                             verbose=True):
        """Performs training preparation.

        Perform training preparation including: model initialization, optimizer initialization, relevant
        gradients deactivation and plotting a list of all trainable params (if verbose is True).

        Args:
            learning_rate (float): learning_rate to train with.
            optimizer(str): optimizer to perform the training with, currently adam and adamw are supported.
            encoder_trainable (bool): if True will perform a Full-FT else will perform BitFit training preparation.
            trainable_components(Union[List[str], None]): list of trainable component. (subset of `BIAS_TERMS_DICT` keys)
            verbose: if True will plot a list of all trainable params

        """
        if self.model:
            raise Exception('Training preparation was already completed.')

        if encoder_trainable and trainable_components:
            raise Exception(
                f"If encoder_trainable is True, you shouldn't supply trainable_components. "
                f"Got trainable_components: {trainable_components}")

        self.encoder_trainable = encoder_trainable
        # model declaration
        config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, return_dict=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
        if not encoder_trainable:
            self._deactivate_relevant_gradients(trainable_components)

        # optimizer declaration
        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=True)
        else:
            raise Exception(f"optimizer arg must be in ['adam', 'adamw'], got: {optimizer}")

        self.learning_rate = learning_rate

        if verbose:
            print('\n\nTrainable Components:\n----------------------------------------\n')
            total_trainable_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, '  --->  ', param.shape)
                    total_trainable_params += param.shape[0] if len(param.shape) == 1 else param.shape[0] * param.shape[
                        1]
            print(
                f'\n----------------------------------------\nNumber of Trainable Parameters: {total_trainable_params}\n')

        self.evaluations = {k: {metric_name: [] for metric_name in TASK_TO_METRICS[self.task_name]} for k in
                            self.data_loaders.keys()}

    def train_and_evaluate(self, num_epochs, output_path=None, evaluation_frequency=1):
        """Trains the encoder model and evaluate it on validation set.

        Learning curves will be saved to the output_path.

        Args:
            num_epochs (int): Number of epochs to perform.
            output_path (str): Directory path to save the learning curves too.
            evaluation_frequency (int): will evaluate every `evaluation_frequency` epochs.

        """

        if not self.data_loaders:
            raise Exception('data loaders were not initialized, please run "preprocess_dataset" before training.')

        if not self.model:
            raise Exception('model was not initialized, please run "training_preparation" before training.')

        if self.device is not None:
            self.model.cuda(self.device)

        for epoch in range(num_epochs):
            # Training
            self._train(self.data_loaders['train'], epoch)

            # Evaluation
            if not epoch % evaluation_frequency:
                for dataloader_type, dataloader in self.data_loaders.items():
                    if not ('test' in dataloader_type):
                        results = self._evaluate(dataloader, dataloader_type.upper())
                        for metric_name, result in results.items():
                            self.evaluations[dataloader_type][metric_name].append(result)
            print('')

            # Plotting
            self.plot_evaluations(output_path)

    def plot_evaluations(self, output_path=None):
        """Plot the learning curves for each metric.

        Args:
            output_path (str): Directory path to save the learning curves too, if None will print the figure.

        """
        for metric_name in TASK_TO_METRICS[self.task_name]:
            for dataloader_type, results_mapper in self.evaluations.items():
                if not ('test' in dataloader_type):
                    label = f'{dataloader_type} (max is {round(max(results_mapper[metric_name]) * 100, 2)})'
                    plt.plot(results_mapper[metric_name], label=label)
            plt.title(f'Learning Curves - {self.task_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend()
            if output_path:
                plt.savefig(os.path.join(output_path, f'learning_curves_{metric_name.lower()}'))
                plt.clf()
            else:
                plt.show()

    def plot_terms_changes(self, output_path=None):
        """Plot/save the terms changes (calculating explained below).

        We define the amount of change in a bias vector b to be (1/dim(b)) * |b_0 - b_f|_1 that is, the average
        absolute change, across its dimensions, between the initial LM values b_0 and its fine-tuned values b_f.

        Args:
            output_path (str): Directory path to save the terms changes heatmap too, if None will print the figure.

        """
        if self.encoder_trainable:
            raise ValueError('Can plot terms changes only when BitFit.')

        if output_path:
            LOGGER.info(f'Saving the BitFit bias terms changes to: {output_path}')

        if 'roberta' in self.model_name:
            base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, return_dict=True).roberta
            fine_tuned_model = self.model.cpu().roberta
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, return_dict=True).bert
            fine_tuned_model = self.model.cpu().bert

        num_layers = self.model.config.num_hidden_layers

        def _calc_mean_diff(ft_p, base_p):
            return np.mean(np.abs(np.array(ft_p.data - base_p.data)))

        changes = []
        for ft_name, ft_param in fine_tuned_model.named_parameters():
            if ft_param.requires_grad and 'layer' in ft_name:
                for base_name, base_param in base_model.named_parameters():
                    if ft_name == base_name:
                        changes.append({'name': ft_name, 'value': _calc_mean_diff(ft_param, base_param)})

        def _get_component_name(name):
            return re.split(r'.[0-9]+.', name)[1]

        def _get_component_layer(name):
            return int(name.split('.')[2])

        keys = list(set(_get_component_name(c['name']) for c in changes))
        keys_mapper = {k: i for i, k in enumerate(keys)}

        total_weights = np.zeros(len(keys))
        for change in changes:
            total_weights[keys_mapper[_get_component_name(change['name'])]] += change['value']

        keys = [keys[i] for i in np.argsort(-total_weights)]
        keys_mapper = {k: i for i, k in enumerate(keys)}

        avg_column = np.zeros(len(keys))
        values_map = np.zeros((len(keys), num_layers + 1))
        for change in changes:
            avg_column[keys_mapper[_get_component_name(change['name'])]] += change['value']
            values_map[keys_mapper[_get_component_name(change['name'])], _get_component_layer(change['name'])] = change[
                'value']
        avg_column /= num_layers
        values_map[:, -1] = avg_column

        fig, ax = plt.subplots(figsize=(num_layers, len(keys)))
        xticklabels = [f'layer {i + 1}' for i in range(num_layers)]
        xticklabels.append('Avg.')

        keys = [BIAS_LAYER_NAME_TO_LATEX[key] for key in keys]
        heatmap(values_map, cmap="Blues", ax=ax, yticklabels=keys, xticklabels=xticklabels)

        plt.xticks(rotation=45)
        plt.yticks(rotation=0, ha='left')

        # align the y-axis text to the left
        yax = ax.get_yaxis()
        pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)

        if output_path:
            plt.savefig(output_path)
            plt.clf()
        else:
            plt.show()

        if self.device is not None:
            self.model.cuda(self.device)

    def save(self, output_path):
        """Saves the evaluator to the output_path.

        Args:
            output_path (str): Directory to save to model to.

        """
        LOGGER.info(f'Saving the model to: {output_path}')

        self.model.cpu()
        data = {'model': self.model, 'model_name': self.model_name, 'task_name': self.task_name,
                'learning_rate': self.learning_rate, 'evaluations': self.evaluations,
                'batch_size': self.batch_size, 'num_labels': self.num_labels,
                'encoder_trainable': self.encoder_trainable}
        with open(output_path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load(path, gpu_device):
        """Loads the evaluator from `path`.

        Args:
            gpu_device (int): GPU device ID.
            path (str): Directory to load to model from.

        """
        with open(path, 'rb') as file:
            data = pickle.load(file)
        evaluator = GLUEvaluator(data['task_name'], data['model_name'], gpu_device)
        evaluator.num_labels = data['num_labels']
        evaluator.batch_size = data['batch_size']
        evaluator.model = data['model']
        evaluator.learning_rate = data['learning_rate']
        evaluator.evaluations = data['evaluations']
        evaluator.encoder_trainable = data.get('encoder_trainable', None)

        return evaluator

    def set_uniform_mask(self, mask_size):
        """Uniformly chooses `mask_size` parameters from the model and generates a boolean mask for every component.

        Uniformly sample `mask_size` parameters from the entire model parameters, and in fine-tuning process only them
        will be fine-tuned.

        Args:
            mask_size (int): number of non-masked parameters.

        """
        if not self.encoder_trainable:
            raise Exception('In order to train with a random mask the encoder must be trainable.')

        if 'roberta' in self.model_name:
            model = self.model.roberta
        else:
            model = self.model.bert

        total_params = 0
        self.masks, params_per_component = dict(), dict()
        for name, param in model.named_parameters():
            self.masks[name] = torch.zeros(param.size(), dtype=torch.bool)
            component_params = reduce(lambda x, y: x * y, param.shape)
            params_per_component[name] = component_params
            total_params += component_params

        tunable_params_per_component = {k: int((v * mask_size) / total_params) for k, v in
                                        params_per_component.items()}

        LOGGER.info(f'Non-Masked params amount: {reduce(lambda x, y: x + y, tunable_params_per_component.values())}. '
                    f'Total params: {total_params}')

        for name, param in model.named_parameters():
            component_mask_size = tunable_params_per_component[name]
            component_params = params_per_component[name]
            indices = np.random.randint(0, component_params, component_mask_size)
            mask = self.masks[name]
            for index in indices:
                if len(param.shape) == 1:
                    mask[index] = True
                else:
                    mask[int(index / param.shape[1]), index % param.shape[1]] = True

    def set_row_and_column_random_mask(self):
        """Initializes the mask by randomly choosing rows or a column from each weight

        Initializes the mask by randomly choosing rows or a column (column size is equal the bias size) from each
        weight, the amount of total non-masked parameters in each weight is equal to the matching bias param size.

        """
        if not self.encoder_trainable:
            raise Exception('In order to train with a random mask the encoder must be trainable.')

        if 'roberta' in self.model_name:
            model = self.model.roberta
        else:
            model = self.model.bert

        self.masks = dict()
        total_params = 0
        for name, param in model.named_parameters():
            self.masks[name] = torch.zeros(param.size(), dtype=torch.bool)
            total_params += reduce(lambda x, y: x * y, param.shape)

            if ('encoder' not in name and 'pooler' not in name) or 'weight' not in name:
                continue

            if len(param.shape) == 1 and 'LayerNorm' in name:  # in case it's a LayerNorm
                self.masks[name][:] = True
                continue

            if np.random.randint(0, 2) or param.shape[0] < param.shape[1]:  # we randomly choose a column
                n_columns = int(param.shape[1])
                column_index = np.random.randint(0, n_columns)
                self.masks[name][:, column_index] = True
            else:  # we randomly choose rows
                bias_shape = int(param.shape[0])
                row_size = int(param.shape[1])
                n_rows_to_activate = int(bias_shape / row_size)
                row_indices = np.random.randint(0, bias_shape, n_rows_to_activate)
                self.masks[name][row_indices] = True

        LOGGER.info(f'Non-Masked params amount: {int(np.sum([np.sum(mask.numpy()) for mask in self.masks.values()]))}. '
                    f'Total params: {total_params}')

    def export_model_test_set_predictions(self, output_path):
        """Infers on test set and saves the predictions to output_path (predictions are in GLUE test server format).

        Args:
            output_path (str): Directory to save the predictions.

        """
        if not self.data_loaders:
            raise Exception(
                'data loaders were not initialized, please run "preprocess_dataset" before test evaluation.')

        if not self.model:
            raise Exception('model was not initialized, please run "training_preparation" before test evaluation.')

        if self.device is not None:
            self.model.cuda(self.device)

        LOGGER.info(f'Exporting model test set predictions to: {output_path}.')

        test_data_loaders = dict()
        if self.task_name == 'mnli':
            test_data_loaders["MNLI-m.tsv"] = self.data_loaders["test_matched"]
            test_data_loaders["MNLI-mm.tsv"] = self.data_loaders["test_mismatched"]
        else:
            test_data_loaders[TASK_NAME_TO_SUBMISSION_FILE_NAME[self.task_name]] = self.data_loaders["test"]

        self.model.eval()

        for prediction_file_name, dataloader in test_data_loaders.items():
            results = list()
            counter = 0
            num_samples = len(dataloader.dataset)
            for batch in dataloader:
                if self.device is not None:
                    batch = tuple(obj.cuda(self.device) for obj in batch)
                input_ids, attention_mask, token_type_ids = batch

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                    outputs = outputs.logits

                if self.is_regression:
                    outputs = outputs.view(-1)
                    outputs = outputs.detach().cpu().numpy()
                    results.extend(list(outputs))
                else:
                    outputs = outputs.view(-1, self.num_labels)
                    outputs = outputs.detach().cpu().numpy()
                    outputs = np.argmax(outputs, axis=-1)
                    if TASK_IS_BINARY[self.task_name]:
                        results.extend([int(pred) for pred in outputs])
                    else:
                        results.extend([self.idx_to_label[pred] for pred in outputs])

                counter += len(outputs)
                print(f'Test inference progress: {counter}/{num_samples}\r', end='')
            print('')
            with open(os.path.join(output_path, prediction_file_name), 'w') as file:
                file.write('index\tprediction\n')
                for idx, result in enumerate(results):
                    file.write(f'{idx}\t{result}\n')

        LOGGER.info(f'Test evaluation is over, evaluation artifacts are: {list(test_data_loaders.keys())}')

    def freeze_classifier(self):
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = False
