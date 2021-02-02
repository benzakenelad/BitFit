import re
from functools import reduce

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from scipy.stats import spearmanr, pearsonr
from seaborn import heatmap
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from datasets import load_dataset

torch.manual_seed(0)
np.random.seed(0)


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
    'attention_output': 'attention.output.dense.bias',
    'output_layernorm': 'output.LayerNorm.bias',
    'attention_layernorm': 'attention.output.LayerNorm.bias',
    'bias': 'bias',
    'LayerNorm': 'LayerNorm',
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


class GLUEEvaluator():
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

    @staticmethod
    def convert_to_data_loader(dataset, model_name, batch_size, random, test=False):
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
        data_sampler = RandomSampler(tensor_dataset) if random else SequentialSampler(tensor_dataset)
        return DataLoader(tensor_dataset, sampler=data_sampler, batch_size=batch_size)

    @staticmethod
    def convert_to_actual_components(components):
        return [BIAS_TERMS_DICT[component] for component in components]

    def preprocess_dataset(self, test, padding, max_sequence_len, batch_size):
        print(f'Downloading dataset: {self.task_name}')
        datasets = load_dataset('glue', self.task_name)

        self.batch_size = batch_size
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        is_regression = self.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
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
        self.data_loaders['train'] = datasets["train"]
        if test:
            self.data_loaders['test'] = datasets["test_matched" if self.task_name == "mnli" else "test"]

        if self.task_name == "mnli":
            self.data_loaders['validation_matched'] = datasets["validation_matched"]
            self.data_loaders['validation_mismatched'] = datasets["validation_mismatched"]
        else:
            self.data_loaders['validation'] = datasets["validation"]

        for k, dataset in self.data_loaders.items():
            self.data_loaders[k] = self.convert_to_data_loader(dataset, self.model_name, self.batch_size, k == 'train')

    def _train(self, train_dataloader, epoch, max_grad_norm=1.0):
        self.model.train()
        trained_samples, loss_sum = 0, 0
        criteria = torch.nn.MSELoss() if self.is_regression else torch.nn.CrossEntropyLoss()
        n = len(train_dataloader.dataset)

        for step, batch in enumerate(train_dataloader):
            # move batch to gpu
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

            # masking the gradients
            if self.masks:
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
        print('\n')

    def _evaluate(self, dataloader, dataloader_type):
        self.model.eval()
        evaluated_samples, accuracy_sum = 0, 0
        all_preds, all_labels = [], []

        for step, batch in enumerate(dataloader):
            # move batch to gpu
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

            if step * self.batch_size > 40000:
                break

        results = {}
        for metric_name in TASK_TO_METRICS[self.task_name]:
            metric = METRIC_NAME_TO_FUNCTION[metric_name]
            result = metric(all_labels, all_preds)
            result = result[0] if self.is_regression else result
            results[metric_name] = result

        return results

    def _deactivate_relevant_gradients(self, encoder_trainable, trainable_components):
        if not encoder_trainable:
            for param in self.model.parameters():
                param.requires_grad = False
            assert len(trainable_components) > 0
            trainable_components = trainable_components + ['pooler.dense.bias', 'classifier']
            for name, param in self.model.named_parameters():
                for component in trainable_components:
                    if component in name:
                        param.requires_grad = True
                        break

    def training_preparation(self, learning_rate, encoder_trainable, trainable_components, optimizer, verbose=True):
        if self.model:
            raise Exception('Training preparation was already completed.')

        self.encoder_trainable = encoder_trainable
        # model declaration
        config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, return_dict=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config).cuda(
            self.device)
        self._deactivate_relevant_gradients(encoder_trainable, trainable_components)

        # optimizer declaration
        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=True)
        else:
            raise Exception(f"optimizer arg must be in ['adam', 'adamw'], got: {optimizer}")

        self.learning_rate = learning_rate

        if verbose:
            print('Trainable Components:\n----------------------------------------\n')
            total_trainable_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, '  --->  ', param.shape)
                    total_trainable_params += param.shape[0] if len(param.shape) == 1 else param.shape[0] * param.shape[
                        1]
            print(f'\n----------------------------------------\nTrainable Parameters: {total_trainable_params}')

        self.evaluations = {k: {metric_name: [] for metric_name in TASK_TO_METRICS[self.task_name]} for k in
                            self.data_loaders.keys()}

    def plot_evaluations(self):
        for metric_name in TASK_TO_METRICS[self.task_name]:
            for dataloader_type, results_mapper in self.evaluations.items():
                label = f'{dataloader_type}_{round(max(results_mapper[metric_name]) * 100, 2)}'
                plt.plot(results_mapper[metric_name], label=label)
            plt.title(metric_name)
            plt.legend()
            plt.show()

    def _plot_summary(self):
        print(f'Task: {self.task_name}\nModel: {self.model_name}\nETA: {self.learning_rate}\nDevice: {self.device}')

    def train_and_evaluate(self, num_epochs):
        if not self.data_loaders:
            raise Exception('data loaders does not exist, please run "preprocess_dataset" before training.')

        self.model.cuda(self.device)
        self._plot_summary()
        for epoch in range(num_epochs):
            # Training
            self._train(self.data_loaders['train'], epoch)

            # Evaluation
            for dataloader_type, dataloader in self.data_loaders.items():
                results = self._evaluate(dataloader, dataloader_type.upper())
                for metric_name, result in results.items():
                    self.evaluations[dataloader_type][metric_name].append(result)

            # Plotting
            self.plot_evaluations()

    def plot_terms_changes(self, save_to=None):
        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, return_dict=True)
        fine_tuned_model = self.model.cpu()
        num_layers = self.model.config.num_hidden_layers

        def _calc_mean_diff(ft_p, base_p):
            return np.mean(np.abs(np.array(ft_p.data - base_p.data)))

        changes = []
        if 'roberta' in self.model_name:  # roberta
            for ft_name, ft_param in fine_tuned_model.roberta.named_parameters():
                if ft_param.requires_grad and 'layer' in ft_name:
                    for base_name, base_param in base_model.roberta.named_parameters():
                        if ft_name == base_name:
                            changes.append({'name': ft_name, 'value': _calc_mean_diff(ft_param, base_param)})
        else:  # bert
            for ft_name, ft_param in fine_tuned_model.bert.named_parameters():
                if ft_param.requires_grad and 'layer' in ft_name:
                    for base_name, base_param in base_model.bert.named_parameters():
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
        heatmap(values_map, cmap="BuPu", ax=ax, linecolor='white', linewidth=0.2, yticklabels=keys,
                xticklabels=xticklabels)

        plt.xticks(rotation=45)
        plt.yticks(rotation=0, ha='left')

        # align the y-axis text to the left
        yax = ax.get_yaxis()
        pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)

        if save_to:
            plt.savefig(save_to)

    def save(self, path=None):
        fine_tuned = 'full_FT' if self.encoder_trainable else 'bitfit'

        if not path:
            path = f'{self.model_name}__{fine_tuned}__{self.task_name}__'
            for dataset_name, metrics in self.evaluations.items():
                path += f'{dataset_name}_'
                for metric_name, scores in metrics.items():
                    path += f'{metric_name}_{round(max(scores), 3)}_'
                path += '_'

        self.model.cpu()
        data = {'model': self.model, 'model_name': self.model_name, 'task_name': self.task_name,
                'learning_rate': self.learning_rate, 'evaluations': self.evaluations,
                'batch_size': self.batch_size, 'num_labels': self.num_labels,
                'encoder_trainable': self.encoder_trainable}
        with open(path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load(path, device):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        evaluator = GLUEEvaluator(data['task_name'], data['model_name'], device)
        evaluator.num_labels = data['num_labels']
        evaluator.batch_size = data['batch_size']
        evaluator.model = data['model'].cuda(device)
        evaluator.learning_rate = data['learning_rate']
        evaluator.evaluations = data['evaluations']
        evaluator.encoder_trainable = data.get('encoder_trainable', None)
        evaluator.device = device

        return evaluator

    def randomize_mask(self, mask_size=100000):
        if not self.encoder_trainable:
            raise Exception('In order to train with random mask the encoder must be trainable.')

        total_params = 0
        self.masks, params_per_component = dict(), dict()
        for name, param in self.model.bert.named_parameters():
            self.masks[name] = torch.zeros(param.size(), dtype=torch.bool)
            component_params = reduce(lambda x, y: x * y, param.shape)
            params_per_component[name] = component_params
            total_params += component_params

        tunable_params_per_component = {k: int((v * mask_size) / total_params) for k, v in
                                        params_per_component.items()}

        print(f'Mask size: {reduce(lambda x, y: x + y, tunable_params_per_component.values())}. '
              f'Total params: {total_params}')

        for name, param in self.model.bert.named_parameters():
            component_mask_size = tunable_params_per_component[name]
            component_params = params_per_component[name]
            indices = np.random.randint(0, component_params, component_mask_size)
            mask = self.masks[name]
            for index in indices:
                if len(param.shape) == 1:
                    mask[index] = True
                else:
                    mask[int(index / param.shape[1]), index % param.shape[1]] = True


def evaluate_test(model, model_name, task_name, device, old_version, batch_size=16, padding='max_length',
                  max_sequence_len=128):
    assert 'bert' in model_name and 'roberta' not in model_name
    print(f'Downloading dataset: {task_name}')
    datasets = load_dataset('glue', task_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        idx_to_label = {k: v for k, v in enumerate(datasets['train'].features['label'].__dict__['_int2str'])}
        num_labels = len(label_list)
    else:
        num_labels = 1

    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]

    def _preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_sequence_len, truncation=True)
        return result

    datasets = datasets.map(_preprocess_function, batched=True, load_from_cache_file=False)

    test_data_loaders = dict()
    if task_name == 'mnli':
        test_data_loaders["MNLI-m.tsv"] = datasets["test_matched"]
        test_data_loaders["MNLI-mm.tsv"] = datasets["test_mismatched"]
    else:
        test_data_loaders[TASK_NAME_TO_SUBMISSION_FILE_NAME[task_name]] = datasets["test"]

    for k, dataloader in test_data_loaders.items():
        test_data_loaders[k] = GLUEEvaluator.convert_to_data_loader(dataloader, model_name, batch_size, random=False,
                                                                    test=True)

    model.eval()

    for file_name, dataloader in test_data_loaders.items():
        results = []
        counter = 0
        num_samples = len(dataloader.dataset)
        for batch in dataloader:
            batch = tuple(obj.cuda(device) for obj in batch)
            input_ids, attention_mask, token_type_ids = batch

            with torch.no_grad():
                if old_version:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                    src_key_padding_mask=None)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    outputs = outputs.logits

            if is_regression:
                outputs = outputs.view(-1)
                outputs = outputs.detach().cpu().numpy()
                results.extend(list(outputs))
            else:
                outputs = outputs.view(-1, num_labels)
                outputs = outputs.detach().cpu().numpy()
                outputs = np.argmax(outputs, axis=-1)
                if TASK_IS_BINARY[task_name]:
                    results.extend([int(pred) for pred in outputs])
                else:
                    results.extend([idx_to_label[pred] for pred in outputs])

            counter += len(outputs)
            print(f'{counter}/{num_samples}\r', end='')

        with open(file_name, 'w') as file:
            file.write('index\tprediction\n')
            for idx, result in enumerate(results):
                file.write(f'{idx}\t{result}\n')
