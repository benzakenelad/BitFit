import argparse
import os
import logging
from utils import setup_logging
from GLUEvaluator import GLUEvaluator, set_seed

setup_logging()
LOGGER = logging.getLogger(__file__)

PADDING = "max_length"
MAX_SEQUENCE_LEN = 128


def parse_args():
    parser = argparse.ArgumentParser(description='BitFit GLUE evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task-name', '-t', required=True, type=str, help='GLUE task name for evaluation.',
                        choices={'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'})
    parser.add_argument('--output-path', '-o', required=True, type=str,
                        help='output directory path for evaluation products.')
    parser.add_argument('--model-name', '-m', type=str, default='bert-base-cased', help='model-name to evaluate with.',
                        choices={'bert-base-cased', 'bert-large-cased', 'roberta-base'})
    parser.add_argument('--full-ft', default=False, action='store_true', help='if mentioned will perform full '
                                                                              '(standard) fine-tuning.')
    parser.add_argument('--bias-terms', metavar='N', type=str, nargs='+', default=['all'],
                        choices={'intermediate', 'key', 'query', 'value', 'output', 'output_layernorm',
                                 'attention_layernorm', 'all'},
                        help='bias terms to BitFit, choose \'all\' for BitFit all bias terms.')
    parser.add_argument('--gpu-device', '-d', type=int, default=None,
                        help='GPU id for BitFit, if not mentioned will train on CPU.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed value to set.')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-3, help='learning rate for training.')
    parser.add_argument('--epochs', '-e', type=int, default=16, help='number of training epochs.')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='training and evaluation batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices={'adam', 'adamw'})
    parser.add_argument('--save-evaluator', action='store_true', default=False,
                        help='if given, will save the evaluator for later inference/examination.')
    parser.add_argument('--predict-test', action='store_true', default=False,
                        help='if given, will infer on test set using the fine-tuned model (predictions file will be in '
                             'GLUE benchmark test server format). Predictions will be saved to output_path.')

    return parser.parse_args()


def validate_args(args):
    if not os.path.exists(args.output_path):
        raise ValueError("--output_path directory doesn't exist.")
    if not os.path.isdir(args.output_path):
        raise ValueError("--output_path must be a path to directory")
    if len(os.listdir(args.output_path)):
        raise ValueError("--output_path directory isn't empty, please supply an empty directory path.")


def plot_training_details(args):
    [LOGGER.info('############################################################################################') for _
     in range(3)]
    LOGGER.info('')

    LOGGER.info('Training Details: ')
    LOGGER.info('----------------------------------------------')
    LOGGER.info(f'Model Name: {args.model_name}')
    LOGGER.info(f'Task Name: {args.task_name}')
    LOGGER.info(f'Output Directory: {args.output_path}')

    if args.gpu_device is not None:
        LOGGER.info(f'Running on GPU #{args.gpu_device}')
    else:
        LOGGER.info(f'Running on CPU')

    if args.full_ft:
        LOGGER.info('Performing full (standard) fine-tuning')
    else:
        LOGGER.info(f"Bias Trainable Terms: {'all bias terms' if 'all' in args.bias_terms else args.bias_terms}")

    LOGGER.info(f'Epochs: {args.epochs}')
    LOGGER.info(f'Learning Rate: {args.learning_rate}')
    LOGGER.info(f'Batch Size: {args.batch_size}')
    LOGGER.info(f"Optimizer: {'AdamW' if args.optimizer == 'adamw' else 'Adam'}")

    LOGGER.info('')
    [LOGGER.info('############################################################################################') for _
     in range(3)]


def main():
    # args parsing
    args = parse_args()
    validate_args(args)
    plot_training_details(args)

    # evaluator creation
    evaluator = GLUEvaluator(args.task_name, args.model_name, args.gpu_device)

    # data preprocessing
    trainable_components = GLUEvaluator.convert_to_actual_components(args.bias_terms)
    evaluator.preprocess_dataset(PADDING, MAX_SEQUENCE_LEN, args.batch_size)

    # training preparation
    evaluator.training_preparation(args.learning_rate, args.full_ft, trainable_components, args.optimizer, verbose=True)

    # train
    evaluator.train_and_evaluate(args.epochs, args.output_path, seed=args.seed)

    # artifacts
    if not args.full_ft:
        evaluator.plot_terms_changes(os.path.join(args.output_path, 'bias_term_changes'))

    # save model
    if args.save_evaluator:
        evaluator.save(os.path.join(args.output_path, 'evaluator'))

    # export model test set predictions
    if args.predict_test:
        evaluator.export_model_test_set_predictions(args.output_path)


if __name__ == '__main__':
    main()
