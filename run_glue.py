import argparse
import os
from GLUEEvaluator import GLUEEvaluator

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
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-3, help='learning rate for training.')
    parser.add_argument('--epochs', '-e', type=int, default=16, help='number of training epochs.')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='training and evaluation batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices={'adam', 'adamw'})
    parser.add_argument('--save_evaluator', action='store_true', default=False,
                        help='if given, will save the evaluator for later inference/examination.')
    parser.add_argument('--predict_test', action='store_true', default=False,
                        help='if given, will infer on test set using the fine-tuned model (predictions file will be in '
                             'GLUE benchmark test server format). Predictions will be saved to output_path.')

    return parser.parse_args()


def validate_args(args):
    if not os.path.exists(args.output_path):
        raise ValueError('--output_path directory doesn\'t exist.')
    if len(os.listdir(args.output_path)):
        raise ValueError('--output_path directory isn\'t empty, please supply an empty directory path.')


def main():
    # args parsing
    args = parse_args()
    validate_args(args)

    # evaluator creation
    evaluator = GLUEEvaluator(args.task_name, args.model_name, args.gpu_device)

    # data preprocessing
    trainable_components = GLUEEvaluator.convert_to_actual_components(args.bias_terms)
    evaluator.preprocess_dataset(PADDING, MAX_SEQUENCE_LEN, args.batch_size)

    # training preparation
    evaluator.training_preparation(args.learning_rate, args.full_ft, trainable_components, args.optimizer, verbose=True)

    # train
    evaluator.train_and_evaluate(args.epochs, args.output_path)

    # artifacts
    if not args.full_ft:
        evaluator.plot_terms_changes(os.path.join(args.output_path, 'bias_term_changes'))

    # save model
    if args.save_evaluator:
        evaluator.save(args.output_path)

    # export model test set predictions
    if args.predict_test:
        evaluator.export_model_test_set_predictions(args.output_path)


if __name__ == '__main__':
    main()
