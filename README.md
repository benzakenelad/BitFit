# BitFit
Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models

# Abstract
We show that with small-to-medium training data, fine-tuning only the bias terms (or a subset of the bias terms) of pre-trained BERT models is competitive with (and sometimes better than) fine-tuning the entire model. For larger data, bias-only fine-tuning is competitive with other sparse fine-tuning methods.
Besides their practical utility, these findings are relevant for the question of understanding the commonly-used process of finetuning: they support the hypothesis that finetuning is mainly about exposing knowledge induced by language-modeling training, rather than learning new task-specific linguistic knowledge. 

# Environment 
First, create an environment with all the dependencies:
```
$ conda env create -n bitfit_env -f environment.yml
```
Then activate it:
```
$ conda activate bitfit_env
```

# Evaluating [GLUE Benchmark](https://arxiv.org/abs/1804.07461) with BitFit

```
python run_glue.py --task-name <task_name>\
       --output-path <output_path>\
       --model-name <model_name>\
       --bias-terms <bias_terms>\
       --gpu-device <gpu_device>\
       --learning-rate <learning_rate>\
       --epochs <epochs>\
       --batch-size <batch_size>
```
For further information about the arguments please run:
```
python run_glue.py -h
```
