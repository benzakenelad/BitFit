# BitFit [(Paper)](https://arxiv.org/abs/2106.10199)
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

# [GLUE Benchmark](https://arxiv.org/abs/1804.07461) evaluation examples:

```
python run_glue.py 
       --output-path <output_path>\
       --task-name <task_name>\
       --model-name <model_name>\
       --fine-tune-type <fine_tune_type>\
       --bias-terms <bias_terms>\
       --gpu-device <gpu_device>\
       --learning-rate <learning_rate>\
       --epochs <epochs>\
       --batch-size <batch_size>\
       --optimizer <optimizer_name>\
       --save-evaluator\
       --predict-test\
       --verbose
```
For further information about the arguments run:
```
python run_glue.py -h
```

Example of executing full fine tuning:
```
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\  
       --model-name bert-base-cased\
       --fine-tune-type full_ft\
       --learning-rate 1e-5
```

Example of executing full BitFit (training all bias terms):
```
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\
       --model-name bert-base-cased\
       --fine-tune-type bitfit\
       --learning-rate 1e-3
```

Example of executing partial BitFit (training a subset of the bias terms):
```
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\
       --model-name bert-base-cased\
       --fine-tune-type bitfit\
       --bias-terms query intermediate\ 
       --learning-rate 1e-3
```

Example of executing "frozen" training (i.e. using the pre-trained transformer as a feature extractor):
```
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\
       --model-name bert-base-cased\
       --fine-tune-type frozen\
       --learning-rate 1e-3
```

Example of training uniformly chosen trainable parameters (similar to "rand_100k" row in Table 3 in BitFit paper)
```
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\
       --model-name bert-base-cased\
       --fine-tune-type rand_uniform\
       --learning-rate 1e-3
```

<!-- Example of training uniformly chosen rows/cols from weight matrices (similar to "rand_row_col" row in Table 3 in BitFit paper)
```
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\
       --model-name bert-base-cased\
       --fine-tune-type rand_uniform\
       --learning-rate 1e-3
``` -->
