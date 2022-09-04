# BitFit [(Paper)](https://arxiv.org/abs/2106.10199)
Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models

# Abstract
We introduce BitFit, a sparse-finetuning method where only the bias-terms of the model (or a subset of them) are being modified. We show that with small-to-medium training data, applying BitFit on pre-trained BERT models is competitive with (and sometimes better than) fine-tuning the entire model. For larger data, the method is competitive with other sparse fine-tuning methods.
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

# MIT License

Copyright (c) 2022 benzakenelad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

