# HOSSemEval-EB23-A-Robust-Dataset-for-Aspect-Based-Sentiment-Analysis-of-Hospitality-Reviews

This repo contains the annotated data and code for our paper: HOSSemEval-EB23-A-Robust-Dataset-for-Aspect-Based-Sentiment-Analysis-of-Hospitality-Reviews

## Short Summary
We aim to tackle the aspect-based sentiment analysis task by given our own annotation data which obtained from Booking.com, covering the period from 2020 to 2023. Given the sentence, we predict all sentiment quads (`aspect category`, `aspect term`, `sentiment polarity`)

## Data 
The data used in this study is not publicly available due to security concerns. If you are interested in using the data, please contact the authors at htson@hcmus.edu.vn

## Requirements
**Project Setup with Python 3.11.7 (Conda)**

This project simplifies setup, streamlining the creation of a Python 3.11.7 environment using Conda and the installation of necessary packages.  Follow these steps:

**1. Execute the Script**

Run the following command in your terminal: 
```bash
$ source setup.sh
```

The script will prompt you to confirm various actions during the setup process. Respond with 'y' (yes) to proceed with each step.

## Modeling
This paper introduces three primary models. These models are: 

### 1. TAS-Transformer Models ([Wang et al.,2020](https://knowledge-representation.org/j.z.pan/pub/WYDL+2020.pdf))
**1. Training**:
To train the model, execute the following commands in your terminal:

```bash
$ python TAS_BERT_joint.py \
--data_dir data/path_name \
--output_dir result/path_name \
--model_name 'bert-large-uncased' \
--vocab_file Vocab/vocab.txt \
--tokenize_method word_split \
--use_crf \
--eval_test \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 32 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--num_train_epochs 10 \
--pretrained True
```

**2. Inference**: 
To accurately gauge the model's performance and select the best training epoch, follow these steps:

```bash
$ python evaluation_for_TSD_ASD_TASD.py \
--output_dir results/path_name \
--num_epochs 10\
--tag_schema BIO
```

**3. Evaluation**: Show classification report and confusion matrix:

```bash
$ python evaluation_for_AD_SD.py \
--output_dir results/path_name \
--num_epochs 10\
--tag_schema BIO\
--best_epoch_file_ad_sd 'test_ep_{best_epoch}'
```

### 2. GAS ([Zhang et al.,2021](https://aclanthology.org/2021.acl-short.64.pdf))

**1. Training**:
To retrain the model with our data or train the model with your data annotated in our format, follow the syntax below:

```bash
$ python main.py --task tasd \
            --dataset `your_patth_store_data` \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20
```

**2. Inference**:
We are trained on our proprietary dataset, is now available on Hugging Face. You can access this model for further experimentation or fine-tuning using the following identifier: [kisejin/T5-GAS-v1](https://huggingface.co/kisejin/T5-GAS-v1)

### 3. ASQP  ([Zhang et al., 2021](https://arxiv.org/pdf/2110.00796.pdf))

**1. Training**:
To retrain the model with our data or train the model with your data annotated in our format, follow the syntax below:

```bash
$ python main.py --task tasd \
            --dataset `your_patth_store_data` \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20
```

**2. Inference**:
Similar to the approach outlined in GAS model, we have released our pretrained model for  public use. This model can be accessed and downloaded from Hugging Face with the following identifier: [kisejin/T5-ASQP-V3](https://huggingface.co/kisejin/T5-ASQP-V3)


## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
```
