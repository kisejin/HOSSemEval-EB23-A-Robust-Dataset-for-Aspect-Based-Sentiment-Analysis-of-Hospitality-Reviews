# HOSSemEval-EB23-A-Robust-Dataset-for-Aspect-Based-Sentiment-Analysis-of-Hospitality-Reviews

This repo contains the annotated data and code for our paper: HOSSemEval-EB23-A-Robust-Dataset-for-Aspect-Based-Sentiment-Analysis-of-Hospitality-Reviews

## Short Summary
We aim to tackle the aspect-based sentiment analysis task by given our own annotation data which obtained from Booking.com, covering the period from 2020 to 2023. Given the sentence, we predict all sentiment quads (aspect category, aspect term, sentiment polarity)

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


### 2. GAS ([Zhang et al.,2021](https://aclanthology.org/2021.acl-short.64.pdf))


### 3. ASQP  ([Zhang et al., 2021](https://arxiv.org/pdf/2110.00796.pdf))
