# Kaggle Competition: Mining Misconceptions in Math

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg)
![Lightning](https://img.shields.io/badge/Lightning-2.0-792EE5)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-0.41-orange.svg)
![Flash Attention](https://img.shields.io/badge/Flash_Attention-2.0-green.svg)
![PyTest](https://img.shields.io/badge/PyTest-7.4-blue.svg)
![C](https://img.shields.io/badge/C-17-A8B9CC.svg)

This repository is dedicated to the Kaggle competition [Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics).

## Table of Contents
- [Kaggle Competition: Mining Misconceptions in Math](#kaggle-competition-mining-misconceptions-in-math)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)

## Overview

In this competition, we have to develop an NLP model driven by ML to accurately predict the affinity between misconceptions and incorrect answers (distractors) in multiple-choice questions. This solution will suggest candidate misconceptions for distractors, making it easier for expert human teachers to tag distractors with misconceptions.

* **ML Problem Formulation**: Framed as a recommendation or ranking problem.
    - Ranking allows the model to generalize to unseen misconceptions.
    - Classification infeasible as most misconceptions have no examples and there are too many of them.
* **Input**:
    - **Question Details**: Subject, construct, question, and incorrect answer.
    - **Misconception**: A statement that describes a common misconception.
* **Output**:
    - A list of 25 misconceptions sorted by their relevance to the question.
* **Dataset**: 
    - **Competition**: EDA of competition dataset can be found [here](notebooks/eda.ipynb).
      - 2500+ misconceptions
      - 1800~ questions
    - **Contrastive**: A contrastive dataset is created to train the recall model. Further details can be found [here](docs/contrastive_dataset.md).
* **Model**: Recommendations will be served in two stages:
    - **Recall**: Rank a large pool of misconceptions based on their relevance to the question. Utilize a relatively smaller LLM (sub-10B parameters).
    - **Rerank**: Re-rank the candidate misconceptions based on their relevance to the incorrect answer. Utilize a larger LLM (10B+ parameters).
* **Loss Function**: Contrastive loss defined as follows:
    $$ \text{Loss} = - \frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{\exp(s(q, p_i))}{\exp(s(q, p_i)) + \sum_{j=1}^{n} \exp(s(q, p_j))} \right) $$
    where $s(q, p)$ is the similarity between the query and positive example, and $s(q, p_j)$ is the similarity between the query and negative examples.
* **Metrics**: Mean Average Precision (MAP@25)
    $$ \text{MAP@25} = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{min(n,25)} P(k) \times rel(k) $$
    where $P(k)$ is the precision at cut-off $k$, and $rel(k)$ is 1 if the correct answer is in the top $k$ results, and 0 otherwise.

## Acknowledgements

1. [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and [MTEB Paper](https://arxiv.org/pdf/2210.07316)
2. [NV-Embed by NVidia](https://arxiv.org/pdf/2405.17428)
3. [BGE by BAAI](https://arxiv.org/pdf/2409.15700)
4. [Sayulala's insightful thread](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
