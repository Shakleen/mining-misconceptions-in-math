# Contrastive Dataset

Contrastive dataset is used for training recommender system models. A contrastive dataset contains the following columns:
1. **Query column**: The question to be answered
2. **Positive example**: The actual answer to the query
3. **Negative examples**: A list of multiple negative examples
4. **Label**: Index of the positive example

## Version 1
* **Commit ID** : 84d0cac87e7dd49d9aa402b7ebaa4ba35bcba038
* **Description**: Negative samples are randomly sampled.
* **Weights and Biases artifact**: contrastive-dataset:v0
* **Class**: [BaseDataset](../src/data_preparation/datasets/base_dataset.py)

## Version 2
* **Commit ID** : b2e143727cb0e010d731cd69fb98bd50318ae998
* **Description**: Changes from version 1:
    * Added sampler argument. Sampler is responsible for sampling negative samples. Sampler encapsulates the logic of sampling negative samples. 
    * The number of negative samples is dynamic. And it is determined on the fly by the sampler.
* **Weights and Biases artifact**: Uses qa-pair-dataset:v0
* **Class**: [BaseDatasetV2](../src/data_preparation/datasets/base_dataset_v2.py) and [RandomNegativeSampler](../src/data_preparation/negative_sampler/random_sampler.py)
