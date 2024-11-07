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
