# Mining Misconceptions in Math

This repository is dedicated to the Kaggle competition [Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics).

## Competition Description

In this competition, we have to develop an NLP model driven by ML to accurately predict the affinity between misconceptions and incorrect answers (distractors) in multiple-choice questions. This solution will suggest candidate misconceptions for distractors, making it easier for expert human teachers to tag distractors with misconceptions. Submissions are evaluated on the mean average precision (MAP) score:

$$ \text{MAP@25} = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{min(n,25)} P(k) \times rel(k) $$


## TODO Tree

- [X] Create contrastive dataset
- [X] Implement MAP@25 metric
- [X] Model development
- [ ] Training pipeline
  - [ ] QLoRA
  - [ ] K-fold Cross validation
  - [ ] Checkpointing
  - [ ] Logging to W&B
- [ ] Inference pipeline

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
