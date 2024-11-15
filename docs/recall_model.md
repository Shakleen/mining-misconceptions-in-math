# Recall Model

## Model Architecture

Composed of two main components:

- **Pretrained LLM**: We use a pretrained LLM as the backbone. Tested with 
    - Mistral-7B
    - LLaMa-3.1-8B
    - Qwen-2.5-7B.
- **Pooling Layer**: Used to pool together embedding vectors across sequence lengths. So far the following methods have been tested:
    - Mean Pooling
    - CLS Token
    - Last Token
    - Latent Attention Module

## Latent Attention Module

Implemented according to the following paper: [NV-Embed by NVidia](https://arxiv.org/pdf/2409.15700). The authors argue that attention-based pooling allows the model to learn important features from the latent space of the LLM hidden states.


## Version 1
* **Commit ID** : 4708cda0fe5657eeee60e376d75cfc71ccdb3614
* **Description**: Recall model with
    - QLoRA
    - Pooling mechanism
* **Class**: [RecallModel](../src/model_development/recall_model.py)


## Version 2 (Planned)
- Swap out the causal mask for a bidirectional mask
