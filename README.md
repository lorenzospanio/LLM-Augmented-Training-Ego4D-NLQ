

# LLM-Augmented Training for Video Localization on Ego4D



This project enhances the VSLNet baseline for the **Natural Language Querying (NLQ)** benchmark on the Ego4D dataset. The core innovation is a two-stage training pipeline that leverages a Large Language Model (**Google's Gemma**) to generate a high-quality synthetic dataset for pre-training, aimed at improving the model's ability to localize temporal moments in long, egocentric videos.

## Overview

The Ego4D Natural Language Querying (NLQ) task challenges a model to identify the precise start and end times of an event in a video that answers a natural language question. While deep learning models like VSLNet provide a strong baseline, their performance is inherently limited by the size and diversity of the official training data.

This project addresses this limitation by introducing a **pre-training stage** on a novel, synthetically generated dataset. By using an LLM to create contextual question-answer pairs from unused video narrations, we expose the model to a wider range of language and scenarios before fine-tuning it on the official Ego4D data.

All of the following steps are shown in more detail in the paper for this project, present in the repository.

## Methodology

The project is built around three key components:

1.  **Data Curation**: We identify and extract thousands of timestamped narrations from Ego4D videos that are not part of the official NLQ training, validation, or test splits. This ensures our synthetic data provides novel information to the model.

2.  **Synthetic Query Generation with Gemma**: We use **Google's Gemma 2B**, a small efficient LLM, to generate questions.
    *   **Contextual Prompting**: Narrations are grouped into small, sequential blocks. The LLM receives the entire block as context, enabling it to generate highly specific and context-aware questions for each individual narration. This is crucial for creating meaningful queries (e.g., asking "What is the person taking *from the fridge*?" instead of a generic "What is the person taking?").
    *   **Automated Pipeline**: The entire process, from feeding narration blocks to parsing the LLM's output into a structured JSON file, is fully automated.

3.  **Two-Stage Training Strategy**:
    *   **Pre-training**: The VSLNet model is first trained exclusively on our generated synthetic dataset. This allows the model to learn a general understanding of matching language to video moments without overfitting to the official data.
    *   **Fine-tuning**: The pre-trained model is then fine-tuned on the official `nlq_train.json` split, adapting its learned knowledge to the specific distribution and style of the benchmark data.

## Project Workflow

The end-to-end pipeline can be visualized as follows:

```
[1. Filter Unused Narrations] -> [2. Group into Contextual Blocks] -> [3. Prompt Engineering] -> [4. Gemma LLM Inference] -> [5. Format into nlq_train format] -> [6. VSLNet Pre-training] -> [7. VSLNet Fine-tuning]
```
    

## How to Run
The entire process is detailed in the Jupyter Notebook.
**Ego4D CLI Setup:** You need to sign the Ego4D license and obtain your access keys from the official website to onfigure the CLI with your keys.


## Results

This project successfully implements an end-to-end pipeline for augmenting video-language datasets. The primary outcome is the robust framework for synthetic data generation, which is modular and can be adapted for other models or datasets. The effectiveness of the pre-training is shown in the improved validation metrics after the fine-tuning stage.


