

# LLM-Augmented Training for Video Localization on Ego4D



This project enhances the VSLNet baseline for the **Natural Language Querying (NLQ)** benchmark on the Ego4D dataset. The core innovation is a two-stage training pipeline that leverages a Large Language Model (**Google's Gemma**) to generate a high-quality synthetic dataset for pre-training, aimed at improving the model's ability to localize temporal moments in long, egocentric videos.

## Overview

The Ego4D Natural Language Querying (NLQ) task challenges a model to identify the precise start and end times of an event in a video that answers a natural language question. While deep learning models like VSLNet provide a strong baseline, their performance is inherently limited by the size and diversity of the official training data.

This project addresses this limitation by introducing a **pre-training stage** on a novel, synthetically generated dataset. By using an LLM to create contextual question-answer pairs from unused video narrations, we expose the model to a wider range of language and scenarios before fine-tuning it on the official Ego4D data.

## Methodology

The project is built around three key components:

1.  **Data Curation**: We identify and extract thousands of timestamped narrations from Ego4D videos that are not part of the official NLQ training, validation, or test splits. This ensures our synthetic data provides novel information to the model.

2.  **Synthetic Query Generation with Gemma**: We use **Google's Gemma 2B**, a powerful and efficient LLM, to generate questions.
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

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<YOUR_USERNAME>/LLM-Augmented-Training-Ego4D-NLQ.git
    cd LLM-Augmented-Training-Ego4D-NLQ
    ```

2.  **Install dependencies:**
    This project was developed in a Google Colab environment. The primary dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure your environment has PyTorch installed with CUDA support.*

3.  **Ego4D CLI Setup:**
    You need to sign the Ego4D license and obtain your access keys from the [official website](https://ego4ddataset.com/). Configure the CLI with your keys.

## How to Run

The entire process is detailed in the Jupyter Notebook `Ego4D_NLQ_LLM_Augmentation.ipynb`. Below are the key command-line steps.

### Step 1: Download Official Data

Use the Ego4D CLI to download the annotations and video features.

```bash
ego4d --output_directory="ego4d_data/" --datasets annotations omnivore_video_swinl_fp16 --benchmarks nlq -y --version v1
```

### Step 2: Generate or Download the Synthetic Dataset

You can either run the generation cells in the notebook, which requires a Hugging Face token and GPU resources, or use the pre-generated dataset provided in this repository.

*   **To Generate**: Follow the steps in the notebook under the section "Automatic Queries Generation using LLMs".
*   **To Use Pre-generated**: Download `nlq_synthetic_train.json` and place it in the root directory.

### Step 3: Prepare Datasets for VSLNet

The VSLNet codebase requires a pre-processing step to convert the JSON annotations and features into a format suitable for training.

```bash
# Prepare the SYNTHETIC dataset
python episodic-memory/NLQ/VSLNet/utils/prepare_ego4d_dataset.py \
    --input_train_split /path/to/your/nlq_synthetic_train.json \
    --video_feature_read_path ego4d_data/v1/omnivore_video_swinl_fp16 \
    --output_save_path data/dataset/nlq_pretrain_on_synthetic

# Prepare the OFFICIAL dataset
python episodic-memory/NLQ/VSLNet/utils/prepare_ego4d_dataset.py \
    --input_train_split ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split ego4d_data/v1/annotations/nlq_val.json \
    --video_feature_read_path ego4d_data/v1/omnivore_video_swinl_fp16 \
    --output_save_path data/dataset/nlq_official_v1
```

### Step 4: Pre-training on Synthetic Data

Run the main training script, pointing it to the prepared synthetic data.

```bash
python episodic-memory/NLQ/VSLNet/main.py \
    --task nlq_pretrain_on_synthetic \
    --mode train \
    --fv synthetic_official \
    --model_dir /path/to/save/pretrained_models/ \
    # ... (other VSLNet parameters)
```

### Step 5: Fine-tuning on Official Data

Run the training script again, but this time load the checkpoint from the pre-training step and use the official dataset.

```bash
python episodic-memory/NLQ/VSLNet/main.py \
    --task nlq_official_v1 \
    --mode train \
    --fv official \
    --model_dir /path/to/save/finetuned_models/ \
    --checkpoint /path/to/your/pretrained_model.t7 \
    --eval_gt_json ego4d_data/v1/annotations/nlq_val.json \
    # ... (other VSLNet parameters)
```

## Results

This project successfully implements an end-to-end pipeline for augmenting video-language datasets. The effectiveness of the pre-training can be measured by observing the convergence speed and final validation metrics (e.g., R@1, IoU=0.5) during the fine-tuning stage. Training logs and TensorBoard events are generated to monitor performance. The primary outcome is the robust framework for synthetic data generation, which is modular and can be adapted for other models or datasets.


