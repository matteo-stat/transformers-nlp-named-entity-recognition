
# transformers-nlp-ner-token-classification

Welcome to the **transformers-nlp-ner-token-classification** repository! 🎉

This repo is all about fine-tuning HuggingFace Transformers for token classification (NER - Named Entity Recognition), setting up pipelines, and optimizing models for faster inference. It comes from my experience developing a custom chatbot, where multiple entities could be found in users messages.

I hope these scripts help you fine-tune and deploy your models with ease!

## Repository Structure

Here’s a quick rundown of what you’ll find in this repo:

- **`checkpoints/ner-token-classification/`**: This is where your model checkpoints will be stored during training. Save your progress and pick up where you left off!

- **`data/ner-token-classification/`**: Contains sample data for training, validation, and testing. These samples are here to demonstrate the expected format for token classification problems. Note that entities in samples are anonymized.

- **`models/ner-token-classification/`**: This is where the fine-tuned and optimized models will be saved. After fine-tuning and optimizing, you'll find your models here, ready for action!

## Scripts

Here's what each script in the repo does:

1. **`01-ner-token-classification-train.py`**  
   Fine-tunes a HuggingFace model on a token classification problem. If you're looking to train your model, this script is your starting point.

2. **`02-ner-token-classification-pipeline.py`**  
   Builds a pipeline for running inference with your fine-tuned model. This script allows you to run inference on single or multiple samples effortlessly.

3. **`03-ner-token-classification-optimize-model-for-inference.py`**  
   Optimizes your model for faster inference on CPU using ONNX Runtime. Perfect for when you're working on a development server with limited GPU memory.

4. **`04-ner-token-classification-pipeline-inference-optmized-model.py`**  
   Similar to the `02` script, but specifically for inference with the optimized model (using ONNX Runtime). Get faster predictions using a CPU!


## Requirements and Installation Warnings

Before you dive into the scripts, here are a few important notes about the dependencies and installation process:

### Dependency Files

   - **`requirements-with-inference-optimization.txt`**  
     Includes dependencies for scripts `01-ner-token-classification-train.py` and `02-ner-token-classification-pipeline.py` (excludes ONNX Runtime dependencies).

   - **`requirements-without-inference-optimization.txt`**  
     Includes dependencies for all scripts, including ONNX Runtime dependencies for optimization and inference.

### Note for PyTorch and NVIDIA GPUs

   If you are using PyTorch with an NVIDIA GPU, it's crucial to ensure you have the correct version of PyTorch installed. Before running the requirements installation, you should install the specific version of PyTorch compatible with your CUDA version (cuda 12.1 in the example below):

   ```bash
   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
