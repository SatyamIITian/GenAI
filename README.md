# GenAI

Fine-Tuning DistilBERT for Sentiment Analysis on SST-2
This repository contains the code and artifacts for a Generative AI mini-project, fine-tuning the DistilBERT model on the SST-2 dataset for binary sentiment classification (positive/negative). The project was developed for the Generative AI – Mini Project course, with presentation and submission scheduled for April 23–24, 2025.
Project Overview

Objective: Fine-tune DistilBertForSequenceClassification to achieve >0.90 accuracy on the SST-2 dataset for sentiment analysis.
Dataset: Stanford Sentiment Treebank (SST-2)
Training: 67,349 samples
Validation/Test: 872 samples


Model: Pretrained distilbert-base-uncased, customized with a binary classification head.
Environment: Google Colab with T4 GPU, using Hugging Face transformers and PyTorch.
Key Features:
Training with 3 epochs, batch size 32, early stopping (patience=3).
Evaluation metrics: accuracy, precision, recall, F1 score.
Visualizations: Loss and metrics (accuracy/F1) plots.
Failure analysis for misclassified or low-confidence predictions.


Expected Accuracy: >0.90 (e.g., ~0.92 based on test runs).

Repository Structure
├── notebooks/
│   └── sentiment_analysis.ipynb  # Colab notebook with step-by-step code
├── plots/
│   ├── loss_curves.png          # Training/validation loss plot
│   └── metrics_curves.png       # Validation accuracy/F1 plot
├── fine_tuned_distilbert/       # Saved model and tokenizer
├── README.md                    # This file
└── presentation.pptx            # Presentation slides (optional)

Prerequisites

Environment: Google Colab with T4 GPU (recommended) or local machine with GPU.
Dependencies:
Python 3.8+
Libraries: transformers, datasets, torch, scikit-learn, matplotlib, numpy, tqdm


Hardware: GPU for efficient training (~20–30 minutes on T4 GPU).

Setup

Clone the Repository:
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]


Install Dependencies:In Colab or local environment, run:
pip install transformers datasets torch scikit-learn matplotlib numpy tqdm


Prepare Colab (if using):

Open notebooks/sentiment_analysis.ipynb in Google Colab.
Set runtime to T4 GPU: Runtime > Change runtime type > T4 GPU.
Restart runtime before running: Runtime > Restart runtime.



Usage

Run the Notebook:

Execute sentiment_analysis.ipynb cell by cell.
Steps include:
Install dependencies.
Load DistilBERT model and SST-2 dataset.
Preprocess data (tokenization, max length 32).
Train model (3 epochs, early stopping).
Evaluate on test set.
Analyze failure cases.
Generate and save plots.
Save fine-tuned model.




Expected Outputs:

Metrics: Test accuracy >0.90, plus precision, recall, F1 (see notebooks/sentiment_analysis.ipynb, Step 7).
Plots: Saved in plots/ as loss_curves.png and metrics_curves.png.
Model: Saved in fine_tuned_distilbert/.
Failure Analysis: Printed in notebook (Step 8).


Demo:

Run the following code in Colab to predict sentiment on custom text:from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained('./fine_tuned_distilbert')
model = DistilBertForSequenceClassification.from_pretrained('./fine_tuned_distilbert')
model.eval()

text = "This movie is great!"
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=32)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax(-1).item()
    confidence = probs[0][pred].item()
print(f"Text: {text}")
print(f"Predicted: {'Positive' if pred == 1 else 'Negative'}, Confidence: {confidence:.3f}")


Example output:Text: This movie is great!
Predicted: Positive, Confidence: 0.950





Results

Test Metrics (example, replace with actual from notebook):| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.92  |
| Precision | 0.91  |
| Recall    | 0.93  |
| F1 Score  | 0.92  |


Plots:
loss_curves.png: Shows decreasing training/validation loss.
metrics_curves.png: Shows increasing validation accuracy/F1, stabilizing at ~0.92.


Failure Analysis:
Identifies misclassified or low-confidence (<0.6) predictions, highlighting ambiguous sentences (e.g., containing ‘not’ or ‘but’).



Artifacts

Notebook: notebooks/sentiment_analysis.ipynb (full code).
Plots: plots/loss_curves.png, plots/metrics_curves.png.
Model: fine_tuned_distilbert/ (model weights, tokenizer).
Presentation: presentation.pptx (optional, includes slides for April 2025 submission).
Failure Analysis: Screenshots or text output from notebook Step 8.

Troubleshooting

Memory Issues:
If Colab crashes, reduce per_device_train_batch_size to 16 in notebook Step 5.
Monitor GPU: !nvidia-smi.


Missing Plots:
Check plots/ with !ls plots/.
Verify train_loss, eval_accuracy logs in Step 9.


Demo Fails:
Ensure fine_tuned_distilbert/ exists (rerun Step 10).
Use screenshot as backup.


Slow Training:
Confirm fp16=True in Step 5.
Training takes ~20–30 minutes on T4 GPU.



References

Paper:
Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv:1810.04805. https://arxiv.org/abs/1810.04805.


Dataset:
SST-2: https://huggingface.co/datasets/sst2 (GLUE benchmark, public license).


Tools:
Hugging Face transformers: https://huggingface.co/docs/transformers.
PyTorch: https://pytorch.org.


Notebook: Based on Google Colab implementation.

License

This project is licensed under the MIT License. See the LICENSE file for details (create if needed).
Contact
For questions, contact SATYAM KUMAR at satyamkumarpathak01@gmail.com or open an issue on GitHub.

Submitted for Generative AI – Mini Project, April 2025.
