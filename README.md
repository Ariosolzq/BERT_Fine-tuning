BERT Consumer Complaint Classification

## 🚀 Introduction

This project trains a BERT-based model to classify consumer complaints into different product categories. The dataset consists of consumer complaints and their corresponding product labels.

## 📂 Project Structure

├── consumer_complaints.csv   # Dataset (Consumer complaints & product labels)
├── model.py                  # BERT Classifier model definition
├── dataset.py                # Custom Dataset class
├── train.py                  # Training script
├── evaluate.py               # Model evaluation script
├── requirements.txt          # Required dependencies
└── README.md                 # This documentation

## 🔨 Installation

1️⃣ Set Up Virtual Environment

python -m venv env
source env/bin/activate  # On Linux/Mac
env\Scripts\activate     # On Windows

2️⃣ Install Dependencies

pip install -r requirements.txt

## 📁 Dataset

The dataset is a CSV file (consumer_complaints.csv), containing:


consumer_complaint_narrative


product (label)

## 🎨 Model Architecture

The model is a BERT-based text classifier:

Uses BERT-base-uncased for feature extraction

Extracts the [CLS] token for sentence representation

Adds a fully connected classification layer

Uses Dropout to prevent overfitting

## 🔄 How to Run

1️⃣ Train the Model

Run the following command to start training:

python train.py

Training Parameters:

Batch size: 16

Epochs: 3

Learning rate: 2e-5

Optimizer: Adam

Loss Function: CrossEntropyLoss

2️⃣ Evaluate the Model

To evaluate accuracy and AUC, run:

python evaluate.py

This computes:

Accuracy

AUC (One-vs-Rest Multi-class classification)

## 🌟 Training Process

✅ Load Dataset & Preprocess

Remove null values

Encode labels

Split into train (70%) / test (30%)

✅ Tokenization

Use BERT Tokenizer (bert-base-uncased)

Convert text into tokenized format

Pad/truncate to max length (default: 128)

✅ Create Dataset & DataLoader

Use ComplaintDataset for structured data

DataLoader loads batches efficiently

✅ Train the BERT Model

Use train_epoch()

Apply Gradient Scaling (autocast) for efficiency

Update optimizer & scheduler

✅ Evaluate the Model

Compute Validation Accuracy & Loss

Calculate AUC Score using roc_auc_score()

## 📊 Model Evaluation

Metric

Description

Accuracy

Measures correct predictions

Loss

Cross-entropy loss for multi-class classification

AUC

Multi-class AUC (One-vs-Rest)

Example Output:

Epoch 1/3
Train loss 0.42  accuracy 87.5%
Val loss 0.40    accuracy 88.3%
AUC: 0.91

## 🚀 Future Improvements

✅ Data Augmentation: Synonyms replacement, back-translation✅ Hyperparameter Tuning: Learning rate, batch size optimization✅ More Advanced Models: Exploring BERT-large, RoBERTa, or DistilBERT

## 📃 References

BERT Paper: https://arxiv.org/abs/1810.04805

Hugging Face Transformers: https://huggingface.co/docs/transformers/index

Scikit-learn AUC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

## 👨‍💻 Author

## 👨‍💻 Ziqi Lin 📧 ziqi1229@umd.edu
