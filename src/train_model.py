import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pandas as pd
import os
from utils import load_phishing_data

def main():
    # Load data
    texts, labels = load_phishing_data('./data/phishing_samples.csv')

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    # Prepare dataset for Huggingface
    dataset = Dataset.from_dict({'text': texts, 'label': encoded_labels})

    # Tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def preprocess(examples):
        return tokenizer(examples['text'], padding=True, truncation=True)

    tokenized_dataset = dataset.map(preprocess, batched=True)

    # Train/test split
    train_test = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./models/fine_tuned_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=2,
    )

    # Metrics function
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save the model and label encoder
    trainer.save_model('./models/fine_tuned_model')
    import joblib
    joblib.dump(label_encoder, './models/fine_tuned_model/label_encoder.joblib')

if __name__ == "__main__":
    main()
