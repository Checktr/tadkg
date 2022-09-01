import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import LongformerTokenizer, LongformerForSequenceClassification, EarlyStoppingCallback, \
    TrainingArguments, Trainer

from src import config
from src.models.helper.dataset import Dataset

"""
:description Script which Generates and Trains Classifiers based upon config.yml

"""


def train_binary_classifier(classifier: str):
    db_path = f'{Path(Path(__file__).parent.parent.parent.parent, config["paths"]["databases"]["test_train"])}'
    dat = sqlite3.connect(db_path)
    test_col = list(config['data']['attributes'].keys())[0]
    query = dat.execute(
        f"SELECT * FROM {config['classifier'][classifier]['table']} WHERE {test_col} IS NOT NULL")
    cols = [column[0] for column in query.description]
    dataset = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

    model_name = config['classifier'][classifier]['model_name']
    model_path = os.path.join(os.path.dirname(__file__),
                              '..', '..', '..', config['classifier'][classifier]['model_path'])
    column_name = config['classifier'][classifier]['attribute_labels']

    # Do rough simple stratification of the data
    positives = dataset.loc[dataset[column_name] == '1']
    negatives = dataset.loc[dataset[column_name] == '0']

    train_ones = positives.sample(frac=config['classifier'][classifier]['positive_sample_ratio'])
    test_ones = positives.drop(train_ones.index)

    negatives_to_positives_ratio = config['classifier'][classifier]['negatives_to_positives_ratio']
    if negatives_to_positives_ratio == -1:
        sample_negatives = negatives
    else:
        sample_size = negatives_to_positives_ratio * positives.size / negatives.size
        sample_negatives = negatives.sample(frac=sample_size)

    train_zeroes = sample_negatives.sample(frac=config['classifier'][classifier]['negative_sample_ratio'])
    test_zeroes = sample_negatives.drop(train_zeroes.index)

    data = pd.concat([train_ones, train_zeroes], ignore_index=True).sample(frac=1).reset_index(drop=True)
    devData = pd.concat([test_ones, test_zeroes], ignore_index=True).sample(frac=1).reset_index(drop=True)

    # Define pretrained tokenizer and model
    if model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        max_length = 512
        batch_size = 8
        epochs = 3
    elif model_name == "allenai/longformer-base-4096":
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=2)
        max_length = 3072
        batch_size = 1
        epochs = 1
    else:
        print(f"The model: {model_name} is not currently implemented.")
        exit(-1)

    # ----- 1. Preprocess data -----#
    attribute = config['classifier'][classifier]['attribute']

    # Preprocess data
    X = list(data[attribute])
    y = list(pd.to_numeric(data[column_name]))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max_length)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=max_length)

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # ----- 2. Fine-tune pretrained model -----#
    # Define Trainer parameters
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Define Trainer
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        seed=0,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    trainer.train()

    # Save Model Weights (should overwrite existing weights)
    model.save_pretrained(model_path)

    # ----- 3. Predict -----#
    # Load test data
    X_test = list(devData[attribute])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=max_length)

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)

    # Load trained model
    if model_name == "bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    elif model_name == "allenai/longformer-base-4096":
        model = LongformerForSequenceClassification.from_pretrained(model_path, num_labels=2)
    else:
        print(f"The model: {model_name} is not currently implemented.")
        exit(-1)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    y_act = pd.to_numeric(devData[column_name])
    x_text = devData[attribute]

    # Create table to show
    columns = [pd.Series(x_text.values), pd.Series(y_act.values), pd.Series(y_pred)]
    model_pred_results = pd.concat(columns, axis=1, ignore_index=True)
    model_pred_results = model_pred_results.rename(
        columns={0: attribute, 1: "actual", 2: "prediction"}
    )

    model_pred_results["sum"] = (
            pd.to_numeric(model_pred_results["actual"]) +
            pd.to_numeric(model_pred_results["prediction"])
    )

    # Output Result Metrics
    print("FAILED TESTS:")
    print(model_pred_results.loc[model_pred_results["sum"] == 1])
    print("CONFUSION MATRIX\n")
    print(confusion_matrix(list(y_act), list(y_pred)))
    print("CLASSIFICATION REPORT\n")
    print(classification_report(y_act, y_pred))


# Trains ALL the classifiers specified in config.yml
if __name__ == '__main__':
    for key in list(config['classifier'].keys()):
        train_binary_classifier(key)
