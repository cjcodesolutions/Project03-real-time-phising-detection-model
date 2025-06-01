import pandas as pd

def load_phishing_data(path):
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def load_class_descriptions(path):
    df = pd.read_csv(path)
    classes = df['class'].tolist()
    descriptions = df['description'].tolist()
    return classes, descriptions
