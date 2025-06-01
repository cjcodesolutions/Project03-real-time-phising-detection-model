from sentence_transformers import SentenceTransformer, util
import pandas as pd
from utils import load_class_descriptions

class ZSLPhishingClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.classes = []
        self.class_embeddings = None

    def load_classes(self, class_desc_path):
        self.classes, descriptions = load_class_descriptions(class_desc_path)
        self.class_embeddings = self.model.encode(descriptions, convert_to_tensor=True)

    def classify(self, text):
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(text_embedding, self.class_embeddings)
        best_idx = cosine_scores.argmax().item()
        confidence = cosine_scores[0][best_idx].item()
        return self.classes[best_idx], confidence

if __name__ == "__main__":
    zsl_classifier = ZSLPhishingClassifier()
    zsl_classifier.load_classes('./data/class_descriptions.csv')

    test_samples = [
        "Urgent! Please update your password at http://phishy-login.com now.",
        "Click this link to claim your free prize: http://scam-gift.com"
    ]

    for sample in test_samples:
        cls, conf = zsl_classifier.classify(sample)
        print(f"Input: {sample}")
        print(f"Predicted Class: {cls} (Confidence: {conf:.3f})\n")
