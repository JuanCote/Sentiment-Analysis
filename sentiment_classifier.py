import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pretrained tokenizer and model
tokenizer = BertTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
model = BertForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

# Ensure the model is on the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def classify_review(review):
    # Tokenize and convert to tensors
    inputs = tokenizer(
        [review], return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)

    # Return the predicted rating
    return prediction.item()  # Returns a numerical rating from 0 to 5
