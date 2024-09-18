# sentiment_analysis_project.py

# Importing necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from fastapi import FastAPI, Request
import uvicorn

# Load IMDb dataset
print("Loading IMDb dataset...")
dataset = load_dataset('imdb')

# Tokenization using BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_batch(text_list):
    return tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')

# Define the Sentiment Classifier Model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)

# Training Setup
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

# Prepare Dataset for Training
def prepare_data(dataset, batch_size=16, sample_size=1000):
    print(f"Preparing data (using {sample_size} samples)...")
    texts = dataset['train']['text'][:sample_size]  # Limiting data for faster training
    labels = torch.tensor(dataset['train']['label'][:sample_size])

    encodings = preprocess_batch(texts)
    dataset_tensor = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    data_loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)
    
    return data_loader

# Run Training
def run_training():
    print("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(n_classes=2).to(device)

    train_loader = prepare_data(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 3
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}')

    # Save the model after training
    torch.save(model.state_dict(), 'sentiment_model.pt')
    print("Model training complete and saved as 'sentiment_model.pt'.")

run_training()

# FastAPI Integration for Deployment
app = FastAPI()

# Load Model for FastAPI
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentClassifier(n_classes=2).to(device)
model.load_state_dict(torch.load('sentiment_model.pt'))
model.eval()

@app.post("/predict/")
async def predict_sentiment(request: Request):
    data = await request.json()
    text = data['text']
    
    # Tokenize input text
    encoding = preprocess_batch([text])
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(output, dim=1).item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"text": text, "sentiment": sentiment}

# For local testing: FastAPI server
if __name__ == "__main__":
    # Test with FastAPI locally
    uvicorn.run(app, host="127.0.0.1", port=8000)
