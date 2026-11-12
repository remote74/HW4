import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./fine_tuned_model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load fine-tuned model
model_ft = AutoModelForSequenceClassification.from_pretrained(model_path)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ft.to(device)
model_ft.eval()