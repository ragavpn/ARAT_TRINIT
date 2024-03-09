import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from PIL import Image
import io
import base64
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle

warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Dataset Preparation
class CustomDataset(Dataset):
    def __init__(self, csv_file, max_samples=1000, tokenizer = None, max_length = 32):
        self.data = self.load_data(csv_file)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for index, row in self.data.iterrows():
            if len(self.samples) >= self.max_samples:
                break

            caption_list = row['captions']
            caption_list = list(set(caption_list))
            path = row['filepath']
            path = "./" + path
            try:
                image = self.load_image(path)
                for caption in caption_list:
                    encoded_caption = self.preprocess_caption(caption)
                    self.samples.append((image, encoded_caption))
            except (IOError,OSError) as e:
                continue
        self.samples = self.samples[:self.max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, caption, one_hot_label = self.samples[index]
        return image, caption, one_hot_label

    def load_data(self, csv_file):
        data = pd.read_csv(csv_file)
        return data

    def load_image(self, image):
        image = Image.open(image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to a specific size
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        image = transform(image)
        return image
    
    def preprocess_caption(self,text):

        # Tokenize the text and add special tokens [CLS] and [SEP]
        tokens = ["[CLS]"] + self.tokenizer.tokenize(text) + ["[SEP]"]

        # Convert tokens to token IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Truncate if the sequence is longer than max_length
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

class CaptionModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, vocab_size = 30522, common_dim=500, hidden_dim=256):
        super(CaptionModel, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = nn.Linear(image_encoder.fc.out_features, common_dim)
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, common_dim)
        self.decoder = nn.LSTM(common_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, captions):
        image_features = self.image_encoder(images)
        image_features = self.image_projection(image_features)

        # Normalize the image features
        image_features = F.normalize(image_features, dim=-1)

        # Reshape image features to (batch_size, 1, common_dim) for LSTM
        image_features = image_features.unsqueeze(1)

        # Pass the image features and captions through the LSTM decoder
        decoder_outputs, _ = self.decoder(torch.cat([image_features, captions], dim=1))

        # Pass the decoder outputs through the dropout layer
        decoder_outputs = self.dropout(decoder_outputs)

        # Pass the decoder outputs through the fully connected layer to get the final output
        output = self.fc_out(decoder_outputs)

        return output


def contrastive_loss(similarity_matrix, labels, margin=0.2):
    loss_contrastive = (1 - labels) * F.relu(margin - similarity_matrix) ** 2 + \
                       labels * similarity_matrix ** 2
    
    return loss_contrastive.mean()

def get_model(checkpoint_path):

    # Load or initialize model, optimizer, scheduler
    if os.path.exists(checkpoint_path):
        # Load checkpoint if it exists
        checkpoint = torch.load(checkpoint_path)
        model = BertModel.from_pretrained(model_name).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded saved model")
    else:
        # Initialize model, optimizer, scheduler if checkpoint doesn't exist
        model = BertModel.from_pretrained(model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer

def save_model(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print("Model Saved")

def load_dataset(pickle_path, path):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = CustomDataset(path,max_samples = max_samples, tokenizer=tokenizer, max_length=max_length)
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
    return dataset

# Define the path to the JSON file
train_dataset_path = "./train.csv"
test_dataset_path = "./test.csv"
valid_dataset_path = "./valid.csv"


# Define hyperparameters and other configurations
model_name = "bert-base-uncased"
checkpoint_path = "Caption_checkpoint.pt"
learning_rate = 0.001
batch_size = 16
num_epochs = 20
max_samples = 1600
max_length = 64  # Define the desired maximum sequence length for BERT

# Write hyperparameters to a text file
hyperparameters = {
    "model_name": str(model_name),
    "learning_rate": str(learning_rate),
    "batch_size": str(batch_size),
    "num_epochs": str(num_epochs),
    "max_samples": str(max_samples),
    "max_length": str(max_length)
}

# Check if hyperparameters have changed
with open("hyperparameters.txt", "r") as f:
    saved_hyperparameters = {}
    for line in f:
        key, value = line.strip().split(": ")
        saved_hyperparameters[key] = value

if hyperparameters != saved_hyperparameters:
    print("Hyperparameters have changed. Deleting old files...")
    if os.path.exists("train_dataset.pkl"):
        os.remove("train_dataset.pkl")
    if os.path.exists("valid_dataset.pkl"):
        os.remove("valid_dataset.pkl")
    if os.path.exists("test_dataset.pkl"):
        os.remove("test_dataset.pkl")
    if os.path.exists("Caption_checkpoint.pt"):
        os.remove("Caption_checkpoint.pt")

# Encoders for the model
resnet = models.resnet50(pretrained=True)
tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)

train_dataset = load_dataset("train_dataset.pkl", train_dataset_path)
valid_dataset = load_dataset("valid_dataset.pkl", valid_dataset_path)
test_dataset = load_dataset("test_dataset.pkl", test_dataset_path)

if hyperparameters != saved_hyperparameters:
    with open("hyperparameters.txt", "w") as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

# Split the dataset into training and validation sets
train_loader = DataLoader(train_dataset.samples, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset.samples, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset.samples, batch_size=batch_size, shuffle=True)

# Create Caption model instance
caption_model = CaptionModel(resnet, bert)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(caption_model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)

# Initialize lists to store train and validation losses
train_losses = []
val_losses = []

for epoch in range(num_epochs):

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    caption_model.train()
    total_loss = 0.0

    for batch_images, batch_captions in train_loader:
        try:
            batch_images = batch_images.to(device)
            batch_captions = {
                key: value.to(device) for key, value in batch_captions.items()
            }

            # Forward pass
            optimizer.zero_grad()
            outputs = caption_model(batch_images, batch_captions)

            # Calculate the loss using CrossEntropyLoss
            loss = criterion(outputs.view(-1, outputs.size(2)), batch_captions.view(-1))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        except KeyboardInterrupt:
            print("Force Checkpoint")
            save_model(caption_model, optimizer, epoch, checkpoint_path)
        
    save_model(caption_model, optimizer, epoch, checkpoint_path)

    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_loader)
    print(f"Train Loss: {average_loss:.4f}")

    # Append the train loss to the list
    train_losses.append(average_loss)

    # Validation loop
    caption_model.eval()  # Set the model to evaluation mode (disable dropout, batch normalization, etc.)
    total_loss = 0.0

    with torch.no_grad():
        for batch_images, batch_captions in valid_loader:
            # Move data to the appropriate device (GPU if available)
            batch_images = batch_images.to(device)
            batch_captions = {
                key: value.to(device) for key, value in batch_captions.items()
            }

            # Forward pass
            outputs = caption_model(batch_images, batch_captions)

            # Calculate the loss using CrossEntropyLoss
            loss = criterion(outputs.view(-1, outputs.size(2)), batch_captions.view(-1))
            total_loss += loss.item()

    # Calculate average loss for the validation set
    average_loss = total_loss / len(valid_loader)
    print(f"Validation Loss: {average_loss:.4f}")

    # Append the validation loss to the list
    val_losses.append(average_loss)

#Test the model
caption_model.eval()  # Set the model to evaluation mode (disable dropout, batch normalization, etc.)
total_loss = 0.0

with torch.no_grad():
    for batch_images, batch_captions in test_loader:
        # Move data to the appropriate device (GPU if available)
        batch_images = batch_images.to(device)
        batch_captions = {
            key: value.to(device) for key, value in batch_captions.items()
        }

        # Forward pass
        optimizer.zero_grad()
        logits_per_image, logits_per_text = caption_model(batch_images, batch_captions)

        # Combine logits for images and texts
        combined_logits = logits_per_image + logits_per_text

        batch_labels = torch.eye(batch_size).to(device)

        # Calculate the loss using CrossEntropyLoss
        loss = contrastive_loss(combined_logits, batch_labels)
        total_loss += loss.item()
    
# Calculate average loss for the test set
average_loss = total_loss / len(test_loader)
print(f"Test Loss: {average_loss:.4f}")

# Plot the train loss
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss')
plt.show()

# Plot the validation loss
plt.figure()
plt.plot(range(1, num_epochs + 1), val_losses)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss')
plt.show()
