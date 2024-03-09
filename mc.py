from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from transformers import AdamW
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)



class CustomDataset(Dataset):
    def __init__(self, csv_file, max_samples=1000, tokenizer = None, max_length = 16):
        self.data = self.load_data(csv_file)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.transform = transforms.ToTensor()

        for index, row in self.data.iterrows():
            if len(self.samples) >= self.max_samples:
                break

            caption_list = row['captions']
            caption_list = caption_list[0].split('\n')
            caption_list = [i.strip() for i in caption_list]
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
        image, caption= self.samples[index]
        return image, caption

    def load_data(self, csv_file):
        data = pd.read_csv(csv_file)
        data['captions'] = data['captions'].apply(lambda x: x.replace('[', '').replace(']', '').replace('.','').replace("'", '').lower().split(','))
        return data

    def load_image(self, image):
        image = Image.open(image).convert("RGB")
        tensor = self.transform(image)
        tensor.requires_grad = True
        return tensor
    
    def preprocess_caption(self,text):
        input =  self.tokenizer.encode(text, max_length=self.max_length, padding='max_length', return_tensors="pt").to(device)
        input.requires_grad = True
        return input

def predict_output(images, model, feature_extractor, tokenizer, max_length, gen_kwargs):
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    pixel_values.requires_grad = True

    output_ids = model.generate(pixel_values, **gen_kwargs)
    output_ids = list(output_ids)

    for i in range(len(output_ids)):
        padding_length = max_length - output_ids[i].numel()
        if padding_length > 0:
            output_ids[i] = F.pad(output_ids[i], (0, padding_length),value = tokenizer.pad_token_id)

    return torch.stack(output_ids)

def load_dataset(pickle_path, path):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = CustomDataset(path,max_samples = max_samples, tokenizer=tokenizer, max_length=max_length)
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
    return dataset

def get_model(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded saved model")

    else:
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer

def save_model(model, optimizer, checkpoint_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print("Model Saved")

def l2_normalize(embedding):
    embedding = embedding.float()
    norm = torch.norm(embedding)
    return embedding / norm if norm > 0 else embedding


# Define the path to the JSON file
train_dataset_path = "./train.csv"
test_dataset_path = "./test.csv"
valid_dataset_path = "./valid.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nlpconnect/vit-gpt2-image-captioning"
checkpoint_path = "Caption_checkpoint.pt"
max_length = 32
learning_rate = 1e-3
batch_size = 16
epochs = 10
max_samples = 3000

# Write hyperparameters to a text file
hyperparameters = {
    "model_name": str(model_name),
    "learning_rate": str(learning_rate),
    "batch_size": str(batch_size),
    "num_epochs": str(epochs),
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

model ,optimizer = get_model(checkpoint_path)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

gen_kwargs = {
    "max_length": max_length,
    "num_beams": 4,
    "do_sample": True,
    "temperature": 0.9,
    "num_return_sequences": 1,
    "pad_token_id": tokenizer.pad_token_id,
}


train_dataset = load_dataset("train_dataset.pkl", train_dataset_path)
valid_dataset = load_dataset("valid_dataset.pkl", valid_dataset_path)
test_dataset = load_dataset("test_dataset.pkl", test_dataset_path)

if hyperparameters != saved_hyperparameters:
    with open("hyperparameters.txt", "w") as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

def contrastive_loss(similarity_matrix, labels, margin=0.2):
    loss_contrastive = (1 - labels) * F.relu(margin - similarity_matrix) ** 2 + \
                       labels * similarity_matrix ** 2
    
    return loss_contrastive.mean()

# Train the model

# Initialize lists to store train and validation losses
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        try:
            images, captions = data
            images = list(images)
            captions = captions.to(device).squeeze(1)

            optimizer.zero_grad()
            outputs = predict_output(images, model, feature_extractor, tokenizer, max_length, gen_kwargs)

            outputs = l2_normalize(outputs)
            captions = l2_normalize(captions)

            outputs.requires_grad = True
            captions.requires_grad = True

            cosine_sim = F.cosine_similarity(outputs, captions, dim=1)

            # Define contrastive loss function
            loss = contrastive_loss(cosine_sim, torch.ones_like(cosine_sim))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        except KeyboardInterrupt:
            print("Force Checkpoint")
            save_model(model, optimizer,checkpoint_path)
    
    save_model(model, optimizer, checkpoint_path)

    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_loader)
    print(f"Train Loss: {average_loss:.4f}")

    # Append the train loss to the list
    train_losses.append(average_loss)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            images, captions = data
            images = list(images)
            captions = captions.to(device).squeeze(1)

            optimizer.zero_grad()
            outputs = predict_output(images, model, feature_extractor, tokenizer, max_length, gen_kwargs)

            outputs = l2_normalize(outputs)
            captions = l2_normalize(captions)

            outputs.requires_grad = True
            captions.requires_grad = True

            cosine_sim = F.cosine_similarity(outputs, captions, dim=1)

            # Define contrastive loss function
            loss = contrastive_loss(cosine_sim, torch.ones_like(cosine_sim))
            total_loss += loss.item()

    # Calculate average loss for the validation set
    average_loss = total_loss / len(valid_loader)
    print(f"Validation Loss: {average_loss:.4f}")

    # Append the validation loss to the list
    val_losses.append(average_loss)

# Plot the train loss
plt.figure()
plt.plot(range(1, epochs + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss')
plt.show()

# Plot the validation loss
plt.figure()
plt.plot(range(1, epochs + 1), val_losses)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss')
plt.show()
