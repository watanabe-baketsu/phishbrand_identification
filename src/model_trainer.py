import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer

from .model_poc import MultiModalT5


# Your MultiModalT5 definition here...

# Create dummy dataset
n_samples = 1000
dummy_text_data = ["This is some HTML content" for _ in range(n_samples)]
dummy_image_data = torch.randn(n_samples, 3, 224, 224)  # Random image data
dummy_target_data = ["generate this text" for _ in range(n_samples)]

# Convert text data to input tensors
tokenizer = T5Tokenizer.from_pretrained('t5-small')
dummy_target_data_encoded = tokenizer(dummy_target_data, return_tensors='pt', padding=True, truncation=True).input_ids

# Create DataLoader
dataset = TensorDataset(dummy_image_data, dummy_target_data_encoded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model and optimizer
model = MultiModalT5()
optimizer = Adam(model.parameters(), lr=1e-4)

# Set the model to training mode
model.train()

# Training loop
n_epochs = 5
for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        image_input, text_target = batch

        # Forward pass
        text_input = ["This is some HTML content" for _ in range(text_target.shape[0])]  # Replace with actual text
        generated_text = model(text_input, image_input)
        generated_text_ids = tokenizer(generated_text, return_tensors='pt', padding=True, truncation=True).input_ids

        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(generated_text_ids.view(-1, generated_text_ids.size(-1)), text_target.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Iteration {i+1}, Loss: {loss.item()}")

    print(f"Epoch {epoch+1} completed")
