import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class CSVImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, has_labels=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        row = self.data.iloc[i]

        # This is because the test csv file does not have a label column
        if self.has_labels:
            label = int(row['label'])
            pixels = row.drop('label').values.astype('float32')
        else:
            label = -1  # dummy value
            pixels = row.values.astype('float32')
        
        pixels = pixels / 255.0
        image = torch.tensor(pixels).view(1, 28, 28)

        if self.transform:
            image = self.transform(image)

        return image, label

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)    # Fully connected layer with input size 28*28=784 to 128 neurons.
        self.relu = nn.ReLU()   # Activation function introducing non-linearity
        self.fc2 = nn.Linear(128, 10)   # Output layer with 10 neurons 

    # defines data flow through the network
    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # output: 64x28x28
        self.pool = nn.MaxPool2d(2, 2)      # Reduces spatial size and makes the model faster and more robust
        self.dropout = nn.Dropout(0.25)     # Helps to reduce overfitting
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Transform(data preprocessing): convert images to tensor and normalize pixel values to [0,1]
transform = transforms.Normalize((0.1307,), (0.3081,))

train_dataset = CSVImageDataset('data/train.csv', transform=transform, has_labels=True)
test_dataset = CSVImageDataset('data/test.csv', transform=transform, has_labels=False)

# Data loaders beaks dataset into batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
model = CNN()
criterion = nn.CrossEntropyLoss()   # is standard for multi-class classification, combines softmax + log loss
optimizer = optim.Adam(model.parameters(), lr=0.001)    # optimizer adapts learning rates per parameter; popular and efficient

# Training Loop
def train(model, device, train_loader, optimizer, criterion, epochs):
    model.train()   # sets PyTorch in training mode

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)   # moves data to GPU if available

            optimizer.zero_grad()       # clear gradients
            output = model(data)        # forward pass
            loss = criterion(output, target)  # compute loss
            loss.backward()             # backpropagation
            optimizer.step()            # update weights

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

def test(model, device, test_loader, submission_filename='submission.csv'):
    model.eval()    # Switch model to evaluation mode
    predictions = []

    with torch.no_grad():   # disables gradient calculation (saves memory)
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)   # runs a forward pass through the NN on batch of images
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    submission = pd.DataFrame({
        'ImageId': list(range(1, len(predictions) + 1)),
        'Label': predictions
    })

    submission.to_csv(submission_filename, index=False)
    print(f"Submission saved to {submission_filename}")

    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train(model, device, train_loader, optimizer, criterion, epochs=5)
test(model, device, test_loader)