import deepchem as dc
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tensorflow

# Load the Tox21 dataset
tasks, datasets, transformer = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = datasets

# Select a single task for demonstration
selected_task = tasks[0]  


class Tox21Dataset(Dataset):
    """PyTorch Dataset for Tox21 using DeepChem data."""
    def __init__(self, deepchem_dataset):
        self.dataset = deepchem_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the molecule as a fingerprint and the label
        x = self.dataset.X[idx]
        y = self.dataset.y[idx, 0]  # Assuming we're focusing on the first task
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# Convert DeepChem datasets to PyTorch datasets
train_dataset_pytorch = Tox21Dataset(train_dataset)
valid_dataset_pytorch = Tox21Dataset(valid_dataset)
test_dataset_pytorch = Tox21Dataset(test_dataset)



class Tox21Predictor(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(Tox21Predictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')



def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predicted = outputs.round()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

model = Tox21Predictor(input_size=train_dataset.X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(train_dataset_pytorch, batch_size=64, shuffle=True)
train_model(model, criterion, optimizer, train_loader)
valid_loader = DataLoader(valid_dataset_pytorch, batch_size=64, shuffle=False)
evaluate_model(model, valid_loader)
