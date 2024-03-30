import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Load USPS dataset
digits = load_digits()


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)


X_train = X_train / 16.0
X_test = X_test / 16.0
# MLP model architecture
mlp_model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(64,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])


mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
mlp_history = mlp_model.fit(X_train, y_train, epochs=20, validation_split=0.2)

X_train_cnn = X_train.reshape(-1, 8, 8, 1)
X_test_cnn = X_test.reshape(-1, 8, 8, 1)


cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=20, validation_split=0.2)

mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test, y_test)


cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)

print(f"MLP Accuracy: {mlp_accuracy*100:.2f}%")
print(f"CNN Accuracy: {cnn_accuracy*100:.2f}%")
def plot_history(history, model_name):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.legend()
    plt.show()

plot_history(mlp_history, 'MLP')
plot_history(cnn_history, 'CNN')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


mlp_model = MLP()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


num_epochs = 20
writer = SummaryWriter()

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = mlp_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


    writer.add_scalar('MLP/loss', loss.item(), epoch)


writer.close()

X_train_cnn_tensor = torch.tensor(X_train_cnn, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
train_dataset_cnn = TensorDataset(X_train_cnn_tensor, y_train_tensor)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


cnn_model = CNN()
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)


num_epochs = 20
criterion = nn.CrossEntropyLoss()
writer_cnn = SummaryWriter()

for epoch in range(num_epochs):
    for batch in train_loader_cnn:
        optimizer_cnn.zero_grad()
        inputs, targets = batch
        inputs = inputs.permute(0, 3, 1, 2)
        outputs = cnn_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_cnn.step()


    writer_cnn.add_scalar('CNN/loss', loss.item(), epoch)


writer_cnn.close()
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_test_cnn_tensor = torch.tensor(X_test_cnn, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


X_test_cnn_tensor = X_test_cnn_tensor.permute(0, 3, 1, 2)


mlp_model.eval()
mlp_outputs = mlp_model(X_test_tensor)
mlp_preds = torch.argmax(mlp_outputs, dim=1)
mlp_accuracy = accuracy_score(y_test, mlp_preds)
mlp_precision = precision_score(y_test, mlp_preds, average='macro')
mlp_recall = recall_score(y_test, mlp_preds, average='macro')
mlp_conf_matrix = confusion_matrix(y_test, mlp_preds)


cnn_model.eval()
cnn_outputs = cnn_model(X_test_cnn_tensor)
cnn_preds = torch.argmax(cnn_outputs, dim=1)
cnn_accuracy = accuracy_score(y_test, cnn_preds)
cnn_precision = precision_score(y_test, cnn_preds, average='macro')
cnn_recall = recall_score(y_test, cnn_preds, average='macro')
cnn_conf_matrix = confusion_matrix(y_test, cnn_preds)

print("MLP Metrics:")
print(f"Accuracy: {mlp_accuracy*100:.2f}%")
print(f"Precision: {mlp_precision:.4f}")
print(f"Recall: {mlp_recall:.4f}")
print("Confusion Matrix:")
print(mlp_conf_matrix)

print("\nCNN Metrics:")
print(f"Accuracy: {cnn_accuracy*100:.2f}%")
print(f"Precision: {cnn_precision:.4f}")
print(f"Recall: {cnn_recall:.4f}")
print("Confusion Matrix:")
print(cnn_conf_matrix)