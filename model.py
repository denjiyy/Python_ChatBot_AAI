import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        # Adding more hidden layers to the architecture
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2)  # Increased size for second hidden layer
        self.l3 = nn.Linear(hidden_size * 2, hidden_size * 2)  # Third hidden layer
        self.l4 = nn.Linear(hidden_size * 2, hidden_size)  # Fourth hidden layer
        self.l5 = nn.Linear(hidden_size, num_classes)  # Output layer

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass the input through each layer
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        return out