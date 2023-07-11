import torch
import torch.nn as nn
import torch.nn.functional as F


class FashNet(nn.Module):
    def __init__(self, in_d, out_d):
        super(FashNet, self).__init__()

        # Create three fully-connected Linear layers: 256, 128, and 64 nodes wide
        # Follow each with a ReLU activation layer
        ## Your code here
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        # Create the last fully-connected Linear layer that takes it down to out_d nodes
        self.fc_last = nn.Linear(64, 10)
        # Create a softmax layer to create probabilities for each class
        self.sftmx = F.softmax
    # This is the method called when the model is doing inference
    def forward(self, x):

        # Run it through the three linear layers and their activation layers
        ## Your code here
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Last layer and softmax
        ## Your code here
        output = self.sftmx(self.fc_last(x))
        return output  ## Your code here