import torch.nn as nn
import torch.nn.functional as F

class PaperCNN(nn.Module):
    """CNN architecture presented in the paper at section 5.1."""

    def __init__(self, num_classes=10):
        super(PaperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.pool = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(0.25)
        self.dense1 = nn.Linear(32 * 11 * 11, 128)
        self.drop2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 32 * 11 * 11)
        x = self.drop1(x)
        x = F.relu(self.dense1(x))
        x = self.drop2(x)
        x = self.dense2(x)
        #x = F.softmax(x, dim=1) # Softmax will be applied in loss function (CrossEntropyLoss)
   
        return x
    
    def forward_features(self, x):
        # everything up to last linear layer (minimal extension task - last layer is bayesian)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        # No dropout here; only feature extraction
        x = x.view(-1, 32 * 11 * 11)
        x = F.relu(self.dense1(x))
        
        return x