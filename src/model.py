import torch
import torch.nn as nn



# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        """
        # 3CV-Layers -> did not work in udacity cuda environment -> error: out of memory
        #               but I tested it on my local macine
        self.model = nn.Sequential(
            #convolutional layer 1 - input tensor 3 x 254 x 254
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            #convolutional layer 2 - input tensor 16 x 127 x 127
            torch.nn.Conv2d(16, 32, 4, stride=1, padding=1),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            #convolutional layer 3 - input tensor 32 x 63 x 63
            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            #classifier
            nn.Flatten(), # 64 x 27 x 27
            #linear layer 1
            nn.Linear(64*27*27, 23828),
            nn.Dropout(dropout),
            torch.nn.ReLU(),
            #linear output layer 
            nn.Linear(23828, num_classes)
        )
        """
        
      
        """
        #### second model try - only 2 cv layers
        self.model = nn.Sequential(
            #convolutional layer 1 - input tensor 3 x 254 x 254
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        
            #convolutional layer 2 - input tensor 16 x 127 x 127
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        
   
        
            #classifier
            nn.Flatten(), # 32 x 28 x 28
            #linear layer 1
            nn.Linear(32*28*28, 16400),
            nn.Dropout(dropout),
            torch.nn.ReLU(),
            #linear output layer 
            nn.Linear(16400, num_classes)
        )
        """
        
        
        # 3CV-Layers 
        self.model = nn.Sequential(
        #convolutional layer 1 - input tensor 3 x 254 x 254
        torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(3, 3),
        
        #convolutional layer 2 - input tensor 16 x 84 x 84
        torch.nn.Conv2d(16, 32, 3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        
        #convolutional layer 3 - input tensor 32 x 42 x 42
        torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        
        #convolutional layer 4 - input tensor 64 x 21 x 21
        torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
            
        #classifier
        nn.Flatten(), # 128 x 9 x9
        #linear layer 1
        nn.Linear(128*9*9, 5200),
        nn.Dropout(dropout),
        torch.nn.ReLU(),
        #linear output layer 
        nn.Linear(5200, num_classes)
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)
        


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
