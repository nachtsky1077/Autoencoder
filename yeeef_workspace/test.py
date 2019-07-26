import torch
class AutoEncoder(torch.nn.Module):
    """
    untied version
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(True), 
            torch.nn.Linear(64, 12), 
            torch.nn.ReLU(True), 
            torch.nn.Linear(12, 2)
        )
        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 12),
            torch.nn.ReLU(True),
            torch.nn.Linear(12, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(True), 
            torch.nn.Linear(128, 28 * 28), 
            torch.nn.Tanh()
        )
        
    def encode(self):
        return self._encoder
    
    def decode(self):
        return self._decoder
    
    def forward(self, input_data):
        return self.encode(input_data)