import torch
print(torch.version.cuda)          # Should be something like '11.8'
print(torch.backends.cudnn.version())  # Should print a number (e.g., 8600)
