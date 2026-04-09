import torch
import flwr as fl

print("=======gpu checking=======")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, end="\n\n")


print("=======flwr checking=======")
print("flwr: " ,fl.__version__)