import torch
import os


def set_gpu(gpus):
	if torch.cuda.is_available():
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		torch.cuda.set_device(0)
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	return device
