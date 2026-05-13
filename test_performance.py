from torch_unet import train_model
from matplotlib import pyplot as plt

# Experiment 1: Vary Kernel Size
kernel_sizes = [3, 5, 7, 9]
torch_times = []
torch_gpu_times = []
for kernel_size in kernel_sizes:
    torch_time = train_model(kernel_size=kernel_size)
    torch_gpu_time = train_model(kernel_size=kernel_size, use_gpu=True)
    torch_times.append(torch_time)
    torch_gpu_times.append(torch_gpu_time)
print(torch_times)
print(torch_gpu_times)
plt.plot(kernel_sizes, torch_times, label='Torch')
plt.plot(kernel_sizes, torch_gpu_times, label='Torch GPU')
plt.legend()
plt.savefig('kernel_size_perf_comp.pdf')
# Experiment 2: Vary Batch Size

# Experiment 3: Vary Number of Batches