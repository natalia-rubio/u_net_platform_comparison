import json
import os

from matplotlib import pyplot as plt

from torch_unet import train_model

results = {}
default_kernel_size = 3
default_batch_size = 16
default_num_batches = 16
# Experiment 1: Vary Kernel Size
print("Experiment 1: Vary Kernel Size")
kernel_sizes_list = [3, 9, 27]
torch_times = []
torch_gpu_times = []
for kernel_size in kernel_sizes_list:
    torch_time = train_model(kernel_size=kernel_size, 
                            batch_size=default_batch_size,
                            num_batches=default_num_batches)
    torch_gpu_time = train_model(use_gpu=True,
                            kernel_size=kernel_size,
                            batch_size=default_batch_size,
                            num_batches=default_num_batches)
    torch_times.append(torch_time)
    torch_gpu_times.append(torch_gpu_time)
print(torch_times)
print(torch_gpu_times)
plt.clf()
plt.plot(kernel_sizes_list, torch_times, label='Torch')
plt.plot(kernel_sizes_list, torch_gpu_times, label='Torch GPU')
plt.xscale('log'); plt.xlabel('Kernel Size')
plt.yscale('log'); plt.ylabel('Time (s)')
plt.legend()
os.makedirs('results', exist_ok=True)
plt.savefig('results/kernel_size_perf_comp.pdf')
plt.savefig('results/kernel_size_perf_comp.png')
results['kernel_size'] = {
    'x': kernel_sizes_list,
    'torch': torch_times,
    'torch_gpu': torch_gpu_times,
}


# Experiment 2: Vary Batch Size
print("Experiment 2: Vary Batch Size")
batch_sizes_list = [4, 8, 16, 32]
torch_times = []
torch_gpu_times = []
for batch_size in batch_sizes_list:
    torch_time = train_model(kernel_size=default_kernel_size, 
                            batch_size=batch_size, 
                            num_batches=default_num_batches)
    torch_gpu_time = train_model(use_gpu=True,
                            kernel_size=default_kernel_size,
                            batch_size=batch_size,  
                            num_batches=default_num_batches)
    torch_times.append(torch_time)
    torch_gpu_times.append(torch_gpu_time)
print(torch_times)
print(torch_gpu_times)
plt.clf()
plt.plot(batch_sizes_list, torch_times, label='Torch')
plt.plot(batch_sizes_list, torch_gpu_times, label='Torch GPU')
plt.xscale('log'); plt.xlabel('Batch Size')
plt.yscale('log'); plt.ylabel('Time (s)')
plt.legend()
os.makedirs('results', exist_ok=True)
plt.savefig('results/batch_size_perf_comp.pdf')
plt.savefig('results/batch_size_perf_comp.png')
results['batch_size'] = {
    'x': batch_sizes_list,
    'torch': torch_times,
    'torch_gpu': torch_gpu_times,
}
# Experiment 3: Vary Number of Batches
print("Experiment 3: Vary Number of Batches")
num_batches_list = [4, 8, 16, 32]
torch_times = []
torch_gpu_times = []
for nb in num_batches_list:
    torch_time = train_model(kernel_size=default_kernel_size,
                            batch_size=default_batch_size,
                            num_batches=nb)
    torch_gpu_time = train_model(use_gpu=True,
                            kernel_size=default_kernel_size,
                            batch_size=default_batch_size,
                            num_batches=nb)
    torch_times.append(torch_time)
    torch_gpu_times.append(torch_gpu_time)
print(torch_times)
print(torch_gpu_times)
plt.clf()
plt.plot(num_batches_list, torch_times, label='Torch')
plt.plot(num_batches_list, torch_gpu_times, label='Torch GPU')
plt.xscale('log'); plt.xlabel('Number of Batches')  
plt.yscale('log'); plt.ylabel('Time (s)')
plt.legend()
os.makedirs('results', exist_ok=True)
plt.savefig('results/num_batches_perf_comp.pdf')
plt.savefig('results/num_batches_perf_comp.png')
results['num_batches'] = {
    'x': num_batches_list,
    'torch': torch_times,
    'torch_gpu': torch_gpu_times,
}

# Save results to json
with open('results/results.json', 'w') as f:
    json.dump(results, f)