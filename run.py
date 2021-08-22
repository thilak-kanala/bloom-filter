import subprocess
import time
import matplotlib.pyplot as plt

cpu_time = []
gpu_time = []

run = ["nvcc", "-G", "./bloom-filter/gpu_bloom/gpu_bloom_v2.cu", "-o", "./bloom-filter/gpu_bloom/gpu_bloom.out"]
output = subprocess.run(run, capture_output=True)
print(output)
# !nvcc -G ./bloom-filter/gpu_bloom/gpu_bloom_v2.cu -o ./bloom-filter/gpu_bloom/gpu_bloom.out

start_n = 0
stop_n = 1000000
step_n = 100000

x = []
for n_words in range(start_n, stop_n, step_n):
  x.append(n_words)

  run = ["python3", "./bloom-filter/data_preprocessing.py", str(n_words)]
  output = subprocess.run(run, capture_output=True)
  print(output)

  start = time.time()
  output = subprocess.run("./bloom-filter/gpu_bloom/gpu_bloom.out", capture_output=True)
  print(output)
  end = time.time()

  # print(f'GPU: {end - start} s')
  gpu_time.append(end - start)

  # !gcc ./bloom-filter/cpu_bloom/cpu_bloom.c ./bloom-filter/cpu_bloom/xxhash64-ref.c -o ./bloom-filter/cpu_bloom/cpu_bloom.out
  run = ["gcc", "./bloom-filter/cpu_bloom/cpu_bloom.c", "./bloom-filter/cpu_bloom/xxhash64-ref.c", "-o", "./bloom-filter/cpu_bloom/cpu_bloom.out"]
  output = subprocess.run(run, capture_output=True)
  print(output)

  run = ["./bloom-filter/cpu_bloom/cpu_bloom.out", str(n_words), str(n_words)]
  start = time.time()
  output = subprocess.run(run, capture_output=True)
  print(output)
  end = time.time()

  # print(f'CPU: {end - start} s')
  cpu_time.append((end - start))

  progress = ((n_words / step_n) * 100) / ((stop_n - start_n) / step_n)
  print(f'{progress} %')

print('Done!')
# print(f'cpu_time: {cpu_time}')
# print(f'gpu_time: {gpu_time}')