title = f'Bloom Filter Insertion - CPU vs GPU'

plt.title(title)

plt.xlabel("n_words")
plt.ylabel("Execution Time (s)")

plt.plot(x, cpu_time, 'x-', label='CPU ()')
plt.plot(x, gpu_time, 'x-', label='GPU ()')

plt.legend()

plt.savefig("graph.png",dpi=1200)

print('Done')