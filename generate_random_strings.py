import random
import string
import sys

letters = string.ascii_letters + string.digits

file = open('random_strings.txt', 'w')

# n_mb = 40
# n_words = int((n_mb * (10 ** 6)) / 16)

n_words = int((sys.argv)[1])

print(f'{n_words} random strings are being generated ...')

for i in range(n_words):
    size = random.randint(16, 128)
    string = ''.join(random.choice(letters) for i in range(size)) 
    file.write(string)

    progress = ((i / n_words) * 100)
    if (progress % 10 == 0): print(f'{progress} %')

    if i < n_words-1:
        file.write('\n')

print('100 %')
        
print("== Done generating random strings ==")
file.close()