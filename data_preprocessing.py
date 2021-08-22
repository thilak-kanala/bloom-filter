import os
import sys
import random

print('== Data Preprocessing Started ... ==')

N_WORDS = int((sys.argv)[1])

f = open("random_strings.txt")

words = f.read().split('\n')[:-1]

f.close()

if os.path.isfile('words_insert.txt'):
    os.remove('words_insert.txt')

modified_words_fh = open("words_insert.txt", 'w+')
modified_words_fh.write(" ")

final_string = " "

i = 0
for word in words[:N_WORDS]:
    modified_words_fh.write(word)
    modified_words_fh.write(" ")

    progress = ((i / N_WORDS) * 100)
    if (progress % 20 == 0):
        print(f'{progress} %')
    i = i + 1

modified_words_fh.close()
print("== words_insert.txt generated ==")

# ----------------------------------------------------------- #

if os.path.isfile('words_query.txt'):
    os.remove('words_query.txt')

modified_words_query_fh = open("words_query.txt", 'w+')
modified_words_query_fh.write(" ")


n_real_words = int(N_WORDS/2)
n_fake_words = N_WORDS - n_real_words

i = 0
for word in words[:n_real_words]:
    modified_words_query_fh.write(word)
    modified_words_query_fh.write(" ")

    progress = ((i / n_real_words) * 100)
    if (progress % 20 == 0):
        print(f'real query words: {progress} %')
    i = i + 1

print('---')

for i in range(0, n_fake_words):
    modified_words_query_fh.write(random.choice(['apple', 'banana', 'mango']))
    modified_words_query_fh.write(" ")

    progress = ((i / n_fake_words) * 100)
    if (progress % 20 == 0):
        print(f'fake query words: {progress} %')
    i = i + 1

modified_words_query_fh.close()

print("== words_query.txt generated ==")

# print(final_string)
print('== Data Prprocessing Done! ==')
print('== Generated "words_insert.txt" and "words_query.txt" ==')
