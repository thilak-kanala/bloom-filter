import os
import sys
import random

random_words = [
'green', 'double', 'disability', 'fold', 'bet', 'tendency', 'extract', 'deputy', 'trolley', 'indoor', 'van', 'rain', 'preference', 'laboratory', 'realize', 'wine', 'pressure', 'hilarious', 'outlook', 'voter', 'account', 'irony', 'end', 'toll', 'am', 'creed', 'absorb', 'allocation', 'hurt', 'last', 'key', 'medieval', 'oil', 'stir', 'hair', 'breathe', 'favorable', 'solve', 'rich', 'quest', 'trap', 'drawer', 'animal', 'fitness', 'construct', 'factory', 'humanity', 'sugar', 'concrete', 'hallway']

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


i = 0
for word in words[:(N_WORDS/2)]:
    modified_words_query_fh.write(word)
    modified_words_query_fh.write(" ")

    progress = ((i / (N_WORDS/2)) * 100)
    if (progress % 20 == 0):
        print(f'real query words: {progress} %')
    i = i + 1

for i in range(0, (N_WORDS/2)):
    modified_words_query_fh.write(random.choice(random_words))
    modified_words_query_fh.write(" ")

    progress = ((i / (N_WORDS/2)) * 100)
    if (progress % 20 == 0):
        print(f'fake query words: {progress} %')
    i = i + 1

modified_words_query_fh.close()

print("== words_query.txt generated ==")

# print(final_string)
print('== Data Prprocessing Done! ==')
print('== Generated "words_insert.txt" and "words_query.txt" ==')
