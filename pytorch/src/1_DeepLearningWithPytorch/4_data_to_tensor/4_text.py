import torch
import numpy as np

##----------------------------------
## char level one-hot encoding
with open("./data/p1ch4/jane-austen/1342-0.txt",mode="r",encoding="utf-8") as f:
    text = f.read()
text

# select 200th line
lines = text.split("\n")
line = lines[200]
line

# create one-hot encoding tensor
letter_t = torch.zeros(len(line),128)
letter_t.shape

# fill one-hot encoding tensor
for i,letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1
letter_t

##----------------------------------
## word level one-hot encoding
def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list
words_in_line = clean_words(line)
line, words_in_line

word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}
len(word2index_dict), word2index_dict['impossible']

word_t = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    print('{:2} {:4} {}'.format(i, word_index, word))

##----------------------------------
## bytes pair encoding

##----------------------------------
## embeddings (blue map)
# 100 个浮点数组成的向量确实可以表示大量的单词。诀窍在于找
# 到一种有效的方法，将单个单词映射到这个 100 维空间中，以便于后续学习，这就叫作“嵌入”。
# classical embeddings model has BERT, GPT-2, etc.
# word2vec module is used to train embeddings
