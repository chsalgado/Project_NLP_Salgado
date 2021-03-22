import nltk
import os
import sys
from nltk.corpus import reuters

nltk.download('reuters')

cats = reuters.categories()

fileIds = reuters.fileids()

basePath = sys.argv[1]

if not os.path.exists(os.path.join(basePath, 'training')):
    os.makedirs(os.path.join(basePath, 'training'))

if not os.path.exists(os.path.join(basePath, 'test')):
    os.makedirs(os.path.join(basePath, 'test'))

for fileId in fileIds:
    raw = reuters.raw(fileId)
    sentences = nltk.sent_tokenize(raw)

    lines = []
    for sentence in sentences:
        sentenceWords = nltk.word_tokenize(sentence)
        line = ' '.join(sentenceWords)
        lines.append(line)

    with open(os.path.join(basePath, fileId), "w") as file:
        file.writelines("%s\n" % l for l in lines)
