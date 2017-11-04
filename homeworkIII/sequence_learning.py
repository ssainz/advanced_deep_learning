


def read_words(file_name):
    words = {}
    with open(file_name, 'rt') as f:
        for line in f:
            tokens = line.split(' ')
            for token in tokens:
                if token in words:
                    words[token] += 1
                else:
                    words[token] = 1

    return words


words = read_words("/Users/sergiosainz/Projects/vtech/DeepLearning/HomeworkIII/text_file.txt")

print(len(words.keys()))

