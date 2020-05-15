
import numpy as np
import torchvision
import torchvision.transforms as transforms


def read_glove_vecs(glove_file):
    with open(glove_file , 'r' , encoding='utf-8') as f:
        words = set()
        word2vec = {}
        for line in f:
            line = line.strip().split()
            words.add(line[0])
            word2vec[line[0]] = np.array(line[1:], dtype = np.float64)
    return words, word2vec

def initial(words, word2vec):
    #一句话固定50个单词 不足50个补0 超过50个截断
    sentence = []
    zero = [0 for i in range(50)]
    num = 0
    for word in words:
        if num == 50:
            break
        if word in word2vec.keys():
            sentence.append(word2vec[word])
            num += 1
    for i in range(50 - num):
        sentence.insert(0, zero)
    return sentence


def read_data(pos_file, neg_file, word2vec):
    pos_set = []
    with open(pos_file, 'r', encoding="windows-1252") as f:
        for line in f.readlines():
            words = line.split(' ')
            sentence = initial(words, word2vec)
            pos_set.append(sentence)

    neg_set = []
    with open(neg_file, 'r', encoding='windows-1252') as f:
        for line in f.readlines():
            words = line.split()
            sentence = initial(words, word2vec)
            neg_set.append(sentence)

    l = len(pos_set)
    train_set = pos_set[:- 1000] + neg_set[:- 1000]
    train_target = [1 for i in range(l - 1000)] + [0 for i in range(l - 1000)]
    test_set = pos_set[-1000:] + neg_set[-1000:]
    test_target = [1 for i in range(1000)] + [0 for i in range(1000)]
    return train_set, train_target, test_set, test_target

def shuffle(train_set, train_target):
    permutaion = np.random.permutation(len(train_target))
    set = np.zeros(np.shape(train_set))
    target = np.zeros(np.shape(train_target))
    print(np.shape(train_target))
    for i in range(len(train_target)):
        set[i] = np.array(train_set[permutaion[i]])
        target[i] = np.array(train_target[permutaion[i]])
    return set,target

def get_data():
    words, word2vec = read_glove_vecs('./glove.6B.50d.txt')
    train_set, train_target, test_set, test_target = read_data('rt-polarity.pos', 'rt-polarity.neg', word2vec)
    print(np.shape(train_set))
    train_set, train_target = shuffle(train_set, train_target)
    return train_set, train_target.reshape(train_target.shape[0]), np.array(test_set), np.array(test_target).reshape(2000)



if __name__ == '__main__':
    x,y,_,_ = get_data()
    print(np.shape(x))
    print(np.shape(y))