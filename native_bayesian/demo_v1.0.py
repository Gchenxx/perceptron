import numpy as np
'''
朴素贝叶斯_文本分类
资料：机器学习实战
思路：首先准备好训练数据，然后根据训练数据创建出一个单词表（去重操作），对创建好的单词表进行向量化，
      根据向量化的单词表，创建每一个文本中的词的向量化，循环求得P(w|1)、P(w|0)、P(c),使用求得的参数，
      对数据进行分类
'''

# 准备数据
def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


# 创建单词表
def create_vocablist(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)  # | 用于求两个集合的并集  此外也表示按位或运算
    return list(vocabset)


# 创建词向量
def setofwords2vec(vocablist, inputset):
    returnvec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            # 伯努利模型（不考虑词在文档中出现的次数）
            returnvec[vocablist.index(word)] = 1  # 词集模型 （set-of-words model）
            # returnvec[vocablist.index(word)] += 1   # 词袋模型 （bag-of-words model）
            # 比较结果得知：如果是词袋模型时，如果“好词”的个数多于“坏次”个数时，会判定为好文章
        else:
            print('the word : %s is not in my vocabulary!' % word)
    return returnvec


# 得到单词表的词向量
def getallwords2vec(listoposts, dataset):
    trainmat = []
    for postindoc in listoposts:
        trainmat.append(setofwords2vec(dataset, postindoc))
    return trainmat


# 根据单词表训练得到 P(w|0) P(w|1) P(1)
def trainnb0(trainmatrix, traincategory):
    numtraindocs = len(trainmatrix)
    numwords = len(trainmatrix[0])
    pabusive = sum(traincategory) / float(numtraindocs)
    # p0num = np.zeros(numwords)
    # p1num = np.zeros(numwords)
    # p0denom = 0.0
    # p1denom = 0.0      优化一：如果P(w0|1)、P(w1|1)、P(w2|1)中出现一个概率为0，整个最后的乘积就为0，因此将所有词出现的次数初始化为1，并将分母初始化为2
    p0num = np.ones(numwords)
    p1num = np.ones(numwords)
#不懂为什么设置为2.0？
    p0denom = 2.0
    p1denom = 2.0
    for i in range(numtraindocs):
        if traincategory[i] == 1:
            p1num += trainmatrix[i]
            p1denom += sum(trainmatrix[i])
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    # p1vect = p1num / p1denom
    # p0vect = p0num / p0denom     优化二：为了防止下溢出（太多很小的数相乘导致），使用log进行处理，log(a*b)=loga+logb
    #                                      使用log的原因： f(x)与f(logx) 在相同区域内同时增加或者减小，并且在相同点上取得机值
    # 贝叶斯假设：特征之间是独立的
    p1vect = np.log(p1num / p1denom)
    p0vect = np.log(p0num / p0denom)  # 条件概率--->想象那个图：分为两个箱子，在两个箱子中拿灰色的球
    return p0vect, p1vect, pabusive


# 分类器
def classifynb(vec2classify, p0vec, p1vec, pclass1):
    p1 = sum(vec2classify * p1vec) + np.log(pclass1)
    p0 = sum(vec2classify * p0vec) + np.log(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 测试函数
def testingnb():
    listoposts, listclasses = load_dataset()
    myvocablist = create_vocablist(listoposts)
    trainmat = getallwords2vec(listoposts, myvocablist)
    # 训练得到 p(0) p(1) p()
    p0v, p1v, pab = trainnb0(np.array(trainmat), np.array(listclasses))
    testentry = ['mr', 'mr', 'mr', 'mr', 'mr', 'garbage']
    thisdoc = np.array(setofwords2vec(myvocablist, testentry))
    print(testentry, 'classified as :', classifynb(thisdoc, p0v, p1v, pab))
    testentry = ['garbage', 'mr']
    thisdoc = np.array(setofwords2vec(myvocablist, testentry))
    print(testentry, 'classified as :', classifynb(thisdoc, p0v, p1v, pab))


if __name__ == '__main__':
    testingnb()
