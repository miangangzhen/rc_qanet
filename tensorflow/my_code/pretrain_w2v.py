import gensim
import json


class MySentences(object):
    def __init__(self, dirnames):
        self.dirnames = dirnames

    def __iter__(self):
        for fname in self.dirnames:
            for line in open(fname, "r", encoding="utf-8"):
                obj = json.loads(line.strip())
                ques_sent = obj['segmented_question']
                yield ques_sent
                for doc in obj['documents']:
                    title_sent = doc['segmented_title']
                    yield title_sent
                    for p in doc['segmented_paragraphs']:
                        yield p


if __name__ == "__main__":
    root = '../../../../data/'
    s_train = root + 'train_preprocessed/trainset/search.train.json'
    z_train = root + 'train_preprocessed/trainset/zhidao.train.json'
    # s_dev = root +'devset/search.dev.json'
    # z_dev = root+'devset/zhidao.dev.json'
    # s_test = root +'testset/search.test.json'
    # z_test = root +'testset/zhidao.test.json'

    fs = [s_train, z_train] #, s_dev, z_dev, s_test, z_test]
    sents = MySentences(fs)

    model = gensim.models.Word2Vec(iter=10, min_count=10, size=100, workers=6, negative=8, window=5)
    model.build_vocab(sents)

    sents = MySentences(fs)
    model.train(sents, total_examples=model.corpus_count, epochs=model.epochs)

    # gzip.open(filename, mode='rb', compresslevel=9, encoding=None, errors=None, newline=None)
    # https://docs.python.org/3/library/gzip.html
    with open('./100_ver_not_pure.txt', 'w', encoding='utf-8') as f:
        for k in model.wv.vocab.keys():
            s = k + ' ' + ' '.join([str(i) for i in model.wv[k]])
            f.write(s + '\n')
