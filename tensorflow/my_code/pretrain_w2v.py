import gensim
import json
import tqdm
"""
预训练词向量
"""

class MySentences(object):
    def __init__(self, dirnames):
        self.dirnames = dirnames

    def __iter__(self):
        for fname in self.dirnames:
            print(fname)
            with open(fname, "r", encoding="utf-8") as f:
                for line in tqdm.tqdm(f.readlines()):
                    obj = json.loads(line.strip())
                    ques_sent = obj['segmented_question']
                    yield ques_sent
                    for doc in obj['documents']:
                        for p in doc['segmented_paragraphs']:
                            yield p


if __name__ == "__main__":
    root = '../data/'
    s_train = root + 'train_preprocessed/trainset/search.train.json_pe'
    z_train = root + 'train_preprocessed/trainset/zhidao.train.json_pe'
    s_dev = root + 'dev_preprocessed/devset/search.dev.json_pe'
    z_dev = root + 'dev_preprocessed/devset/zhidao.dev.json_pe'
    s_test = root +'test1_preprocessed/test1set/search.test1.json_pe'
    z_test = root +'test1_preprocessed/test1set/zhidao.test1.json_pe'

    fs = [s_train, z_train, s_dev, z_dev, s_test, z_test]
    # fs = [s_dev, z_test]

    sents = MySentences(fs)

    # mod 20190507
    # sg=1 skip-gram
    model = gensim.models.Word2Vec(iter=10, min_count=10, size=128, workers=6, negative=8, window=5, sg=1)
    model.build_vocab(sents)

    sents = MySentences(fs)
    model.train(sents, total_examples=model.corpus_count, epochs=model.epochs)

    with open('./w2v_pe.txt', 'w', encoding='utf-8') as f:
        for k in model.wv.vocab.keys():
            s = k + ' ' + ' '.join([str(i) for i in model.wv[k]])
            f.write(s + '\n')
