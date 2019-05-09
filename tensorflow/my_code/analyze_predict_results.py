#!-*-coding=utf-8-*-
import json
import time
from collections import Counter
"""
用这个程序，看一下预测结果长什么样子
"""

def select_para(sample):
    results = []
    for d_idx, doc in enumerate(sample['documents']):
        para_infos = []
        for para_tokens, p_tokens, title, title_tokens in zip(doc['segmented_paragraphs'],
                                                              doc['paragraphs'], doc['title'],
                                                              doc['segmented_title']):
            # for para_tokens in doc['segmented_paragraphs']:
            # get question
            question_tokens = sample['segmented_question']

            p_char_tokens = [i for i in p_tokens]
            title_char_tokens = [i for i in title]
            q_char_tokens = [i for i in sample['question']]

            # 这里改过了，和原版是不一样的，计算recall_wrt_question分数的方法被修改了
            common_with_question = Counter(para_tokens) & Counter(question_tokens)
            common_with_question_char = Counter(p_char_tokens) & Counter(q_char_tokens)
            title_common_with_question = Counter(title_tokens) & Counter(question_tokens)
            title_common_with_question_char = Counter(title_char_tokens) & Counter(q_char_tokens)

            correct_preds = sum(common_with_question.values())
            recall_eq = 0 if correct_preds == 0 else float(correct_preds) / len(question_tokens)

            correct_preds = sum(common_with_question_char.values())
            recall_eq_char = 0 if correct_preds == 0 else float(correct_preds) / len(q_char_tokens)

            correct_preds = sum(title_common_with_question.values())
            recall_tq = 0 if correct_preds == 0 else float(correct_preds) / len(question_tokens)

            correct_preds = sum(title_common_with_question_char.values())
            recall_tq_char = 0 if correct_preds == 0 else float(correct_preds) / len(q_char_tokens)

            # if correct_preds == 0:
            #     recall_wrt_question = 0
            # else:
            #     recall_wrt_question = float(correct_preds) / len(question_tokens)
            recall_wrt_question = float(recall_eq + recall_eq_char) / 2
            # recall_eq = 0 if correct_preds==0 else float(correct_preds) / len(question_tokens)

            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
        para_infos.sort(key=lambda x: (-x[1], x[2]))
        fake_passage_tokens = []
        for para_info in para_infos[:1]:
            fake_passage_tokens += para_info[0]
        results.append("".join(fake_passage_tokens))
    return results


def get_test_questions():
    questions = []
    paras_selecteds = []
    with open("../data/test1_preprocessed/test1set/search.test1.json_pe", "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            paras_selected = select_para(item)
            paras_selecteds.append(paras_selected)
            questions.append(item["question"])
    with open("../data/test1_preprocessed/test1set/zhidao.test1.json_pe", "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            questions.append(item["question"])
    return questions, paras_selecteds


def get_test_answer(path):
    answers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            answers.append(item["answers"])
    return answers


if __name__ == "__main__":
    questions, paras_selecteds = get_test_questions()
    path = "pe_vocabfalse_test"
    # path = "pe_order_test"
    answers = get_test_answer("results/{}.predicted.json".format(path))

    for question, answer, paras_selected in zip(questions, answers, paras_selecteds):
        print("question: ", question)
        print("answer: ", answer[0].replace("\n", ""))
        print("paras_selected: ", paras_selected)
        print()
