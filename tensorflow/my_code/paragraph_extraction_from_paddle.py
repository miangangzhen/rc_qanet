#!/usr/bin/python
#-*- coding:utf-8 -*-
import os
import sys
from collections import Counter

if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json
import copy
from my_code.analyze_data import ROOT_PATH, TRAIN_SEARCH_PATH, TRAIN_ZD_PATH, DEV_SEARCH_PATH, DEV_ZD_PATH, TEST_SEARCH_PATH, TEST_ZD_PATH

def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_paragraph_score(sample):
    """
    For each paragraph, compute the f1 score compared with the question
    Args:
        sample: a sample in the dataset.
    Returns:
        None
    Raises:
        None
    """
    question = sample["segmented_question"]
    for doc in sample['documents']:
        doc['segmented_paragraphs_scores'] = []
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(question) > 0:
                related_score = metric_max_over_ground_truths(f1_score,
                        para_tokens,
                        question)
            else:
                related_score = 0.0
            doc['segmented_paragraphs_scores'].append(related_score)


def dup_remove(doc, mode):
    """
    For each document, remove the duplicated paragraphs
    Args:
        doc: a doc in the sample
    Returns:
        bool
    Raises:
        None
    """
    paragraphs_his = {}
    del_ids = []
    para_id = None
    if 'most_related_para' in doc:
        para_id = doc['most_related_para']
    doc['paragraphs_length'] = []
    for p_idx, (segmented_paragraph, paragraph_score) in \
        enumerate(zip(doc["segmented_paragraphs"], doc["segmented_paragraphs_scores"])):
        doc['paragraphs_length'].append(len(segmented_paragraph))
        paragraph = ''.join(segmented_paragraph)
        if paragraph in paragraphs_his:
            del_ids.append(p_idx)
            if p_idx == para_id:
                para_id = paragraphs_his[paragraph]
            continue
        paragraphs_his[paragraph] = p_idx
    # delete
    prev_del_num = 0
    del_num = 0
    for p_idx in del_ids:
        if mode in {"train", "dev"} and p_idx < para_id:
            prev_del_num += 1
        del doc["segmented_paragraphs"][p_idx - del_num]
        del doc["segmented_paragraphs_scores"][p_idx - del_num]
        del doc['paragraphs_length'][p_idx - del_num]
        del_num += 1
    if len(del_ids) != 0:
        if 'most_related_para' in doc:
            doc['most_related_para'] = para_id - prev_del_num
        doc['paragraphs'] = []
        for segmented_para in doc["segmented_paragraphs"]:
            paragraph = ''.join(segmented_para)
            doc['paragraphs'].append(paragraph)
        return True
    else:
        return False


def paragraph_selection(sample, mode):
    """
    For each document, select paragraphs that includes as much information as possible
    Args:
        sample: a sample in the dataset.
        mode: string of ("train", "dev", "test"), indicate the type of dataset to process.
    Returns:
        None
    Raises:
        None
    """
    # predefined maximum length of paragraph
    MAX_P_LEN = 800
    # predefined splitter
    splitter = u'<splitter>'
    # topN of related paragraph to choose
    topN = 3
    doc_id = None
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0:
        doc_id = sample['answer_docs'][0]
        if doc_id >= len(sample['documents']):
            # Data error, answer doc ID > number of documents, this sample
            # will be filtered by dataset.py
            return
    for d_idx, doc in enumerate(sample['documents']):
        if 'segmented_paragraphs_scores' not in doc:
            continue
        status = dup_remove(doc, mode=mode)
        segmented_title = doc["segmented_title"]
        title_len = len(segmented_title)
        para_id = None
        if doc_id is not None:
            para_id = sample['documents'][doc_id]['most_related_para']
        total_len = title_len + sum(doc['paragraphs_length'])
        # add splitter
        para_num = len(doc["segmented_paragraphs"])
        total_len += para_num
        if total_len <= MAX_P_LEN:
            incre_len = title_len
            total_segmented_content = copy.deepcopy(segmented_title)
            for p_idx, segmented_para in enumerate(doc["segmented_paragraphs"]):
                if doc_id == d_idx and para_id > p_idx:
                    incre_len += len([splitter] + segmented_para)
                if doc_id == d_idx and para_id == p_idx:
                    incre_len += 1
                total_segmented_content += [splitter] + segmented_para
            if doc_id == d_idx:
                answer_start = incre_len + sample['answer_spans'][0][0]
                answer_end = incre_len + sample['answer_spans'][0][1]
                sample['answer_spans'][0][0] = answer_start
                sample['answer_spans'][0][1] = answer_end
            doc["segmented_paragraphs"] = [total_segmented_content]
            doc["segmented_paragraphs_scores"] = [1.0]
            doc['paragraphs_length'] = [total_len]
            doc['paragraphs'] = [''.join(total_segmented_content)]
            doc['most_related_para'] = 0
            continue
        # find topN paragraph id
        para_infos = []
        for p_idx, (para_tokens, para_scores) in \
                enumerate(zip(doc['segmented_paragraphs'], doc['segmented_paragraphs_scores'])):
            para_infos.append((para_tokens, para_scores, len(para_tokens), p_idx))
        para_infos.sort(key=lambda x: (-x[1], x[2]))
        topN_idx = []
        for para_info in para_infos[:topN]:
            topN_idx.append(para_info[-1])
        final_idx = []
        total_len = title_len
        if doc_id == d_idx:
            if mode == "train":
                final_idx.append(para_id)
                total_len = title_len + 1 + doc['paragraphs_length'][para_id]
        for id in topN_idx:
            if total_len > MAX_P_LEN:
                break
            if doc_id == d_idx and id == para_id and mode == "train":
                continue
            total_len += 1 + doc['paragraphs_length'][id] 
            final_idx.append(id)
        total_segmented_content = copy.deepcopy(segmented_title)
        final_idx.sort()
        incre_len = title_len
        for id in final_idx:
            if doc_id == d_idx and id < para_id:
                incre_len += 1 + doc['paragraphs_length'][id]
            if doc_id == d_idx and id == para_id:
                incre_len += 1
            total_segmented_content += [splitter] + doc['segmented_paragraphs'][id]
        if doc_id == d_idx:
            answer_start = incre_len + sample['answer_spans'][0][0]
            answer_end = incre_len + sample['answer_spans'][0][1]
            sample['answer_spans'][0][0] = answer_start
            sample['answer_spans'][0][1] = answer_end
        doc["segmented_paragraphs"] = [total_segmented_content]
        doc["segmented_paragraphs_scores"] = [1.0]
        doc['paragraphs_length'] = [total_len]
        doc['paragraphs'] = [''.join(total_segmented_content)]
        doc['most_related_para'] = 0


def test(line, mode):
    sample = json.loads(line, encoding='utf8')
    compute_paragraph_score(sample)
    paragraph_selection(sample, mode)
    print(sample)

if __name__ == "__main__":
    # train_search = r"""{"documents": [{"is_selected": true, "title": "武松怎么死的", "most_related_para": 1, "segmented_title": ["武松", "怎么", "死", "的"], "segmented_paragraphs": [["打", "方腊", "后", "班", "师", "回京", "时", "，", "在", "六合寺", "出家", "，", "八", "十", "岁", "圆寂", "。"], ["武松", "从小", "父母", "双", "亡", "，", "由", "兄长", "武大郎", "抚养", "长大", "。", "武松", "自", "小", "习", "武", "，", "武艺高强", "，", "性格", "急", "侠", "好", "义", "。", "一次", "醉酒", "后", "，", "在", "阳谷县", "（", "今", "聊城市", "阳谷县", "）", "景阳冈", "打死", "一只", "猛虎", "，", "因此", "被", "阳谷县", "令", "任命", "为", "都", "头", "。", "武松", "兄长", "武大郎", "是", "一", "个", "侏儒", "，", "其", "美貌", "妻子", "潘金莲", "试图", "勾引", "武松", "，", "被", "拒绝", "，", "后", "被", "当地", "富户", "西门庆", "勾引", "，", "奸情", "败露", "后", "，", "两人", "毒死", "了", "武大郎", "。", "为", "报仇", "，", "武松", "先", "杀", "潘金莲", "再", "杀", "西门庆", "，", "因此", "获", "罪", "被", "流放", "孟州", "。", "在", "去", "孟州", "途中", "，", "在", "十字坡", "酒店", "结识", "了", "张青", "孙二娘", "；", "在", "孟州", "，", "武松", "受到", "施恩", "的", "照顾", "，", "为", "报恩", "，", "武松", "醉打蒋门神", "，", "帮助", "施恩", "夺回", "了", "“", "快活林", "”", "酒店", "。", "不过", "武松", "也", "因此", "遭到", "蒋门神", "勾结", "官府", "以及", "张团练", "的", "暗算", "，", "被迫", "大开杀戒", "，", "血溅鸳鸯楼", "，", "并", "书", "“", "杀人者", "打虎", "武松", "也", "”", "。", "之后", "，", "夜", "走", "蜈蚣岭", "，", "在", "坟", "庵", "杀死", "恶", "道", "飞天蜈蚣", "王道", "人", "。", "在", "逃亡", "过程", "中", "，", "得", "张青", "、", "孙二娘", "夫妇", "帮助", "，", "假扮", "成", "带", "发", "修行", "的", "“", "行者", "”", "。", "武松", "投奔", "二龙山", "后", "成为", "该", "支", "“", "义", "军", "”", "的", "三", "位", "主要", "头领", "之", "一", "，", "后", "三山", "打", "青州", "时", "归依", "梁山", "。", "在", "征讨", "方腊", "战斗", "中", "，", "武松", "被", "包道乙", "暗算", "失去", "左臂", "。", "后", "班", "师", "时", "武松", "拒绝", "回", "汴京", "，", "在", "六合寺", "出家", "，", "八", "十", "岁", "圆寂", "。"], ["争议", "怎么", "死", ".", ".", ".", "被", "朝廷", "害死", "了", ".", ".", ".", "武松", "排行", "第", "二", "，", "江湖", "上", "人称", "武二郎", ",", "清河县", "人", "。", "景阳冈", "借", "着", "酒", "劲", "打死老虎", "，", "威震天下", "，", "做", "了", "阳谷县", "步兵", "都", "头", "。", "哥哥", "武大郎", "被", "奸夫", "淫妇", "潘金莲", ".", ".", "."], ["争议", "因为", "他", "杀", "了", "蔡某", "（", "一", "个", "官", "）", "，", "惨死", "在", "地牢", "里"], ["争议", "被", "朝廷", "逼死", "的"], ["争议", "2010", "年", "最火爆", "的", "武侠", "网游", "《", "醉逍遥", "》"], ["内置", "跨", "地图", "自动", "寻路", "，", "只要", "轻点鼠标", "，", "就", "可以", "轻松", "做", "任务", "刷", "怪"], ["PAER"], ["装备", "、", "宠物", "都是", "游戏", "产", "出", "，", "是", "一款", "平民化", "的", "免费", "网游", "，", "很", "适合", "学生", "上班族", "玩"]], "paragraphs": ["打方腊后班师回京时，在六合寺出家，八十岁圆寂。", "武松从小父母双亡，由兄长武大郎抚养长大。武松自小习武，武艺高强，性格急侠好义。一次醉酒后，在阳谷县（今聊城市阳谷县）景阳冈打死一只猛虎，因此被阳谷县令任命为都头。武松兄长武大郎是一个侏儒，其美貌妻子潘金莲试图勾引武松，被拒绝，后被当地富户西门庆勾引，奸情败露后，两人毒死了武大郎。为报仇，武松先杀潘金莲再杀西门庆，因此获罪被流放孟州。在去孟州途中，在十字坡酒店结识了张青孙二娘；在孟州，武松受到施恩的照顾，为报恩，武松醉打蒋门神，帮助施恩夺回了“快活林”酒店。不过武松也因此遭到蒋门神勾结官府以及张团练的暗算，被迫大开杀戒，血溅鸳鸯楼，并书“杀人者打虎武松也”。之后，夜走蜈蚣岭，在坟庵杀死恶道飞天蜈蚣王道人。在逃亡过程中，得张青、孙二娘夫妇帮助，假扮成带发修行的“行者”。武松投奔二龙山后成为该支“义军”的三位主要头领之一，后三山打青州时归依梁山。   在征讨方腊战斗中，武松被包道乙暗算失去左臂。后班师时武松拒绝回汴京，在六合寺出家，八十岁圆寂。", "争议怎么死 ...被朝廷害死了 ...武松排行第二，江湖上人称武二郎,清河县人。景阳冈借着酒劲打死老虎，威震天下，做了阳谷县步兵都头。哥哥武大郎被奸夫淫妇潘金莲...", "争议因为他杀了蔡某（一个官），惨死在地牢里", "争议被朝廷逼死的", "争议2010年最火爆的武侠网游《醉逍遥》", "内置跨地图自动寻路，只要轻点鼠标，就可以轻松做任务刷怪", "PAER", "装备、宠物都是游戏产出，是一款平民化的免费网游，很适合学生上班族玩"]}, {"is_selected": false, "title": "历史上真实的武松是怎么死的？是否如水浒传中的那样？ ", "most_related_para": 0, "segmented_title": ["历史", "上", "真实", "的", "武松", "是", "怎么", "死", "的", "？", "是否", "如", "水浒传", "中", "的", "那样", "？"], "segmented_paragraphs": [["武松", "是", "施耐庵", "所作", "古典名著", "《", "水浒传", "》", "中", "的", "重要", "人物", "。", "因", "其", "排行", "在", "二", "，", "又", "叫", "“", "武二郎", "”", "。", "血溅鸳鸯楼", "后", "，", "为", "躲避", "官府", "抓捕", "，", "改", "作", "头陀", "打扮", "，", "江湖", "人称", "“", "行者", "武松", "”", "。", "武松", "曾经", "在", "景阳冈", "上", "空", "手", "打死", "一只", "吊睛", "白", "额", "虎", "，", "“", "武松打虎", "”", "的", "事迹", "在", "后世", "广为流传", "。", "武松", "最终", "在", "征", "方腊", "过程", "中", "被", "飞刀", "所", "伤", "，", "痛", "失", "左臂", "，", "最后", "在", "六", "和", "寺", "病逝", "，", "寿", "至", "八", "十", "。", "武松", "是", "水浒", "中", "得", "善终", "的", "少数", "人", "之", "一", "。"], ["《", "水浒传", "》", "中", "武松", "是", "书", "中", "不", "多", "的", "被", "作者", "推崇", "的", "人物", "，", "一个人", "就", "写", "了", "十", "章", "，", "这", "在", "《", "水浒传", "》", "是", "不", "多", "的", "，", "上", "梁山", "前", "的", "“", "景阳冈武松打虎", "”", "、", "“", "供", "人头", "武二郎", "设", "祭", "”", "、", "“", "武松", "醉打蒋门神", "”", "、", "“", "武松", "大", "闹", "飞云", "浦", "”", "、", "“", "张都监", "血溅鸳鸯楼", "，", "武行者", "夜", "走", "蜈蚣岭", "”", "…", "…", "哪", "一", "段", "故事", "都是", "脍炙人口", "的", "。", "尤其", "是", "后人", "附", "会", "的", "“", "单臂", "擒", "方腊", "”", "更", "使得", "英雄", "武松", "名", "满", "华夏", "。", "武松", "也", "被", "金圣叹", "推介", "为", "“", "天人", "\"", "，", "武松", "也", "成", "了", "山东", "好汉", "的", "代名词", "。"], ["其实", "，", "《", "水浒传", "》", "它", "不是", "一", "部", "历史", "小说", "，", "而是", "一", "部", "虚构", "的", "演义", "小说", "。", "所谓", "“", "演义", "小说", "”", "，", "指", "的", "是", "小说", "中", "的", "部分", "人物", "和", "故事", "，", "历史", "上", "的", "确", "有", "过", "，", "但", "有", "相当一部分", "甚至", "大部分", "却", "是", "编造", "的", "。", "借助", "文学", "艺术", "的", "力量", "，", "梁山泊", "一百单八将", "的", "故事", "在", "中国", "早", "已经", "家喻户晓", "，", "老", "幼", "皆知", "了", "。", "但是", "梁山", "一百单八将", "大部分", "人物", "都是", "创作", "出来", "的", "，", "纯属", "子虚乌有", "。"], ["武松", "怎么", "死", "的", "?"], ["那么", "你", "知道", "历史", "上", "的", "武松", "失", "怎么", "死", "的", "吗", "?", "武松", "为何", "会", "惨遭", "重", "刑", "死于", "狱", "中", "呢", "?", "以下", "文章", "将", "为您", "揭晓", "。", "历史", "上", "确", "有", "武松", "其", "人", "。", "不过", "，", "历史", "上", "真实", "的", "武松", "仅仅", "是", "宋江", "部", "下", "的", "一", "个", "普通", "头领", "而已", "，", "至于", "小说", "中", "什么", "景阳岗", "打虎", "、", "杀", "西门庆", "等", "情节", "，", "显然", "都", "出于", "艺术创作", "，", "真实", "的", "武松", "和", "小说", "中", "所", "写", "的", "武松", "完全", "不同", "。"], ["《", "临安县志", "》", "、", "《", "西湖", "大", "观", "》", "、", "《", "杭州府志", "》", "、", "《", "浙江通志", "》", "等", "史籍", "都", "记载", "了", "北宋", "时", "杭州", "知府", "中", "的", "提辖", "武松", "勇于", "为民", "除", "恶", "的", "侠义", "壮举", "。", "据", "记载", "，", "杭州", "知府", "高权", "见", "武松", "武艺高强", "，", "人才出众", "，", "遂", "邀请", "入", "府", "，", "让", "他", "充当", "都", "头", "。"], ["不", "久", "，", "因", "功", "高", "被", "提", "为", "提辖", "，", "成为", "知府", "高权", "的", "心腹", "。", "后来", "高权", "因", "得罪", "权贵", "，", "被奸", "人", "诬", "谄", "而", "罢官", "。", "武松", "也", "因此", "受到", "牵连", "，", "被", "赶出", "衙门", "。", "继任", "的", "新", "知府", "是", "太师", "蔡京", "的", "儿子", "蔡鋆", "，", "是", "个", "大", "奸臣", "。", "他", "倚仗", "其", "父", "的", "权势", "，", "在", "杭州", "任", "上", "虐", "政", "殃", "民", "，", "百姓", "怨声载道", "，", "人称", "蔡鋆", "为", "“", "蔡虎", "”", "。", "武松", "对", "这个", "奸臣", "恨之入骨", "，", "决心", "拼", "上", "性", "命", "也", "要", "为民除害", "。"], ["一", "日", "，", "他", "身", "藏", "利刃", "，", "隐匿", "在", "蔡府", "之", "前", "，", "等", "蔡", "前呼后拥", "而来", "之", "际", "，", "箭", "一般", "冲", "上", "前", "去", "，", "向", "蔡鋆", "猛", "刺", "数", "刀", "，", "当", "即", "结果", "了", "他", "的", "性", "命", "。", "官兵", "蜂拥", "前", "来", "围攻", "武松", "，", "武松", "终", "因", "寡不敌众", "被", "官兵", "捕获", "。", "后", "惨遭", "重", "刑", "死于", "狱", "中", "。", "当地", "“", "百姓", "深", "感", "其", "德", "，", "葬于", "杭州", "西泠桥", "畔", "”", "，", "后人", "立碑", "，", "题", "曰", "“", "宋", "义士", "武松", "之", "墓", "”", "。"], ["这", "段", "真实", "的", "记载", "，", "想", "必", "作者", "施耐庵", "肯定", "是", "知道", "的", "，", "将", "其中", "的", "几个", "细节", "充分", "渲染", "，", "便", "成", "了", "小说", "中", "的", "武松", "。", "至于", "武松", "的", "最后", "结局", "，", "《", "水浒传", "》", "中", "写到", "他", "成", "了", "清", "忠", "祖师", "，", "得", "享", "天", "年", "，", "实在", "是", "一", "种", "符合", "老百姓", "心愿", "的", "美好", "的", "艺术", "处理", "。"]], "paragraphs": ["武松是施耐庵所作古典名著《水浒传》中的重要人物。因其排行在二，又叫“武二郎”。血溅鸳鸯楼后，为躲避官府抓捕，改作头陀打扮，江湖人称“行者武松”。武松曾经在景阳冈上空手打死一只吊睛白额虎，“武松打虎”的事迹在后世广为流传。武松最终在征方腊过程中被飞刀所伤，痛失左臂，最后在六和寺病逝，寿至八十。武松是水浒中得善终的少数人之一。", "《水浒传》中武松是书中不多的被作者推崇的人物，一个人就写了十章，这在《水浒传》是不多的，上梁山前的“景阳冈武松打虎”、“供人头武二郎设祭”、“武松醉打蒋门神”、“武松大闹飞云浦”、“张都监血溅鸳鸯楼，武行者夜走蜈蚣岭”……哪一段故事都是脍炙人口的。尤其是后人附会的“单臂擒方腊”更使得英雄武松名满华夏。武松也被金圣叹推介为“天人\"，武松也成了山东好汉的代名词。", "其实，《水浒传》它不是一部历史小说，而是一部虚构的演义小说。所谓“演义小说”，指的是小说中的部分人物和故事，历史上的确有过，但有相当一部分甚至大部分却是编造的。借助文学艺术的力量，梁山泊一百单八将的故事在中国早已经家喻户晓，老幼皆知了。但是梁山一百单八将大部分人物都是创作出来的，纯属子虚乌有。", "武松怎么死的?", "那么你知道历史上的武松失怎么死的吗?武松为何会惨遭重刑死于狱中呢?以下文章将为您揭晓。历史上确有武松其人。不过，历史上真实的武松仅仅是宋江部下的一个普通头领而已，至于小说中什么景阳岗打虎、杀西门庆等情节，显然都出于艺术创作，真实的武松和小说中所写的武松完全不同。", "《临安县志》、《西湖大观》、《杭州府志》、《浙江通志》等史籍都记载了北宋时杭州知府中的提辖武松勇于为民除恶的侠义壮举。据记载，杭州知府高权见武松武艺高强，人才出众，遂邀请入府，让他充当都头。", "不久，因功高被提为提辖，成为知府高权的心腹。后来高权因得罪权贵，被奸人诬谄而罢官。武松也因此受到牵连，被赶出衙门。继任的新知府是太师蔡京的儿子蔡鋆，是个大奸臣。他倚仗其父的权势，在杭州任上虐政殃民，百姓怨声载道，人称蔡鋆为“蔡虎”。武松对这个奸臣恨之入骨，决心拼上性命也要为民除害。", "一日，他身藏利刃，隐匿在蔡府之前，等蔡前呼后拥而来之际，箭一般冲上前去，向蔡鋆猛刺数刀，当即结果了他的性命。官兵蜂拥前来围攻武松，武松终因寡不敌众被官兵捕获。后惨遭重刑死于狱中。当地“百姓深感其德，葬于杭州西泠桥畔”，后人立碑，题曰“宋义士武松之墓”。", "这段真实的记载，想必作者施耐庵肯定是知道的，将其中的几个细节充分渲染，便成了小说中的武松。至于武松的最后结局，《水浒传》中写到他成了清忠祖师，得享天年，实在是一种符合老百姓心愿的美好的艺术处理。"]}, {"is_selected": false, "title": "> 武松的故事 ", "most_related_para": 0, "segmented_title": [">", "武松", "的", "故事"], "segmented_paragraphs": [["我们", "都", "知道", "武松", "之所以", "会", "上", "了", "梁山", "，", "是因为", "血溅鸳鸯楼", "，", "犯", "下", "了", "大事", "儿", "，", "无处可去", "，", "最后", "只得", "上", "了", "梁山", "落草为寇", "。", "而", "为什么", "会", "干", "下", "血溅鸳鸯楼", "之", "事", "呢", "？", "那", "是因为", "被", "刺", "配", "孟州", "后", "受到", "施恩", "的", "照顾", "，", "为了", "报恩", "，", "武松", "才", "犯", "下", "了", "错", "。", "而", "为什么", "会", "被", "刺", "配", "孟", "…", "…", "["]], "paragraphs": ["我们都知道武松之所以会上了梁山，是因为血溅鸳鸯楼，犯下了大事儿，无处可去，最后只得上了梁山落草为寇。而为什么会干下血溅鸳鸯楼之事呢？那是因为被刺配孟州后受到施恩的照顾，为了报恩，武松才犯下了错。而为什么会被刺配孟……["]}, {"is_selected": false, "title": "历史上真实的武松是怎么死的？ ", "most_related_para": 2, "segmented_title": ["历史", "上", "真实", "的", "武松", "是", "怎么", "死", "的", "？"], "segmented_paragraphs": [["武松", "自", "小", "在", "今", "为", "河北省", "邢台市", "清河县", "长大", "，", "拜", "当时", "江湖", "上", "鼎鼎大名", "的", "武林高手", "，", "少林派", "武", "师", "谭正芳", "最小", "的", "徒弟", "，", "陕西", "大侠", "铁臂", "膀", "周侗", "为师", "，", "学", "得", "一", "身", "好", "武艺", "。", "按", "这", "一", "史实", "，", "武松", "应", "为", "同", "出", "周门", "的", "民族英雄", "岳飞", "的", "师兄", "。", "我们", "知道", "岳飞", "就是", "周侗", "的", "弟子", "，", "只是", "那时", "周侗", "年事已高", "。"], ["武松", "学", "得", "惊", "人艺", "之后", "，", "又", "结识", "了", "同样", "是", "武林高手", "的", "宋江", "。", "不要", "以为", "宋江", "武功", "平庸", "，", "象", "小说", "《", "水浒传", "》", "中", "描述", "的", "那样", "，", "只", "会", "一些", "平常", "功夫", "，", "使", "枪", "弄", "棒", "的", "，", "并", "不", "突出", "。", "真实", "的", "宋江", "可", "不是", "等闲之辈", "，", "他", "武功", "高超", "，", "智谋", "过人", "，", "“", "以", "三", "十", "六", "人", "横行", "齐", "魏", "，", "官军", "数", "万", "，", "无", "敢", "抗", "者", "”", "，", "后", "为", "朝廷", "招安", "。", "宋江", "与", "圆通", "法师", "共创", "武术", "门派", "—", "—", "子午", "门", "。", "这", "门", "功夫", "因", "注重", "天地", "人气", "融会贯通", "，", "多", "在", "子时", "、", "午时", "习", "练", "而", "得", "名", "。", "它", "非常", "适合", "山东", "大汉", "，", "讲究", "拳脚", "大开大合", "，", "出击", "勇猛", "。", "武松", "结识", "宋江", "后", "，", "学", "得", "子午", "门", "功夫", "，", "被", "立", "为", "子午", "门", "第一代", "掌门人", "，", "宋江", "与", "圆通", "法师", "被", "尊", "为", "子午", "门", "始祖", "，", "现在", "山东", "东平湖", "一", "带", "仍", "有", "子午", "门", "传", "人", "。"], ["武松", "学", "得", "一", "身", "好", "武艺", "后", "，", "来", "到", "杭州", "，", "经常", "在", "涌金门", "外", "，", "靠", "卖艺", "为", "生", "，", "当时", "的", "知府", "高权", "见", "其", "武艺高强", "，", "人才出众", "，", "就", "招收", "为", "都", "头", "。", "因", "工作", "勤勉", "，", "立功", "不", "少", "，", "不", "久", "，", "被", "提升", "为", "提辖", "。", "可是", "天有不测风云", "，", "在", "当时", "黑暗", "的", "政治", "斗争", "中", "，", "高权", "被", "罢官", "，", "武松", "也", "受到", "牵连", "，", "被逐出", "衙门", "。", "新任", "知府", "是", "当朝", "大", "奸臣", "蔡京", "之", "子", "蔡鋆", "，", "专", "事", "欺压", "百姓", "，", "人们", "怨声载道", "，", "称", "其", "为", "蔡虎", "。", "武松", "对此", "恶", "官", "恨之入骨", "，", "下定决心", "要", "为民除害", "，", "以", "一", "己", "之", "力", "扫除", "大宋", "天空", "恼", "人", "阴霾", "。", "一次", "，", "在", "蔡虎", "出行", "时", "，", "武松", "持", "利刃", "斜刺", "里", "冲", "上去", "，", "对", "蔡虎", "连", "刺", "数", "刀", "，", "使", "其", "当场", "毙命", "。", "但", "其", "护卫", "人员", "一拥而上", "，", "和", "武松", "拼杀", "，", "武松", "终", "因", "寡不敌众", "，", "被", "擒获", "。", "后", "惨死", "狱", "中", "。", "当地", "百姓", "深", "感", "其", "德", "，", "葬于", "杭州", "西泠桥", "畔", "。", "后人", "立碑", "：", "“", "宋", "义士", "武松", "之", "墓", "”", "。"]], "paragraphs": ["武松自小在今为河北省邢台市清河县长大，拜当时江湖上鼎鼎大名的武林高手，少林派武师谭正芳最小的徒弟，陕西大侠铁臂膀周侗为师，学得一身好武艺。按这一史实，武松应为同出周门的民族英雄岳飞的师兄。我们知道岳飞就是周侗的弟子，只是那时周侗年事已高。", "武松学得惊人艺之后，又结识了同样是武林高手的宋江。不要以为宋江武功平庸，象小说《水浒传》中描述的那样，只会一些平常功夫，使枪弄棒的，并不突出。真实的宋江可不是等闲之辈，他武功高超，智谋过人，“以三十六人横行齐魏，官军数万，无敢抗者”，后为朝廷招安。宋江与圆通法师共创武术门派——子午门。这门功夫因注重天地人气融会贯通，多在子时、午时习练而得名。它非常适合山东大汉，讲究拳脚大开大合，出击勇猛。武松结识宋江后，学得子午门功夫，被立为子午门第一代掌门人，宋江与圆通法师被尊为子午门始祖，现在山东东平湖一带仍有子午门传人。", "武松学得一身好武艺后，来到杭州，经常在涌金门外，靠卖艺为生，当时的知府高权见其武艺高强，人才出众，就招收为都头。因工作勤勉，立功不少，不久，被提升为提辖。可是天有不测风云，在当时黑暗的政治斗争中，高权被罢官，武松也受到牵连，被逐出衙门。新任知府是当朝大奸臣蔡京之子蔡鋆，专事欺压百姓，人们怨声载道，称其为蔡虎。武松对此恶官恨之入骨，下定决心要为民除害，以一己之力扫除大宋天空恼人阴霾。一次，在蔡虎出行时，武松持利刃斜刺里冲上去，对蔡虎连刺数刀，使其当场毙命。但其护卫人员一拥而上，和武松拼杀，武松终因寡不敌众，被擒获。后惨死狱中。当地百姓深感其德，葬于杭州西泠桥畔。后人立碑：“宋义士武松之墓”。"]}, {"is_selected": false, "title": "武松怎么死的武松怎么死的", "most_related_para": 0, "segmented_title": ["武松", "怎么", "死", "的", "武松", "怎么", "死", "的"], "segmented_paragraphs": [["武松", "从小", "父母", "双", "亡", "，", "由", "兄长", "武大郎", "抚养", "长大", "。", "武松", "自", "小", "习", "武", "，", "武艺高强", "，", "性格", "急", "侠", "好", "义", "。", "一次", "醉酒", "后", "，", "在", "阳谷县", "（", "今", "聊城市", "阳谷县", "）", "景阳冈", "赤手空拳", "打死", "一只", "猛虎", "，", "因此", "被", "阳谷县", "令", "任命", "为", "都", "头", "。", "武松", "兄长", "武大郎", "是", "一", "个", "侏儒", "，", "其", "美貌", "妻子", "潘金莲", "试图", "勾引", "武松", "，", "被", "拒绝", "，", "后", "被", "当地", "富户", "西门庆", "勾引", "，", "奸情", "败露", "后", "，", "两人", "毒死", "了", "武大郎", "。", "为", "报仇", "，", "武松", "先", "杀", "潘金莲", "再", "杀", "西门庆", "，", "因此", "获", "罪", "被", "流放", "孟州", "。", "在", "去", "孟州", "途中", "，", "在", "十字坡", "酒店", "结识", "了", "张青", "孙二娘", "；", "在", "孟州", "，", "武松", "受到", "施恩", "的", "照顾", "，", "为", "报恩", "，", "武松", "醉打蒋门神", "，", "帮助", "施恩", "夺回", "了", "“", "快活林", "”", "酒店", "。", "不过", "武松", "也", "因此", "遭到", "蒋门神", "勾结", "官府", "以及", "张团练", "的", "暗算", "，", "被迫", "大开杀戒", "，", "血溅鸳鸯楼", "，", "并", "书", "“", "杀人者", "打虎", "武松", "也", "”", "。", "在", "逃亡", "过程", "中", "，", "得", "张青", "、", "孙二娘", "夫妇", "帮助", "，", "假扮", "成", "带", "发", "修行", "的", "“", "行者", "”", "。", "武松", "投奔", "二龙山", "后", "成为", "该", "支", "“", "义", "军", "”", "的", "三", "位", "主要", "头领", "之", "一", "，", "后", "三山", "打", "青州", "时", "归依", "梁山", "。", "在", "征讨", "方腊", "的", "战斗", "中", "，", "武松", "被", "包道乙", "暗算", "失去", "左臂", "。", "后", "班", "师", "时", "武松", "拒绝", "回", "汴京", "，", "在", "六合寺", "出家", "，", "八", "十", "岁", "圆寂", "。"]], "paragraphs": ["武松从小父母双亡，由兄长武大郎抚养长大。武松自小习武，武艺高强，性格急侠好义。一次醉酒后，在阳谷县（今聊城市阳谷县）景阳冈赤手空拳打死一只猛虎，因此被阳谷县令任命为都头。武松兄长武大郎是一个侏儒，其美貌妻子潘金莲试图勾引武松，被拒绝，后被当地富户西门庆勾引，奸情败露后，两人毒死了武大郎。为报仇，武松先杀潘金莲再杀西门庆，因此获罪被流放孟州。在去孟州途中，在十字坡酒店结识了张青孙二娘；在孟州，武松受到施恩的照顾，为报恩，武松醉打蒋门神，帮助施恩夺回了“快活林”酒店。不过武松也因此遭到蒋门神勾结官府以及张团练的暗算，被迫大开杀戒，血溅鸳鸯楼，并书“杀人者打虎武松也”。在逃亡过程中，得张青、孙二娘夫妇帮助，假扮成带发修行的“行者”。武松投奔二龙山后成为该支“义军”的三位主要头领之一，后三山打青州时归依梁山。 在征讨方腊的战斗中，武松被包道乙暗算失去左臂。后班师时武松拒绝回汴京，在六合寺出家，八十岁圆寂。"]}], "answer_spans": [[13, 263]], "question_id": 338214, "fake_answers": ["自小习武，武艺高强，性格急侠好义。一次醉酒后，在阳谷县（今聊城市阳谷县）景阳冈打死一只猛虎，因此被阳谷县令任命为都头。武松兄长武大郎是一个侏儒，其美貌妻子潘金莲试图勾引武松，被拒绝，后被当地富户西门庆勾引，奸情败露后，两人毒死了武大郎。为报仇，武松先杀潘金莲再杀西门庆，因此获罪被流放孟州。在去孟州途中，在十字坡酒店结识了张青孙二娘；在孟州，武松受到施恩的照顾，为报恩，武松醉打蒋门神，帮助施恩夺回了“快活林”酒店。不过武松也因此遭到蒋门神勾结官府以及张团练的暗算，被迫大开杀戒，血溅鸳鸯楼，并书“杀人者打虎武松也”。之后，夜走蜈蚣岭，在坟庵杀死恶道飞天蜈蚣王道人。在逃亡过程中，得张青、孙二娘夫妇帮助，假扮成带发修行的“行者”。武松投奔二龙山后成为该支“义军”的三位主要头领之一，后三山打青州时归依梁山。在征讨方腊战斗中，武松被包道乙暗算失去左臂。后班师时武松拒绝回汴京，在六合寺出家，八十岁圆寂。"], "question": "武松怎么死的", "segmented_answers": [["武松", "自", "小", "习", "武", "，", "武艺高强", "，", "性格", "急", "侠", "好", "义", "。", "一次", "醉酒", "后", "，", "在", "阳谷县", "景阳冈", "打死", "一只", "猛虎", "，", "因此", "被", "阳谷县", "令", "任命", "为", "都", "头", "。", "武松", "兄长", "武大郎", "是", "一", "个", "侏儒", "，", "其", "美貌", "妻子", "潘金莲", "试图", "勾引", "武松", "，", "被", "拒绝", "，", "后", "被", "当地", "富户", "西门庆", "勾引", "，", "奸情", "败露", "后", "，", "两人", "毒死", "了", "武大郎", "。", "为", "报仇", "，", "武松", "先", "杀", "潘金莲", "再", "杀", "西门庆", "，", "因此", "获", "罪", "被", "流放", "孟州", "。", "在", "去", "孟州", "途中", "，", "在", "十字坡", "酒店", "结识", "了", "张青", "孙二娘", "；", "在", "孟州", "，", "武松", "受到", "施恩", "的", "照顾", "，", "为", "报恩", "，", "武松", "醉打蒋门神", "，", "帮助", "施恩", "夺回", "了", "“", "快活林", "”", "酒店", "。", "不过", "武松", "也", "因此", "遭到", "蒋门神", "勾结", "官府", "以及", "张团练", "的", "暗算", "，", "被迫", "大开杀戒", "，", "血溅鸳鸯楼", "。", "之后", "，", "在", "坟", "庵", "杀死", "恶", "道", "飞天蜈蚣", "王道", "人", "。", "在", "逃亡", "过程", "中", "，", "得", "张青", "、", "孙二娘", "夫妇", "帮助", "，", "假扮", "成", "带", "发", "修行", "的", "“", "行者", "”", "。", "武松", "投奔", "二龙山", "后", "成为", "该", "支", "“", "义", "军", "”", "的", "三", "位", "主要", "头领", "之", "一", "，", "后", "三山", "打", "青州", "时", "归依", "梁山", "。", "在", "征讨", "方腊", "战斗", "中", "，", "武松", "被", "包道乙", "暗算", "失去", "左臂", "。", "后", "班", "师", "时", "武松", "拒绝", "回", "汴京", "，", "在", "六合寺", "出家", "，", "八", "十", "岁", "圆寂", "。"]], "answers": ["武松自小习武，武艺高强，性格急侠好义。一次醉酒后，在阳谷县景阳冈打死一只猛虎，因此被阳谷县令任命为都头。武松兄长武大郎是一个侏儒，其美貌妻子潘金莲试图勾引武松，被拒绝，后被当地富户西门庆勾引，奸情败露后，两人毒死了武大郎。为报仇，武松先杀潘金莲再杀西门庆，因此获罪被流放孟州。在去孟州途中，在十字坡酒店结识了张青孙二娘；在孟州，武松受到施恩的照顾，为报恩，武松醉打蒋门神，帮助施恩夺回了“快活林”酒店。不过武松也因此遭到蒋门神勾结官府以及张团练的暗算，被迫大开杀戒，血溅鸳鸯楼。之后，在坟庵杀死恶道飞天蜈蚣王道人。在逃亡过程中，得张青、孙二娘夫妇帮助，假扮成带发修行的“行者”。武松投奔二龙山后成为该支“义军”的三位主要头领之一，后三山打青州时归依梁山。在征讨方腊战斗中，武松被包道乙暗算失去左臂。后班师时武松拒绝回汴京，在六合寺出家，八十岁圆寂。"], "answer_docs": [0], "segmented_question": ["武松", "怎么", "死", "的"], "question_type": "DESCRIPTION", "fact_or_opinion": "FACT", "match_scores": [0.9649484536082473]}"""
    # test(train_search, "train")

    # for mode, path in zip(["train", "train", "dev", "dev", "test", "test"], [TRAIN_SEARCH_PATH, TRAIN_ZD_PATH, DEV_SEARCH_PATH, DEV_ZD_PATH, TEST_SEARCH_PATH, TEST_ZD_PATH]):
    for mode, path in zip(
            ["test", "test", "test", "test"],
            ["../data/test2_preprocessed/test2set/search.test1.json", "../data/test2_preprocessed/test2set/search.test2.json",
             "../data/test2_preprocessed/test2set/zhidao.test1.json", "../data/test2_preprocessed/test2set/zhidao.test2.json"]):
        with open(os.path.join(ROOT_PATH, path+"_pe"), "w", encoding="utf-8") as fo:
            with open(os.path.join(ROOT_PATH, path), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == "":
                        continue
                    sample = json.loads(line, encoding='utf8')

                    compute_paragraph_score(sample)
                    paragraph_selection(sample, mode)
                    fo.write(json.dumps(sample, ensure_ascii=False)+"\n")
