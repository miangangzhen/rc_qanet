#!-*-coding=utf-8-*-
from rc_model import RCModel
from run_colab import parse_args
import os
import pickle
import tensorflow as tf
import jieba
import json


def convert_to_ids(vocab, q, p):
    result = {}
    result["question_id"] = 1
    result['question_token_ids'] = vocab.convert_to_ids(jieba.lcut(q))
    # result['question_token_ids'] = vocab.convert_to_ids(q)

    p_tokens = jieba.lcut(p)
    # p_tokens = p
    result['passages'] = [{"passage_token_ids": vocab.convert_to_ids(p_tokens), "passage_tokens": p_tokens}]
    result["question_type"] = "DESCRIPTION"
    return result


def _dynamic_padding(args, batch_data, pad_id):
    """
    Dynamically pads the batch_data with pad_id
    """
    pad_p_len = args.max_p_len
    pad_q_len = args.max_q_len
    batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                       for ids in batch_data['passage_token_ids']]
    batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                        for ids in batch_data['question_token_ids']]
    return batch_data, pad_p_len, pad_q_len


def gen_mini_batches(sample, args, pad_id):
    batch_data = {'raw_data': [sample],
                  'question_token_ids': [],
                  'question_length': [],
                  'passage_token_ids': [],
                  'passage_length': [],
                  'start_id': [],
                  'end_id': []}
    max_passage_num = args.max_p_num
    for sidx, sample in enumerate(batch_data['raw_data']):
        for pidx in range(max_passage_num):
            if pidx < len(sample['passages']):
                batch_data['question_token_ids'].append(sample['question_token_ids'])
                batch_data['question_length'].append(min(len(sample['question_token_ids']), args.max_q_len))
                passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                batch_data['passage_token_ids'].append(passage_token_ids)
                batch_data['passage_length'].append(min(len(passage_token_ids), args.max_p_len))
            else:
                batch_data['question_token_ids'].append([])
                batch_data['question_length'].append(0)
                batch_data['passage_token_ids'].append([])
                batch_data['passage_length'].append(0)
    batch_data, padded_p_len, padded_q_len = _dynamic_padding(args, batch_data, pad_id)
    for _ in batch_data['raw_data']:
        batch_data['start_id'].append(0)
        batch_data['end_id'].append(0)
    return [batch_data]


class RCModelPredictor():
    def __init__(self, args):
        args.max_p_num = 1

        with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
            self.vocab = pickle.load(fin)

        self.rc_model = RCModel(self.vocab, args)
        if tf.gfile.Exists(args.model_dir):
            print("load model from {}".format(args.model_dir))
            self.rc_model.restore(model_dir=args.model_dir, model_prefix=None)

    def predict(self, question, passage):
        sample = convert_to_ids(self.vocab, question, passage)
        test_batches = gen_mini_batches(sample, args, pad_id=self.vocab.get_id(self.vocab.pad_token))
        self.rc_model.evaluate(test_batches,
                          result_dir=args.result_dir, result_prefix='test.predicted')

def predict_on_given_data(args, question, passage):
    args.max_p_num = 1

    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    sample = convert_to_ids(vocab, question, passage)
    rc_model = RCModel(vocab, args)
    if tf.gfile.Exists(args.model_dir):
        print("load model from {}".format(args.model_dir))
        rc_model.restore(model_dir=args.model_dir, model_prefix=None)
    test_batches = gen_mini_batches(sample, args, pad_id=vocab.get_id(vocab.pad_token))
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')

def show_answer(args):
    result_file = os.path.join(args.result_dir, "test.predicted.json")
    with open(result_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            d = json.loads(line)
            print(d["answers"])

if __name__ == "__main__":
    args = parse_args()
    args.max_p_len = 800
    args.model_dir = "models_pe_totaldata_49"
    args.result_dir = "./"
    args.vocab_dir = "resource/vocab_pe/"

    rcModelPredictor = RCModelPredictor(args)

    question = "这次事故造成多少伤亡"
    passage = "据新华社首尔5月13日电 （记者耿学鹏、 田明）韩国中部一家工厂13日发生爆炸，造成1人死亡、3人重伤。当地媒体援引消防部门披露的信息报道说，韩国忠清北道堤川市一家生产手机零部件的工厂在当地时间下午2时30分左右发生爆炸并起火。消防人员迅速赶至并控制了火势，伤者被送往附近医院治疗。事故中的死者为1名38岁的女性。3名伤者全身严重烧伤，年龄分别为56岁、49岁和46岁。 2019-05-14 00:00:00:0致1死3伤4790701韩一工厂爆炸/enpproperty"
    rcModelPredictor.predict(question, passage)
    show_answer(args)

    question = "是哪家公司的设备"
    passage = "原标题：三级医院清查自助机蓄电池安全　　本报讯(记者刘欢)一所市属医院用于挂号、缴费的一台自助移动服务终端机日前因内部蓄电池故障引发火灾。为消除安全隐患，市卫健委决定停用同类自助机设备，并在全系统开展蓄电池安全专项清查。　　据市防火安全委员会办公室通报，近期一所市属医院内存放的一台自助移动服务终端机内部蓄电池发生故障起火，引燃周边可燃物并引发火灾，影响和干扰了正常工作秩序。据了解，该终端机由华北触控电子技术有限公司组装生产,包括终端主机、显示器、打印机、锂离子蓄电池供电系统等部件，用于病患自助挂号、缴费、结算等移动式服务。市卫健委已要求各医院停止使用华北触控电子技术有限公司组装生产的自助移动服务终端机或是同类相似产品，待设备维护单位进行全面检测合格后方可再投入使用。　　为消除安全隐患，市卫健委向市中医局、市医管局、各区卫生健康委、各三级医院和各直属单位下发通知，要求在市属三级医院开展蓄电池安全专项清查。各单位对于目前正在使用、库存保管的各类蓄电池(含待修、待报废仪器设备上的蓄电池)进行全面盘点，特别是华北触控电子技术有限公司组装生产的自助移动服务终端机或是同类相似产品，要查清数量、具体分布和产品来源。要逐一核查在用蓄电池的质量情况，查看其是否具备欠压、过流保护及短路保护功能；要对长期停用设施上的电源设备和蓄电池进行断电处理，并定期检查；要对待报废蓄电池进行统一处理，及时淘汰老旧产品。同时，要查清是否逐级明确管理责任，特别是操作使用、保管保养、检查维护等工作，是否明确到具体人员。凡未明确的，一律要明确到位，防止失管漏管现象发生。要对自助机蓄电池的配套设施进行清查，如加强对蓄电池充电、存放的集中管理,全面检查充电线路、插座，加装短路和漏电保护装置，杜绝私拉乱接电源线路。　　市卫健委还要求各医院加强隐患整治。整治电瓶车乱停乱放，禁止停放在楼梯间、疏散通道、安全出口处，禁止占用消防通道；整治蓄电池失管失保，避免风吹雨打，防止因蓄电池进水引起短路导致车辆或设备自燃；整治长时间违规充电，一般情况下平均充电时间不得超过4小时；整治私自拆卸改装行为，各类蓄电池一旦发现故障，要安排专业人员进行维修，不得擅自拆卸电气保护装置、加装防盗器等设备，确保电气线路和保护装置的完整有效。　　市卫健委要求各单位在日常安全生产检查过程中，将各类蓄电池日常使用管理情况作为重点，及时发现并纠正违法违规行为，确保管理责任落地落实。同时，要结合电气火灾警情提示，加强相关人员的安全教育培训，提升员工识别、发现、化解火灾隐患、扑救初起火灾的能力。"
    rcModelPredictor.predict(question, passage)
    show_answer(args)

    question = "意外伤害保险与工伤保险能否同时受偿"
    passage = "朱某是一港口货运公司员工。2009年12月，朱某因业务需要到武汉出差，公司派车接送。在赶往武汉途中，车辆发生交通事故滑出路面侧翻，造成朱某脊椎骨折，肋骨骨折。事后朱某向公司申请工伤鉴定并获得3万元赔偿。又因朱某购买了意外伤害保险，其欲向保险公司申请理赔。 【分歧】 朱某在获得工伤保险赔偿之后能否再向保险公司申请意外伤害赔偿? 第一种意见认为朱某不能同时获得意外伤害保险赔偿与工伤保险赔偿，否则当事人会取得大于损害之赔偿，违背法律精神; 第二种意见认为工伤保险与意外伤害保险存在性质差异，二者并无冲突，朱某可以在获得工伤赔偿之后再向保险公司申请意外伤害保险赔偿。 【管析】 原文笔者同意第二种意见，其理由是：“一、根据《工伤保险条例》第14条，职工有下列情形之一的，应当认定为工伤：(五)因工外出期间，由于工作原因受到伤害或者发生事故下落不明的。很显然，朱某在为公司办理业务外出途中发生交通事故致身体伤害，应认定为工伤无可争议，因此，朱某理应获得工伤保险赔偿。二、《保险法》第2条规定了保险是指投保人根据合同约定，向保险人支付保险费，保险人对于合同约定的可能发生的事故因其发生所造成的财产损失承担赔偿保险金责任，或者当被保险人死亡、伤残、疾病或者达到合同约定的年龄、期限等条件时承担给付保险金责任的商业保险行为。可以看出，朱某向保险公司申请意外伤害保险赔偿的前提是双方存在保险合同，且该合同是双方平等自愿的真实意思表示。此外，该合同属于市场经济主体之间的商业行为，这一点有别于工伤保险的强制性和社会保障性。《工伤保险条例》与《保险法》均未作出工伤保险赔偿与意外伤害保险赔偿只能二者择其一的法律性规定”。 本文笔者赞同原文笔者的观点，但认为其说理还不够充分，还须作如下几点补充： 一、意外伤害保险是以被保险人的身体利益为保险标的，以被保险人遭受意外伤害为保险事故，当被保险事故发生时，由保险人按合同给付保险金的人身保险。根据保险法“人身保险的被保险人因第三者的行为而发生死亡、伤残或者疾病等保险事故的，保险人向被保险人或者受益人给付保险金后，不得享有向第三者追偿的权利。但被保险人或受益人仍有权向第三者请求赔偿”的规定，无论被保险人是否已经获得赔偿，保险人均应按约向被保险人给付合同约定的保险金，故属于人身保险的意外伤害医疗保险依法应适用给付原则。该案中朱某虽已从社保部门领取了工伤医疗费，但保险公司仍应按照保险合同的约定向其支付意外伤害医疗保险金。 二、工伤保险属于社会保险范畴，是强制保险，用人单位必须缴纳。如职工出现工伤的，依法获得社会保险保障。如果没有办理社会保险，用人单位仍需承担赔付责任。而人身意外伤害保险则为商业保险，自愿投保。 可见，社会保险与商业保险，二者之间存在根本区别，分属不同的法律部门，由不同的法律调整，体现不同的目的，二者不存在替代关系和包容关系，国家立法没有明确规定二者重合，不能同时获得赔付。商业保险和社会工伤保险就同一工伤事故进行赔付，不存在“你消我长”的关系，而应该按自己的赔付或给付标准支付赔偿金或补偿金。 至于有人认为二者之间部分重合或者全部重合，说什么“投保人不能通过两种保险获得双份赔偿而赚钱”是对商业保险中的适用范围的扩大化，也没有法律依据支持。 综上，在该案中，朱某可以在获得工伤赔偿之后再向保险公司申请意外伤害保险赔偿。"
    rcModelPredictor.predict(question, passage)
    show_answer(args)

    question = "心肌梗死是什么"
    passage = "记者日前从联合国“国际遗传工程和生物技术中心”（ICGEB）获悉，其研究小组开发了一种新型疗法，通过植入基因药物，可以帮助诱导心脏细胞再生，治疗心脏衰竭。 　　心肌梗死是冠状动脉硬化心脏病的一种，由冠状动脉的突然阻塞导致心肌细胞缺血坏死，患者的心梗部位会出现永久性结构性损伤，增加了心力衰竭的风险，危害非常严重，可直接致人死亡。据世界卫生组织统计，这种疾病已经影响到世界上超过2300万人口。 　　这项已于近日发表在《自然》杂志上的最新研究称，该团队在心梗模型猪的心脏上植入了一小块名为“微RNA-199a”（microRNA-199a）的基因药物，一个月后，模型猪的心脏功能几乎完全恢复。 　　这是第一个证明可以通过植入有效的基因药物来实现心脏再生的实验。这种药物可以刺激大型动物的心脏再生，再生心脏具有与人类一样的心脏结构和生理功能。 　　据文章第一作者、研究团队负责人马罗·加卡教授介绍，该实验用一种病毒将微RNA分子递送到梗塞的心脏细胞中。在大型哺乳动物中，可通过严格控制治疗制剂，刺激内源性心肌细胞增殖，从而实现心脏修复。但目前，还不能有效地控制剂量和给药时间，从长远来看，还会产生不良反应。加卡补充道：“虽然我们已知这在小鼠中效果显著，但对大型哺乳动物而言，临床试验仍需过一段时间才能进行，我们还要学习如何在大型动物和患者中，将RNA作为合成分子进行管理。” 　　加卡称：“这是一个非常激动人心的时刻。在多次尝试用干细胞再生心脏失败后，我们第一次看到了真正的心脏修复。”该研究是ICGEB与意大利圣安娜高等教育学院和蒙纳斯塔里奥基金会医院合作完成。 　　总编辑圈点 　　说到心脏再生，不免让人想到前段时间闹得沸沸扬扬的哈佛医学院教授撤稿事件——被称为心脏干细胞开山鼻祖的研究者，其实伪造和篡改了数据。“鼻祖”造假了，这个领域还好吗？其实，一种干细胞不成功，并不代表一个领域的所有出路都被堵死。为了让心梗后的心脏也能恢复活力，研究人员一直在想各种办法。看，这项研究就在猪身上取得了初步成果。科研，就是在纠错中进步，在实验中调整，一点点推进，直到最终为人类带来福音。 （责任编辑：王蔚）"
    rcModelPredictor.predict(question, passage)
    show_answer(args)
