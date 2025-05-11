# -*- coding: utf-8 -*-

import os, sys 

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = FILE_DIR[:FILE_DIR.rfind('/src')]
sys.path.append(PROJECT_DIR)

import re 
import random 
random.seed(42)
from rouge_chinese import Rouge
import jieba
from loguru import logger

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
import json 

from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from langchain.evaluation.criteria.eval_chain import CriteriaResultOutputParser 


def save_json(R, path, **kwargs):
    """ Obj, path """
    with open(path, 'w', encoding='utf8') as f:
        json.dump(R, f, indent=2, ensure_ascii=False, **kwargs)
    print(f"{path} saved with {len(R)} samples!")

def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        obj = json.load(f)
    print(f"{path} loaded with {len(obj)} samples!")
    return obj

class CriteriaResultOutputParserZH(CriteriaResultOutputParser):
    """ 它原本只匹配头尾或者单独的 Y|N 
    需要除去中文的标点符号。 case `Y。` """
    def parse(self, text: str) -> dict:
        text = text.strip('。”"')
        return super().parse(text)
    
def get_prompt(s):
    """ 从 prediction or instruction 中抽取出 预测时使用的 prompt（no llm chat template） """
    if s.strip() == "":
        raise Exception(f"sample `prompt` is empty string. {s} ")
    pattern = re.compile(r'(\|>user\n)(.*?)(<\|im_end\|>\n<\|im_start\|>assistant)', re.DOTALL)
    try:
        return re.search(pattern, s).group(2)
    except Exception as e:
        # print(s)
        # raise e 
        return s # api mode 对应的 prompt 不包含 chat template 所以直接返回

def get_input(s):
    """ 除去 instruction """
    return get_prompt(s).split("\n")[-1].strip()

def containment_acc(pred, gold):
    """如果gold string 包含在 pred 中就算对
    """
    return gold.strip() in pred 



class BenchmarkLangchainEvaluator:
    """ 使用 langchain 的 evaluator 来评估预测的结果 
        需要手动配置
        - llm
        - criteria
        - prompt 

        输出是一个 dict
        # 使用 langchain 速度比较慢
    """
    def __init__(self, llm=None, criteria=None, prompt=None):
        
        assert llm is not None
        assert criteria is not None and prompt is not None
        
        self.llm = llm 
        self.evaluator = load_evaluator("labeled_criteria", llm=self.llm, criteria=criteria, prompt=prompt)
        self.evaluator.output_parser = CriteriaResultOutputParserZH()    

    def evaluate(self, sample):
        sample_prompt = get_prompt(sample['prompt'])
        eval_output = self.evaluator.evaluate_strings(
            prediction=sample['prediction'],   
            reference=sample['output'],
            input=sample_prompt
        )
        return eval_output


from src.cipai_utils import FormatEvaluator
CiFormat = FormatEvaluator()

def ci_format_acc(pred, cipai):
    """ 词牌格式检查 """
    return CiFormat.eval_single(cipai=cipai, poem_text=pred)

def couplet_extraction(text):
    """ 从 text 中提取对联 """
    pattern = r"下联：(.*?)\n"
    m = re.search(pattern, text)
    if m:
        return m.group(1).strip()
    else:
        return text.strip()

def couplet_format_acc(pred, gold):
    """ 对仗检查 
        only word counts
        TODO: pos_tag matching
    """
    pred = couplet_extraction(pred)

    pred_parts = [p.strip() for p in re.split(r'[ \t\n，。？！、,.?!]', pred.strip()) if p.strip() ]
    gold_parts = [p.strip() for p in re.split(r'[ \t\n，。？！、,.?!]', gold.strip()) if p.strip() ]

    if len(pred_parts) != len(gold_parts):
        return False
    
    for p, g in zip(pred_parts, gold_parts):
        if len(p) != len(g):
            return False
    
    return True 



choices = ["A", "B", "C", "D"]
def extract_choice(response):
    '''
        Always return a choice, even cannot match by regex,
        to ensure fair comparison to other models.
    '''
    response = str(response)
    if response[0] in choices:
        return response[0]
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        (r'答案(是|为)选项 ?([ABCD])', 2),
        (r'故?选择?：? ?([ABCD])',1),
        (r'正确的?选项(是|为) ?([ABCD])',2),
        (r'答案(应该)?(是|为)([ABCD])',3),
        (r'选择答案 ?([ABCD])',1),
        (r'答案?：?([ABCD])',1),
        (r'([ABCD])(选?项)?是?符合题意',1),
        (r'答案选项：? ?([ABCD])', 1), # chatglm
        (r'答案(选项)?为(.*?)([ABCD])', 3), # chatgpt
        (r'content=(.*?)([ABCD])(.*?)', 2), # finetuned qwen
        (r'选项是：?\s*?\*\*([ABCD])\*\*', 1),  # markdown bold 
        (r'选项是：?\s*?\*\*([ABCD])\.\s?', 1),  # markdown bold dot 
        (r'选项[是为]?：?\s*?([ABCD])\.?\s?', 1),  # 
        (r'^\s*?([ABCD])\.?\s*?[\n\S]', 1),  # single line answer at beginning
        (r'^\s+?([ABCD])\.?\s*?[\n\S]', 1),  # answer at start of line
        (r'([ABCD]) ?选?项(是|为)?正确',1),
        (r'选项 ?([ABCD]) ?(是|为)?正确',1),
        (r'故?选择?：?\s*?(\*\*)?([ABCD])\.?\s?', 2),   # 故选**A. 正虚瘀结证**。
        (r'\*\*答案是?\*\*：\s*?([ABCD])\.?\s?', 1),   # **答案**：D. 痰湿蕴肺证

    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            logger.debug(pattern)
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 2. Recursive match
    patterns = [
        (r'([ABCD])(.*?)当选', 1),
        # (r'([ABCD])(.*?)正确', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            # logger.debug(pattern)
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer

    # 3. Weak single match
    patterns = [
        (r'[^不]是：\s*?([ABCD])', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            # logger.debug(pattern)
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 4. Check the only mentioend choices
    pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    # return choices[random.randint(0,3)]
    return "Z"  # direct wrong


def multiple_choice_acc(pred, gold):
    """ 从 llm 输出的pred 字符串中提取 选项，与 gold 对比 """
    try:
        pred = extract_choice(pred)
        logger.info(f"pred={pred}, gold={gold}")
        return pred.strip() == gold.strip()
    except Exception as e:
        logger.info(f"extract_choice failed:{e} \n pred={pred}, gold={gold}")
        return False


def calculate_rouge_l(reference, generated):
    # 使用 jieba 进行中文分词
    reference_tokens = ' '.join(jieba.cut(reference))
    generated_tokens = ' '.join(jieba.cut(generated))
    
    scorer = Rouge(metrics=["rouge-l"])
    score = scorer.get_scores(generated_tokens, reference_tokens)[0]['rouge-l']['f']
    return score * 100

def calculate_bleu(reference, generated):
    # 使用 jieba 进行中文分词
    reference_tokens = jieba.lcut(reference)
    generated_tokens = jieba.lcut(generated)
    return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=SmoothingFunction().method1) * 100
    # return sentence_bleu(references=[reference], hypothesis=generated, smoothing_function=SmoothingFunction().method1)


def calculate_sacrebleu(reference, generated):
    """  """
    score =  sacrebleu.sentence_bleu(generated, [reference], tokenize='zh')
    # print(score)
    return score.score

    

if __name__ == '__main__':

    def test_multiple_choice():
        generated = "正确答案为：\n\n**B. 饮食积滞证**"    
        reference = "A"

        print(multiple_choice_acc(generated, reference))

    test_multiple_choice()
    exit()

    # generated = "【药品商品名称】 金刚藤糖浆\n【药品名称】 金刚藤糖浆\n【批准文号】 国药准字Z43020300\n【成分】 本品主要成分为：金刚藤\n【剂型】 本品为棕褐色的液体；气微，味甜、微涩\n【规格】 150ml\n【功效】 用于附件炎、盆腔炎\n【用法用量】 口服，一次20ml，一日3次\n【不良反应】 偶见恶心、呕吐，停药后可自行消失\n【注意事项】 孕妇禁用\n【相互作用】 如与其他药物同时使用可能会发生药物相互作用，详情请咨询医师或药师\n【疗效】 清热解毒，消肿散结\n【药品包装】 药用聚乙烯烃塑料瓶，每瓶150ml\n【制药公司】 怀化正好制药有限公司\n【40】金刚藤糖浆是以百合科植物菝葜的根茎为原料提炼制成的纯天然制剂，具有活血化瘀、解毒祛湿、散结的作用，药物实验对金黄色葡萄球菌、大肠杆菌等有抑菌作用\n【43】 国家基本药物目录（2012）,中药保护品种二级\n【注意事项】 孕妇忌服\n【功能主治】清热解毒，消肿散结\n【40】金刚藤糖浆是以百合科植物菝葜的根茎为原料提炼制成的纯天然制剂，具有活血化瘀、解毒祛湿、散结的作用，药物实验对金黄色葡萄球菌、大肠杆菌等有抑菌作用\n【43】 国家基本药物目录（2012）,中药保护品种二级"
    
    # reference = "{\"药品\": [\"金刚藤糖浆\"], \"药物成分\": [\"金刚藤\"], \"药物剂型\": [\"液体\"], \"药物性味\": [\"气微\"], \"疾病\": [\"附件炎\"], \"症状\": [\"恶心\"], \"人群\": [\"孕妇\"], \"中药功效\": [\"清热解毒\"]}"

    generated = "江彬想要各个城门的钥匙，都督府过来问乔宇，乔宇说：守备官是用来防范非常事变的。"
    reference = "江彬向都督府索要城门的钥匙，都督府问宇。宇说：守备者，是为了防止不测。"

    print(calculate_rouge_l(reference, generated))
    print(calculate_bleu(reference, generated))
    print(calculate_sacrebleu(reference, generated))