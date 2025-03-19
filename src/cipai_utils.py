# -*- coding: utf-8 -*-
# @File    :   cipai_utils.py
# @Time    :   2024/07/08 11:36:44
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    计算评测指标
    读取从 cnkgraph 爬取的词牌信息
    构建诗歌格式类 PoemFormat 能够实现格式检查
'''
import os, sys 

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = FILE_DIR[:FILE_DIR.rfind('/src')]
sys.path.append(PROJECT_DIR)

import re 
import json 

def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        obj = json.load(f)
    print(f"{path} loaded with {len(obj)} samples!")
    return obj

from loguru import logger
from dataclasses import dataclass

@dataclass
class PoemFormat:
    """ 
        定义诗歌格式
        使用 check_format 函数检查诗歌是否符合格式
    """
    name: str 
    standard_template: str
    repeat: int = None 
    desc: str = None 
    candidate_templates: list = None  # 增加多种格律模板

    @property
    def standard_format(self, template=None):
        """ repeat 参数的处理 """
        if template is None:
            template = self.standard_template

        if not self.repeat:
            return re.split(r'[ \t\n]', template.strip())
        else:
            return re.split(r'[ \t\n]', template.strip()) * self.repeat
        

    def check_format(self, poem, verbose=False):
        """ only consider order and length """
        gold = [ x.strip() for x in self.standard_format ]
        pred = [ x.strip() for x in re.split(r'[ \t\n，。？！、,.?!]', poem.strip()) if x.strip()] # space or tab to split 

        flag = True 

        num_parts_gold, num_parts_pred = len(gold), len(pred)
        if num_parts_gold != num_parts_pred:
            if num_parts_pred in [ num_parts_gold * i for i in (2, 4, 8) ]:  # 实现 repeat 缺失的自动适应
                gold = gold * (num_parts_pred // num_parts_gold)    
            else:
                flag = False                # 不是整数倍 直接判为错误
        
        for p, g in zip(pred, gold):
            if len(p) != len(g):
                flag = False
                break 

        if not flag and verbose:
            logger.warning(f"{self}")
            logger.warning(f"gold: {gold}")
            logger.warning(f"pred: {pred}")
            input()
        return flag

    # 实现一个函数 宽泛的匹配所有 template 的 format
    def check_format_multiple(self, poem):
        for t in self.candidate_templates:
            if self.check_format(poem, t):
                return True
        return False 


cipai = ['浣溪沙', '鹧鸪天', '菩萨蛮', '蝶恋花', '临江仙', 
         '满江红', '清平乐', '水调歌头', '虞美人', '沁园春', 
         '念奴娇', '满庭芳', '西江月', '金缕曲', '点绛唇', 
         '减字木兰花', '踏莎行', '浪淘沙', '水龙吟', '望江南',
         '如梦令', '南乡子', '贺新郎', '卜算子', '采桑子',
         '摸鱼儿', '忆江南',  '渔家傲', '江城子', '鹊桥仙', 
         '忆秦娥', '青玉案', '苏幕遮', '一剪梅', '声声慢', '醉花阴']

formats = [
    PoemFormat("五言绝句", "AAAAA", repeat=4),
    PoemFormat("七言绝句", "AAAAAAA", repeat=4),
    PoemFormat("五言律诗", "AAAAA", repeat=8),
    PoemFormat("七言律诗", "AAAAAAA", repeat=8),    
]

cipai2info = load_json(os.path.join(FILE_DIR, 'cipai2info.json'))
for cipai, info in cipai2info.items():
    template = info['longyusheng_template'][0] # 先用这个，后续可以考虑支持多个
    desc = template['gelv']
    example = template['example']
    # t = template['template'].replace('句', "\t").replace('韵', '\t').replace('读', '\t')

    candidates = info['longyusheng_template'] + info['qinding_templates']  # multiple

    t = [ sent for sent in re.split(r'[\t\n]', template['template']) if sent.strip()]
    t = "\t".join(t)
    f = PoemFormat(cipai, t, desc=desc, candidate_templates=candidates)
    formats.append(f)
    # assert f.check_format(example), f"check failed: {cipai}"

logger.info(f"num formats: {len(formats)}")

class FormatEvaluator:
    def __init__(self) -> None:
        self.formats = formats
        self.cipai2format = { f.name: f for f in self.formats }

    def eval_single(self, cipai, poem_text):
        """判断一首诗是否符合格式
        Args:
            cipai: 诗的类型，如五言绝句、词牌名等
            poem_text: llm 生成的 string
        """
        try:
            format_template = self.cipai2format[cipai]  
        except:
            logger.error(f"unknown cipai: {cipai}")
            return False
        
        # ignore title 
        no_title = poem_text[poem_text.find('\n')+1:].strip()
        no_title_author = no_title[no_title.find('\n')+1:].strip()

        # 三对其一
        return format_template.check_format(poem_text) or format_template.check_format(no_title) or format_template.check_format(no_title_author)
    
    def eval_batch(self, poems):
        """ 
        
        """
        from collections import defaultdict 
        cipai_freq = defaultdict(int)
        failed_cipai = defaultdict(int)
        cnt = 0 
        for sample in poems:
            flag = self.eval_single(sample['cipai'], sample['output'])
            # flag = self.eval_single(sample['cipai'], sample['prediction'])
            cipai_freq[sample['cipai']] += 1
            if flag:
                cnt += 1
            else:
                # logger.error(f"format check failed: {sample}")  
                # print(sample['output'])
                # print(self.cipai2format[sample['cipai']] )
                # print("="*100)
                print(sample['cipai'])
                failed_cipai[sample['cipai']] += 1
                pass 
        logger.info(f"num poems: {len(poems)}, num correct: {cnt}")
        for cipai, freq in cipai_freq.items():
            logger.info(f"{cipai}: {freq}, {failed_cipai[cipai]/freq}")
        return cnt / len(poems)




def _eval_demo():
    data = load_json("./eval/xxxxx_ci_gen_demo.json")
    data = [ x for x in data if x['label'] == 'ci_gen' ]
    print(FormatEvaluator().eval_batch(data))


if __name__ == '__main__':
    _eval_demo()
    # print(FormatEvaluator().eval_single('浣溪沙','春雨渡江有忆  \n小桥流水细如愁。梦里江南雨落柔。烟村孤影叹春流。  \n一帘幽影惹寒意，听君低语诉相留。  \n桃花影里问江楼。  '))
    # print(FormatEvaluator().eval_single('鹊桥仙','扬旗击鼓，斩蛟射虎，头颅碎黄麻天使。鱄诸匕首信豪雄，信当日、一人而已。华表崔巍，松杉森肃，壮士千秋不死。从来忠义出屠沽，惭愧杀、干儿义子。'))

