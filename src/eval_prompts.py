# -*- coding: utf-8 -*-
# @File    :   llm_eval.py
# @Time    :   2024/09/04 11:19:22
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    用于评估的 prompts 
'''


from langchain_core.prompts import PromptTemplate

custom_criteria = {
    "包含要点": "理解问题和预期回答，之后从预期回答中提取出一个答案的要点，回答必须包含这个要点。",
    "与预期回答一致": "回答必须与预期回答一致。如果预期回答是 ABCD 四个选项中的一个，那么回答必须包含预期回答中的选项。如果预期回答是一段文字，那么回答必须与预期回答传达的信息一致。",
    "回答简洁明了": "回答需要简洁明了，如果出现大量重复内容则直接判断为错误。",
}


fstring = """作为一名阅卷老师，请根据以下评分标准对以下回答进行评价，评价结果的形式是：如果回答正确，输出"Y" ，如果回答错误输出"N"。绝对不要输出额外的内容。

评分标准: {criteria}


数据:
---------
问题: {input}
预期回答: {reference}
回答: {output}
---------
依次提供你对每个标准的解释。最后在新的一行上输出"Y"或"N" (这一行只有 Y 或 N ，不要出现其他内容)。

"""
langchain_eval_prompt = PromptTemplate.from_template(fstring)

poem_creation_criteria = {
    "必须符合格律格式规范": "必须符合格律格式规范，如五言绝句、七言绝句等，且每句字数相同。",
    "必须符合诗歌主题": "作品必须符合要求中的关键词，主题；基于对作品的理解进行评判。",
}

poem_fstring = """作为一名熟悉中国古代文化和文学的老师，请对以下作品进行评价，评价结果的形式是：如果回答正确，输出"Y" ，如果回答错误输出"N"。绝对不要输出额外的内容。

评价准则: {criteria}

要求: {input}
作品: {output}

参考: {reference}

依次提供你对每个标准的解释。最后在新的一行上输出"Y"或"N" (这一行只有 Y 或 N ，不要出现其他内容)。

"""

poem_creation_eval_prompt = PromptTemplate.from_template(poem_fstring)


##################################################
#     general prompt 
##################################################



general_criteria = {
    "包含要点": "理解问题和标准回答，之后从标准答案中提取出一个答案要点，预测答案必须包含这个要点。",
    "与标准答案一致": "如果预测答案是一段文字，那么预测答案必须与预标准答案传达的信息一致。",
    "符合问题和任务要求": "如果问题要求创作诗词, 那么回答中的诗词创作必须符合要求的主题和格式。基于对作品的理解进行评判。",
    "预测答案简洁明了": "预测答案需要简洁明了，如果出现大量重复内容则直接判断为错误。",
}

fstring = """作为一名阅卷老师，请根据以下评分标准对以下回答进行评价，评价结果的形式是：如果回答正确，输出"Y" ，如果回答错误输出"N"。绝对不要输出额外的内容。

评分标准: {criteria}


数据:
---------
问题: {input}
预期回答: {reference}
回答: {output}
---------
依次提供你对每个标准的解释。结合前面的判断，最后在新的一行上输出"Y"或"N"。

"""
general_lacc_prompt = PromptTemplate.from_template(fstring)



##################################################
#     
##################################################
from langchain.evaluation import load_evaluator
from langchain.evaluation.criteria.eval_chain import CriteriaResultOutputParser 
class CriteriaResultOutputParserZH(CriteriaResultOutputParser):
    """ 它原本只匹配头尾或者单独的 Y|N 
    需要除去中文的标点符号。 case `Y。` """
    def parse(self, text: str) -> dict:
        text = text.strip("。")
        return super().parse(text)

##################################################
#     prompts for vllm 
##################################################
tmp = """数据包括“任务输入”，“标准答案”，几条“准则”，需要评估正确与否的“预测答案”。
根据准则比较“预测答案”和“标准答案”，如果“预测答案”不满足任意一条准则，最终的评估结果为错误（结论“N”）。只有“预测答案”满足所有准则，最终的评估结果为正确（结论“Y”）。"""

general_prompt_dep = """请根据标准答案判断预测的回答是否正确。如果回答正确，输出"Y" ，如果回答错误输出"N"。绝对不要输出除了"Y"或"N"之外的任何内容。"""
general_prompt = """
你正在根据一组准则评估给出的数据。
以下是数据：
[数据开始]
***
[任务输入]: {input}
***
[标准答案]: {output}
***
[准则]: {criteria}
***
[预测答案]: {prediction}
***
[数据结束]

预测答案是否符合各项准则？首先，逐步写出你对于每个准则的推理过程，以确保你的结论是正确的。避免一开始就简单地陈述评估结论。然后在下一行仅写出单个字符“Y”或“N”（不带引号或标点符号），如果预测答案不满足任意一条准则，写下“N”，否则写下“Y”。在最后，在新的一行中再单独重复一次该字符，表示评估的最终结论。
"""
# print(general_prompt.format(input="xx", prediction="xx", criteria="xx", output="xx"))
poem_nmt_inv_prompt = """"""


general_criteria = {
    "包含要点": "理解问题和标准回答，之后从标准答案中提取出一个答案要点，预测答案必须包含这个要点。",
    "与标准答案一致": "如果预测答案是一段文字，那么预测答案必须与预标准答案传达的信息一致。",
    "符合问题和任务要求": "如果问题要求创作诗词, 那么回答中的诗词创作必须符合要求的主题和格式。基于对作品的理解进行评判。",
    "预测答案简洁明了": "预测答案需要简洁明了，如果出现大量重复内容则直接判断为错误。",
}

sft_criteria = "\n" + "\n".join([f"{i}: {c}" for i, c in general_criteria.items()])

prompt = """作为作为一名阅卷老师，请根据以下评分标准对以下回答进行评价，评价结果的形式是：如果回答正确，输出"Y" ，如果回答错误输出"N"。绝对不要输出额外的内容。
评分标准: {criteria}
<start>
***
[任务输入]: {input}
***
[标准答案]: {output}
***
[预测答案]: {prediction}
***
<end>

[评价结果]:\n"""

sft_prompt = PromptTemplate.from_template(prompt)


if __name__ == '__main__':
    example = {
        "input": "请创作一首五言绝句，主题为春天，要求符合格律格式规范。",
        "output": "春风送暖入屠苏，桃李花开满园春。燕舞莺歌春意浓，人间仙境似仙境。",
        "reference": "春风送暖入屠苏，桃李花开满园春。燕舞莺歌春意浓，人间仙境似仙境。",
        "criteria": custom_criteria,

    }

    example = {
        "instruction": "根据以下翻译给出对应的古诗词原文。",
        "input": "我饮酒不需要劝杯，反而担心酒杯空了。分别相离也是可恨的事情，这次的分别是那么的匆忙。酒席上美女贵宾云集，花园外豪富高门坟冢，人世间谁能算是英雄？一笑出门而去，千里外的风吹得花落。\n孙权刘备这样的人物，才能指使我做事，而不是阁下。我发出种种的感慨，这些交心于你知道。只是感觉自己一生游遍湖海，除了喝醉吟些风花雪月，便是一事无成。身上的所有东西都是陛下赐予，希望我在湖北的作为能使君王明鉴。",
        "output": "题目: 水调歌头·我饮不须劝\n作者:辛弃疾\n淳熙丁酉，自江陵移帅隆兴，到官之三月被召，司马监、赵卿、王漕饯别。司马赋《水调歌头》，席间次韵。时王公明枢密薨，坐客终夕为兴门户之叹，故前章及之。\n我饮不须劝，正怕酒樽空。别离亦复何恨？此别恨匆匆。头上貂蝉贵客，苑外麒麟高冢，人世竟谁雄？一笑出门去，千里落花风。\n孙刘辈，能使我，不为公。余发种种如是，此事付渠侬。但觉平生湖海，除了醉吟风月，此外百无功。毫发皆帝力，更乞鉴湖东。",
        "prompt": "<s><|im_start|>user\n根据以下翻译给出对应的古诗词原文。\n我饮酒不需要劝杯，反而担心酒杯空了。分别相离也是可恨的事情，这次的分别是那么的匆忙。酒席上美女贵宾云集，花园外豪富高门坟冢，人世间谁能算是英雄？一笑出门而去，千里外的风吹得花落。\n孙权刘备这样的人物，才能指使我做事，而不是阁下。我发出种种的感慨，这些交心于你知道。只是感觉自己一生游遍湖海，除了喝醉吟些风花雪月，便是一事无成。身上的所有东西都是陛下赐予，希望我在湖北的作为能使君王明鉴。<|im_end|>\n<|im_start|>assistant\n",
        "label": "poem_nmt_inv",
        "metric": "lacc",
        "prediction": "吾不能饮，恐空怀越乡之忧；相别何恨，偏承促柱之调。宾主此宵乐可知，明朝看花去，为寿大江流。\n寄深且问。一舸任廉公，父兄遗恨。忍待斜光岁岁辉南极，恐归来，吹我洞庭秋。望眼欲穿江树晚，乱帆如画故园愁。朱栏花槛待诗留。\n\n这首词是清代康熙年间知名诗人袁枚所写的一首词，主要表达了词人对于别离的悲伤和对于未来的期许。全词运用了丰富的意象，透过饮酒、分别、花香、风力、悲欢等，展现出了词人复杂的情感和对人生不同阶段的反思。在翻译时，我们尽量捕捉原词中的意境与深意，同时在现代汉语中寻找最为接近的表述来传达这种情感和哲思。此外，考虑到此词的历史背景，译者还需要确保使用合适的词汇来反映原文所蕴含的文学性和文化深度。因此，翻译时不仅要传递字面含义，还须体现原诗的音韵和节奏美感，以及对人生世事洞悉后的那份坦然与超然。",
        "score": 1,
        "llm_eval": "N"
    }

    # print(langchain_eval_prompt.format_prompt(**example).text)
    # print(poem_creation_eval_prompt.format_prompt(**example).text)

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="/sshfs/pretrains/Qwen/Qwen2-7B-Instruct",
        temperature=0,
        openai_api_key='EMPTY',
        openai_api_base=f'http://map:8001/v1', # sparql only proto
        max_tokens=1024,
        verbose=True
    )

    from langchain_core.prompts import StringPromptTemplate
    from langchain_core.prompt_values import StringPromptValue
    x = StringPromptValue(text='hello')
    print(llm.generate_prompt([x]))
    input()


    evaluator = load_evaluator('labeled_criteria', llm=llm, criteria=general_criteria)
    # evaluator = load_evaluator('labeled_criteria', llm=llm, criteria=poem_creation_criteria)
    evaluator.output_parser = CriteriaResultOutputParserZH()
    print(evaluator)
    print()

    o = evaluator.evaluate_strings(
        prediction=example['prediction'],
        input=example['input'],
        reference=example['output'],
    )
    print(o)

    print(o['score'])

