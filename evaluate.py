# -*- coding: utf-8 -*-
# @File    :   main.py
# @Time    :   2024/10/10 10:26:30
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    重写评估脚本，必须使用 vllm 后端
'''

import os,sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
from collections import defaultdict
from vllm import LLM, SamplingParams
from loguru import logger

from tqdm import tqdm 
import pandas as pd
from langchain_core.prompt_values import StringPromptValue

from src.utils import calculate_rouge_l, calculate_sacrebleu
from src.utils import multiple_choice_acc, ci_format_acc, ci_format_multiple_acc, couplet_format_acc
from src.utils import ci_format_tonal_score, ci_format_tonal_multiple_score
from src.utils import load_json, save_json 

from src.eval_prompts import CriteriaResultOutputParserZH, sft_prompt, sft_criteria, general_criteria,  general_lacc_prompt


def prepare_prompt(tokenizer, prompt):
    """ 
        对于 baichuan 等没有 apply_chat_template 的模型，直接返回 prompt
    """
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt
    return text 


def prepare_lacc_prompt(sample):
    prompt = general_lacc_prompt.format_prompt(
        criteria=general_criteria, input=sample['input'], output=sample['prediction'], reference=sample['output']
    ).text 
    return prompt

def prepare_lacc_prompt_sft(sample):
    """ sft 的 judge 模型的 prompt """
    prompt = sft_prompt.format_prompt(
        criteria=sft_criteria, input=sample['input'], prediction=sample['prediction'], output=sample['output']).text
    return prompt

# 
label2name = {
    'ac_rc': 'ACRC',
    'idiom_rc': 'IRC',
    'tcm_sd': 'TCMSRC',
    'lc_qa': 'LCQA',
    'as_qa': 'ASQA',
    'tcm_qa': 'TCMQA',
    'ask_author': 'BA',
    'ask_dynasty': 'BD',
    'ask_collection': 'BCC',
    'poem_creation':'PG',
    'poem_source': 'PLST',
    'quote_source': 'FQST',
    'idiom_source': 'IST',
    'poem_nmt_inv' :'IPT',
    'poem_nmt': 'PT',
    'ac_nmt': 'ACT',
    'poem_appre': 'PA',
    'tcm_ner': 'TCMNER',
    'idiom_exp': 'IE',
    'prescription_detail': 'PE',
    'couplet_gen': 'CG',
    'ci_gen': 'CiG'
}

# four categories: RC, QA, TG, PG
name2category = {
    "ACRC": "RC",
    "IRC": "RC",
    "TCMSRC": "RC",
    "ASQA": "QA",
    "LCQA": "QA",
    "BA": "QA",
    "BD": "QA",
    "BCC": "QA",
    "PG": "QA",
    "PLST": "QA",
    "FQST": "QA",
    "IST": "QA",
    "IPT": "QA",
    "PT": "TG",
    "ACT": "TG",
    "PA": "TG",
    "IE": "TG",
    "TCMQA": "TG",
    "PE": "TG",
    "CG": "PG",
    "CiG": "PG",
}

class BenchmarkEvaluator:
    """ 
    acc 多项选择
    lacc 用 llm 去判断
    pacc llm 判断，但 prompt 不一样: 诗句、词
    cacc llm 判断，但 prompt 不一样: 对联
    """
    meta_table = [
        ['ac_rc',              'ACRC',    'wyw_sft',              'ac_rc.json',                'acc',   '文言文阅读理解单项选择题'],
        ['idiom_rc',           'IRC',     'irc_sft',              'idiom_rc.json',             'acc',   '成语单项选择题'],
        ['tcm_sd',             'TCMSRC',  'sd_test',              'tcmsd_rc.json',             'acc',   '中医辨证选择题'],
        ['lc_qa',              'LCQA',    'tongjiazi_sft',        'lc_qa.json',                'lacc',  '通假字识别问答 loan character'],
        ['as_qa',              'ASQA',    'xiehouyu',             'as_qa.json',                'lacc',  '歇后语给下句 allegorical saying'],
        ['ask_author',         'BA',      'ask_author',           'book_author.json',          'lacc',  '给书名,回答作者'],
        ['ask_dynasty',        'BD',      'ask_dynasty',          'book_dynasty.json',         'lacc',  '给书名,回答朝代'],
        ['ask_collection',     'BCC',     'ask_collection',       'book_collection.json',      'lacc',  '给书名分类,10个藏集'],
        ['poem_creation',      'PG',      'poem_creation',        'poem_gen.json',             'lacc',  '诗词的创作'],  # 用llm判断因为 task 任务定义太宽泛
        ['poem_source',        'PLST',    'poem_sentence2source', 'poem_line_st.json',         'lacc',  '诗句出处'],
        ['quote_source',       'FQST',    'guji_sentence2source', 'famous_quote_st.json',      'lacc',  '文言文名句出处'],
        ['idiom_source',       'IST',     'idiom_source',         'idiom_st.json',             'lacc',  '成语出处'],
        ['poem_nmt_inv',       'IPT',     'poem_nmt_inv_sft',     'poem_nmt_inv.json',         'lacc',  '根据译文给出原文'],
        ['poem_nmt',           'PT',      'poem_nmt_sft',         'poem_nmt.json',             'bleu',  '诗词的翻译'],
        ['ac_nmt',             'ACT',     'erya_1k_sft',          'ac_nmt.json',               'bleu',  'erya 数据集筛选的数据'],
        ['idiom_exp',          'IE',      'idiom_explanation',    'idiom_exp.json',            'bleu',  '成语释义'],
        ['poem_appre',         'PA',      'poem_appreciation',    'poem_appre.json',           'bleu',  '诗词赏析'],
        ['tcm_qa',             'TCMQA',   'tianchi_qa',           'tcm_qa.json',               'bleu',  '中药问答'],
        ['prescription_detail', 'PE',     'fangji_name2detail',   'prescription_exp.json',     'bleu',  '中药方剂名称到详情'],
        ['couplet_gen',        'CG',      'couplet_sft',          'couplet_gen.json',          'cacc',  '对下联'],         # 对联评估
        ['ci_gen',             'CiG',     'cipai_sft',            'ci_gen.json',               'pacc',  '根据词牌名写一首词要求格律符合'], # 词牌评估
    ]
    file2num = {}
    def __init__(self, config, gold_key="output", pred_key="prediction"):
        self.gold_key = gold_key
        self.pred_key = pred_key

        self.config = config 
        print(self.config)
        self.fewshot_n = config.fewshot
        
        # 将 meta_table 转换为 DataFrame 以便更轻松地访问
        self.meta_df = pd.DataFrame(self.meta_table, 
                                  columns=['label', 'name', 'keyword', 'filename', 'metric', 'description'])
        
        # cache prediction results
        self.save_path = os.path.join(self.config.save_dir, f"{self.config.version}.json")
        if not os.path.exists(self.save_path):

            self.backend = None 
            if not config.mode == 'api':
                self._load_model()  # self.backend set to 'vllm' or 'torch'
            else: 
                logger.info("API mode, no model loaded")
            self.all_data = self._prepare_all_prompts(raw=True if self.config.mode == 'api' else False) 
            # 这里没问题，api 模式下，raw 为 True 会在服务端把 prompt 加上chat template， 否则需要自己加上
        else:
            logger.info("Loading cached predictions from {}".format(self.save_path))
            self.all_data = load_json(self.save_path)


    def _load_model(self, model=None, ngpus=None, max_length=None, greedy=False, prefix_cache=False):
        """ 
        默认使用 config 的配置
        greedy 开启后是 greedy decoding
        """
        self.llm = LLM(
            model=self.config.model_name_or_path if model is None else model,
            trust_remote_code=True,
            tensor_parallel_size=self.config.gpu_num if ngpus is None else ngpus,
            enable_prefix_caching=prefix_cache,
            enforce_eager=True,
            quantization=self.config.quantization,
        )
        self.tokenizer = self.llm.get_tokenizer()
        if greedy:
            self.sp = SamplingParams(max_tokens=self.config.max_length if max_length is None else max_length, top_k=1, temperature=0.0, n=1)  # greedy decoding
        else:
            self.sp = SamplingParams(max_tokens=self.config.max_length if max_length is None else max_length, top_p=0.95, temperature=1.0, n=1)

        logger.info("Model loaded using vllm")
        self.backend = 'vllm'

    def _prepare_all_prompts(self, raw=False):
        """ 
        先构造好所有的prompt 然后进行推理
        raw: 是否使用原始的prompt=instruction\ninput，否则使用模板
        """

        all_data = [] 
        logger.info(f"raw is {raw}, not using template")
        
        for _, row in self.meta_df.iterrows():
            label, filename = row['label'], row['filename']
            
            data = load_json(os.path.join(self.config.data_dir, filename))
            
            for s in data:
                if raw:
                    p = f"{s['instruction']}\n{s['input']}"
                else:
                    p = prepare_prompt(self.tokenizer, f"{s['instruction']}\n{s['input']}") 
                s['prompt'] = p 
                s['label'] = label 
                s['metric'] = row['metric']
                s['sample_id'] = s['sample_id']
            all_data.extend(data)
            self.file2num[filename] = len(data)

        logger.info(f"{len(all_data)} prompts prepared for {len(self.meta_df)} tasks")
        if not os.path.exists(os.path.join(".", "cache")):
            os.makedirs(os.path.join(".", "cache"))
        save_json(all_data, os.path.join(".", "cache", f"{self.config.version}_prompts.json"))
        return all_data
        

    def get_prediction(self, prompts):

        def _api_evaluation(prompts):
            try:
                import chattool
            except:
                raise ImportError("chattool not installed, `pip install -U chattool`")
            if self.config.api_base is None and self.config.chat_url is None:                  # 本地 ip 和 port
                chattool.api_base = f'http://{self.config.host}:{self.config.port}/v1'
            elif self.config.chat_url is None and self.config.api_base is not None:         # 远程 api_base    
                chattool.api_base = self.config.api_base
            elif self.config.chat_url is not None:
                logger.info(f"Using chat_url,  set to {self.config.chat_url}")
            else:
                raise ValueError("chat_url should be set alone. ")
            
            chattool.api_key = self.config.api_key
            ckpt_file = f"./cache/api_{self.config.version}_result_shot{self.fewshot_n}.jsonl"
            logger.info(f"开始调用API `{chattool.api_base}` 进行评估, {len(prompts)} prompts to process, cache in `{ckpt_file}`")
            chattool.async_chat_completion(
                msgs=prompts,
                nproc=4,
                chat_url=self.config.chat_url,       # 远程 api_url， glm-4 
                chkpoint=ckpt_file,
                max_tries=5,
                model=self.config.model_name_or_path,
            )

            chats = chattool.load_chats(f"./cache/api_{self.config.version}_result_shot{self.fewshot_n}.jsonl")
            R = [] 
            for c in chats:
                try:
                    pred = c.chat_log[-1]['content']
                    R.append(pred)
                except Exception as e:
                    # c is None, because the api does not return any response
                    logger.info(f"Error in api_evaluation: {e} || Perhaps you need to rerun this script!")
                    R.append("")
            return R


        def _vllm_evaluation(prompts):

            outputs = self.llm.generate(prompts=prompts, sampling_params=self.sp, use_tqdm=True)
            R = [] 
            for o in outputs:
                output_text = o.outputs[0].text
                R.append(output_text)


            logger.info("评估模型调用结束，即将释放资源")
            del self.llm 
            self.llm = None
            return R 
            

        if self.config.mode == 'api':
            return _api_evaluation(prompts)
        
        else:
            if self.backend == 'vllm':
                return _vllm_evaluation(prompts)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            
    def lacc_evaluation(self, V):
        """ 
        使用 llm 进行推理， 评估答案的对错
        """
        
        prompts = []
        for sample in V:
            # 将这些task 转成 prompt 用 vllm 推理
            
            prompt = prepare_lacc_prompt(sample)
            prompts.append(prompt)
        logger.info(f"llm_eval(lacc) eval prompt prepared! {len(prompts)}")
        print(prompts[0])

        self._load_model(
            model="/sshfs/pretrains/Qwen/Qwen2-7B-Instruct",
            ngpus=1, max_length=256,
            greedy=True,
            prefix_cache=True
        )
        prompts = [ prepare_prompt(self.tokenizer, p) for p in prompts ]  # apply chat template 
        outputs = self.llm.generate(prompts=prompts, sampling_params=self.sp, use_tqdm=True)
        
        for s, o in zip(V, outputs):
            output_text = o.outputs[0].text
            s['llm_eval'] = output_text
            s['score'] = 1 if output_text.strip().split("\n")[-1].lower() == 'y' else 0

        logger.info(f"llm_eval(lacc) eval finished!")
        return V 
            

    def lacc_evaluation_api(self, V):
        """ 
        使用 langchain 进行 evaluation
        """
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="sft",
            temperature=0,
            openai_api_key='EMPTY',
            openai_api_base=self.config.llm_evaluator_url, 
            max_tokens=1024,
            verbose=True
        )
        # check if llm is available
        try:
            llm.generate_prompt([StringPromptValue(text="hello")])
        except Exception as e:
            logger.error(f"Using LLM evaulator on {self.config.llm_evaluator_url} failed!")
            raise ValueError(f"LLM is not available: {e}")

        def run_evaluator():
            if not hasattr(self, 'langchain_evaluator'):
                from langchain.evaluation import load_evaluator
                self.langchain_evaluator = load_evaluator('labeled_criteria', llm=llm, criteria=general_criteria)  # 问题在于prompt中英掺杂
                self.langchain_evaluator.output_parser = CriteriaResultOutputParserZH()

            for sample in tqdm(V, desc="langchain_eval"):
                output = self.langchain_evaluator.evaluate_strings(
                    prediction=sample['prediction'],
                    input=sample['instruction'] +"\n"+  sample['input'],
                    reference=sample['output'],
                )
                try:
                    sample['score'] = int(output['score'])
                    sample['llm_eval'] = output['reasoning']
                except:
                    sample['score'] = 0
                    sample['llm_eval'] = "Error\t" + str(output)

        def run_prompt_manual():
            """  langchain evaluator 用改动的全中文的 prompt """
            
            output_parser = CriteriaResultOutputParserZH()
            for sample in tqdm(V, desc="langchain_eval"):

                response = llm.generate_prompt([StringPromptValue(text=prepare_lacc_prompt_sft(sample))])  # sft
                result = [ output_parser.parse_result(generation)  for generation in response.generations][0]
                resp_text = str(response.generations[0])
                
                try:
                    sample['score'] = int(result['score'])
                    sample['llm_eval'] = resp_text
                except:
                    sample['score'] = 0
                    sample['llm_eval'] = "Error\t" + resp_text

                # print(prompt)
                # print(response)
                # print(result)
        
        run_prompt_manual()
        logger.info(f"langchain_eval eval finished!")
        return V


    def __call__(self, data_to_eval=None, postfix="_evaluated.json") :
        """
            1. 先试用 prompt 进行 infer 并保存结果
            2. 根据任务名label选择对应的评测指标
            3. 计算指标 并打印出来
        """
        data_to_eval = self.all_data if data_to_eval is None else data_to_eval

    
        if os.path.exists(self.save_path):
            logger.info(f"Prediction results file already loaded as self.all_data")
            R = self.all_data
            if os.path.exists(self.save_path.replace(".json", postfix)):
                logger.info(f"Evaluated results file already exists, loading from {self.save_path.replace('.json', postfix)}")
                R = load_json(self.save_path.replace(".json", postfix))
                logger.warning("Results will be re-evaluated!")
        else:
            prompts = [s['prompt'] for s in data_to_eval]
            logger.info(f"Using {self.config.model_name_or_path} to infer {len(prompts)} prompts")
            logger.info(f"prompts[0]: {prompts[0]}")
            predictions = self.get_prediction(prompts) # LLM infer 返回结果是 List[string]

            R = [] 

            for p, s in zip(predictions, data_to_eval):
                s[self.pred_key] = p
                R.append(s)
            
            save_json(R, self.save_path)
            logger.info(f"Results saved to {self.save_path}")

        V = [] # sample using lacc for vllm infer
        C = [] # metric-calculated samples
        # 根据任务名label选择对应的评测指标
        for sample in tqdm(R, desc="Evaluating"):

            # if not sample['label'] in ['ci_gen', 'couplet_gen']:  # 只评估诗词和对联
            # if not sample['label'] in ['poem_nmt_inv']: 
            if 'label' not in sample:
                sample['label'] = 'ci_gen'
            if not sample['label'] in ['ci_gen']:  
                continue

            # 使用 DataFrame 查找相应的度量标准
            try:
                metric = self.meta_df.loc[self.meta_df['label'] == sample['label'], 'metric'].iloc[0]
            except IndexError:
                logger.error(f"Sample with label '{sample['label']}' not found in meta_df. Available labels: {self.meta_df['label'].unique()}")
                continue
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue
 
            if metric == 'bleu':
                score = calculate_sacrebleu(reference=sample[self.gold_key], generated=sample[self.pred_key])
            elif metric == 'acc':
                score = multiple_choice_acc(pred=sample[self.pred_key], gold=sample[self.gold_key])
            elif metric == 'cacc':
                score = couplet_format_acc(pred=sample[self.pred_key], gold=sample[self.gold_key])
            elif metric == 'pacc':
                # score = ci_format_multiple_acc(pred=sample[self.pred_key], cipai=sample['cipai'])
                score = ci_format_acc(pred=sample[self.pred_key], cipai=sample['cipai'])
                tonal_score = ci_format_tonal_score(pred=sample[self.pred_key], cipai=sample['cipai'])
                tonal_multiple_score = ci_format_tonal_multiple_score(pred=sample[self.pred_key], cipai=sample['cipai'])
                sample['tonal_score'] = tonal_score
                sample['tonal_multiple_score'] = tonal_multiple_score
            elif metric == 'lacc':
                V.append(sample)
                continue
            else:
                raise ValueError(f"Unknown metric {metric}")
            
            sample['score'] = score
            C.append(sample)

        # vllm
        # evaluated_V = self.lacc_evaluation(V)
        # evaluated_V = self.lacc_evaluation_api(V)
        evaluated_V = [] # 暂时关闭 lacc 评估, 如果已有结果, 则直接使用
        C.extend(evaluated_V)


        evaluated_result_path = self.save_path.replace('.json', postfix)
        # if not os.path.exists(evaluated_result_path):
        save_json(C, evaluated_result_path)
        logger.info(f"Evaluated result saved to {evaluated_result_path}")

        scores = defaultdict(list)
        for sample in C:
            scores[sample['label']].append(sample['score'])

        for label, scores_list in scores.items():
            avg_score = sum(scores_list) / len(scores_list)
            metric = self.meta_df.loc[self.meta_df['label'] == label, 'metric'].iloc[0]
            avg_score = avg_score if avg_score > 1 or metric =='bleu' else avg_score * 100  # convert to percentage
            num = len(scores_list)
            
            # 查找相应的名称和描述
            name = self.meta_df.loc[self.meta_df['label'] == label, 'name'].iloc[0]
            desc = self.meta_df.loc[self.meta_df['label'] == label, 'description'].iloc[0]
            
            print(f" {name:<10}: {avg_score:.4f}\t{num:^10}{desc:>30}")

    print("\n","="*80)


class ICL_Evaluator(BenchmarkEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.fewshot_n = config.fewshot

    def prepare_icl_prompt_for_fewshot_setting(self, label, path, num=5):
        """ 
        为每一个任务保留 5 个样本，用于 fewshot setting
        """
        file = "dev_fewshot.json"
        if not hasattr(self, 'fewshot_dev') or label not in self.fewshot_dev:
            if os.path.exists(file):
                task2examplars = load_json(file)
                if label not in task2examplars:
                    data = load_json(path)
                    task2examplars[label] = data[:5]
                    save_json(task2examplars, file)
            else:
                data = load_json(path)
                task2examplars = {label: data[:5]}
                save_json(task2examplars, file)

            self.fewshot_dev = task2examplars
        else:
            task2examplars = self.fewshot_dev

        examplars = task2examplars[label][:num]
        def format_fewshot_prompt(examplars):
            
            return """请参考以下示例，完成题目。""" + "\n\n".join([f"{s['instruction']}\n{s['input']}\n{s['output']}" for s in examplars])
        
        return format_fewshot_prompt(examplars)

    def _prepare_all_prompts(self, raw=False):
        """ 
        先构造好所有的prompt 然后进行推理
        raw: 是否使用原始的prompt=instruction\ninput，否则使用模板
        """

        all_data = [] 
        logger.info(f"raw is {raw}, check if using template")

        for _, row in self.meta_df.iterrows():
            label, filename = row['label'], row['filename']

            if not label in ['ci_gen']:  # 只评估诗词和对联
                continue

            data = load_json(os.path.join(self.config.data_dir, filename))

            if raw and not hasattr(self, 'tokenizer'):      # api mode, no tokenizer avaliable, use qwen tokenizer to estimate length
                from transformers import AutoTokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.len_estimator)
                except:
                    logger.info(f"using `{self.config.len_estimator}` tokenizer to estimate length")
                    input("Press Enter to continue...(will start model downloading from huggingface or ctrl-c to modify here to use local model)")
                    tokenizer = AutoTokenizer.from_pretrained(self.config.len_estimator)
                self.tokenizer = tokenizer

            for s in data:
                fewshot_n = self.fewshot_n
                icl_prompt = self.prepare_icl_prompt_for_fewshot_setting(label, os.path.join(self.config.data_dir, filename).replace('test.json', 'dev.json'), num=fewshot_n)
                p = prepare_prompt(self.tokenizer, icl_prompt + '\n\n' + s['instruction']+"\n"+s['input']) 
                p_len = len(self.tokenizer.tokenize(p))
                while p_len > self.tokenizer.model_max_length - 1:
                    fewshot_n -= 1 
                    icl_prompt = self.prepare_icl_prompt_for_fewshot_setting(label, os.path.join(self.config.data_dir, filename).replace('test.json', 'dev.json'), num=fewshot_n)
                    p = prepare_prompt(self.tokenizer, icl_prompt + '\n\n' + s['instruction']+"\n"+s['input']) 
                    p_len = len(self.tokenizer.tokenize(p))
                    if fewshot_n < 1:
                        break
                
                if raw:
                    p = icl_prompt + '\n\n' + s['instruction'] + "\n" + s['input']

                s['prompt'] = p 
                s['label'] = label 
                s['metric'] = row['metric']
                s['sample_id'] = s['sample_id']
            all_data.extend(data)
            self.file2num[filename] = len(data)

        logger.info(f"{len(all_data)} prompts prepared for {len(self.meta_table)} tasks")
        return all_data


def main():
    """ 
        vllm 0.5B takes less than 4mins on a single 3090 GPU
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", "-m", type=str, default="/sshfs/pretrains/Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--mode", type=str, choices=['api', 'vllm'], default='vllm')
    parser.add_argument("--gpu_num", "-g", type=int, default=1)
    parser.add_argument("--data_dir", "-d", type=str, default="./test_data")
    parser.add_argument("--version", "-v", type=str, default="test_version", help="mark the version of the results")
    parser.add_argument("--quantization", "-q", type=str, default=None, help="quantization type")
    parser.add_argument("--fewshot", "-f", type=int, default=0, help="fewshot number")
    parser.add_argument("--len_estimator", "-le", type=str, default="/sshfs/pretrains/Qwen/Qwen2-0.5B-Instruct", help="length estimation tokenizer")

    # when API mode enabled 
    parser.add_argument("--port", "-p", type=str, default="8000")
    parser.add_argument("--host", "-ip", type=str, default="127.0.0.1")
    parser.add_argument("--api_key", "-k", type=str, default="EMPTY") 
    parser.add_argument("--api_base", "-b", type=str, default=None) 
    parser.add_argument("--chat_url", "-c", type=str, default=None) 

    # langchain eval llm 
    parser.add_argument("--langchain_eval_llm", "-lc", choices=['qwen2_0.5b', 'qwen2_7b', 'gpt4o-mini'], default='qwen2_0.5b')
    parser.add_argument("--llm_evaluator_url", "-e", type=str, default="http://tenqserver:8002/v1")

    # save results
    parser.add_argument("--save_dir", "-s", type=str, default="./results/")
    parser.add_argument("--max_length", "-l", type=int, default=1024)
    args = parser.parse_args()
    
    print("args parsed!")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        logger.info(f"Save results to {args.save_dir}")
    if args.fewshot > 0:
        evaluator = ICL_Evaluator(
            config=args
        )
    else:
        evaluator = BenchmarkEvaluator(
            config=args
        )
    # evaluator(postfix="_evaluated_multiple.json")
    evaluator(postfix="_evaluated.json")

if __name__ == '__main__':
    main()
