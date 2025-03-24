# Fùxì Benchmark

Fùxì  (负屃), a comprehensive benchmark that evaluates both understanding and generation capabilities across 21 diverse tasks.


| Task | Description | Type | Eval Metric | # |
|------|-------------|------|-------------|---|
| Ancient Chinese RC | Multiple choice question on ancient Chinese text reading comprehension | NLU | Acc | 300 |
| Idiom RC | Multiple choice question on idiom reading comprehension | NLU | Acc | 1000 |
| TCM Syndrome RC | Multiple choice question on TCM syndrome differentiation | NLU | Acc | 1098 |
| Loan Character QA | Loan character identification question answering | NLG | Acc | 500 |
| Allegorical Saying QA | Allegorical saying question answering, to complete the second half | NLG | Acc | 553 |
| Book Author | Given a book title, answer the author | NLG | Acc | 384 |
| Book Dynasty | Given a book title, answer the dynasty | NLG | Acc | 372 |
| Book Collection Classification | Classify book titles into 10 collections | NLG | Acc | 532 |
| Poetry Generation | Poetry creation according to topic words | NLG | Acc | 300 |
| Poetry Line Source Tracing | Identify the source of a poem line | NLG | Acc | 400 |
| Famous Quote Source Tracing | Identify the source of a classical Chinese quote | NLG | Acc | 200 |
| Idiom Source Tracing | Name the source book or essay of an idiom | NLG | Acc | 962 |
| Inverse Poetry Translation | Decipher the original poem from translated modern Chinese version | NLG | Acc | 300 |
| Poetry Translation | Translate a poem into modern Chinese | NLG | BLEU | 300 |
| Ancient Chinese Translation | Translate ancient Chinese text into modern Chinese (extracted from the erya dataset) | NLG | BLEU | 1000 |
| Poetry Appreciation | Analysis of imagery, style, sentiment in classical Chinese poetry | NLG | BLEU | 109 |
| Idiom Explanation | Idiom explanation | NLG | BLEU | 1236 |
| TCM QA | Answer questions given related TCM text | NLG | BLEU | 740 |
| Prescription Explanation | Prescription details for TCM | NLG | BLEU | 404 |
| Couplet Generation | Generate the second line based on the first line | NLG | Acc | 500 |
| Ci Generation | Write a poem following a specific ci pattern with tonal rules | NLG | Acc | 300 |

# Usage

For zero-shot evaluation:
```shell
CUDA_VISIBLE_DEVICES=0 sh ./eval_zeroshot.sh
```

For few-shot in-context evaluation:
```shell
CUDA_VISIBLE_DEVICES=0 sh ./eval_icl.sh
```

The LLM evaluator configuration is in function `lacc_evaluation_api`

# Citation
<!-- arxiv coming soon -->

```bibtex
@misc{zhao2025fuxibenchmarkevaluatinglanguage,
      title={F\`ux\`i: A Benchmark for Evaluating Language Models on Ancient Chinese Text Understanding and Generation}, 
      author={Shangqing Zhao and Yuhao Zhou and Yupei Ren and Zhe Chen and Chenghao Jia and Fang Zhe and Zhaogaung Long and Shu Liu and Man Lan},
      year={2025},
      eprint={2503.15837},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.15837}, 
}
```