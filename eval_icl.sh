# python evaluate.py -s ./results_icl -m /sshfs/pretrains/Qwen/Qwen2.5-0.5B-Instruct -v qwen2.5_0.5b -f 5
 

python evaluate.py -s ./results_icl -f 5 --mode api -m deepseek-r1-250120 -v deepseek_r1_volcano --api_key $ARK_API_KEY --api_base $ARK_API_BASE   

# API models 
python evaluate.py -s ./results_icl -f 5 --mode api -m gpt-4o -v gpt-4o --api_key $CHATANY_API_KEY --api_base $CHATANY_API_URL  
python evaluate.py -s ./results_icl -f 5 --mode api -m gpt-4o-mini -v gpt-4o-mini --api_key $CHATANY_API_KEY --api_base $CHATANY_API_URL  
python evaluate.py -s ./results_icl -f 5 --mode api -m glm-4-plus -v glm-4-plus --api_key $CHATANY_API_KEY --api_base $CHATANY_API_URL  
python evaluate.py -s ./results_icl -f 5 --mode api -m qwen-max-format -v qwen-max --api_key $CHATANY_API_KEY --api_base $CHATANY_API_URL  


# local models
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Qwen/Qwen2-0.5B-Instruct -v qwen2_0.5b  
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Qwen/Qwen2-7B-Instruct -v qwen2_7b  
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/ancient/Xunzi-Qwen2-7B -v Xunzi_qwen2_7b
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/THUDM/glm-4-9b-chat -v glm4_9b_chat



# llama31 
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Meta-Llama/Meta-Llama-3.1-8B-Instruct -v llama31_8b_chat 
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/shenzhi-wang/Llama3.1-8B-Chinese-Chat -v llama31_8b_chinese_chat

# internlm 7b 20b 
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/InternLM/internlm-7b-instruct -v internlm25_7b_chat
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/InternLM/internlm-20b-instruct -v internlm25_20b_chat


# qwen2.5 1.5b 3b 7b 14b 72b
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Qwen/Qwen2.5-1.5B-Instruct -v qwen2.5_1.5b_inst  
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Qwen/Qwen2.5-3B-Instruct -v qwen2.5_3b_inst  
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Qwen/Qwen2.5-7B-Instruct -v qwen2.5_7b_inst  
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Qwen/Qwen2.5-14B-Instruct -v qwen2.5_14b_inst  
python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Qwen/Qwen2.5-72B-Instruct -v qwen2.5_72b_inst  


python evaluate.py -s ./results_icl -f 5 -m /sshfs/pretrains/Qwen/QwQ-32B-AWQ -v qwq_32b_icl_cig
python evaluate.py -s ./results_icl -f 5 --mode api -m deepseek-r1-distill-qwen-32b-250120 -v deepseek_r1_qwen32b_icl --api_key $ARK_API_KEY --api_base $ARK_API_BASE  
