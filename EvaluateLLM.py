import os
import json
import time
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from EasyChatLLM import ChatPipeline
import lzma
import multiprocessing as mp
import time
import re


def compression_worker(q):
    """后台压缩进程"""
    while True:
        data = q.get()
        if data is None:  # 终止信号
            break
        try:
            json_str = json.dumps(data, ensure_ascii=False)
            # 修改写入逻辑：先写入临时文件再重命名
            temp_file = './cache/EvaluateLLM.lastrun.json.lzma.tmp'
            with lzma.open(temp_file, 'wb') as f:
                f.write(json_str.encode('utf-8'))
            # 原子化替换文件
            if os.path.exists('./cache/EvaluateLLM.lastrun.json.lzma'):
                os.remove('./cache/EvaluateLLM.lastrun.json.lzma')
            os.rename(temp_file, './cache/EvaluateLLM.lastrun.json.lzma')
        except Exception as e:
            print(f"压缩保存失败: {str(e)}")
            # 清理临时文件（如果存在）
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass



class TimeEstimator:
    """新增时间预估类"""
    def __init__(self):
        self.start_time = time.time()
        self.processed_count = 0
        self.total_count = 0
        
    def update(self, processed, total,skip_count):
        self.skip_count = skip_count
        self.processed_count = processed
        self.total_count = total
        
    def estimate(self):
        if self.processed_count == 0:
            return "预计时间: 计算中..."
        elapsed = time.time() - self.start_time
        speed = elapsed / (self.processed_count-self.skip_count)
        remaining = (self.total_count - self.processed_count) * speed
        return f"进度: {self.processed_count}/{self.total_count} ({self.processed_count/self.total_count:.1%}) 剩余: {remaining/60:.1f}分钟 已跳过{self.skip_count}条处理过的数据"

# === 修改配置文件结构 ===
CONFIG_FILE = './configs/EvaluateLLM.config.json'

# === 配置文件检查 ===
if not os.path.exists(CONFIG_FILE):
    default_config = {
        "local_model_config": {
            "model_path": "",
            "adapter_path":"",
            "max_new_tokens": 32768,
            "temperature": 0.7,
            "top_p": 0.9,
            "enable_thinking": True,
            "system_prompt": ""  # 新增system_prompt配置
        },
        "task": "detect",
        "mode": "batch",
        "input_path": "",
        "output_dataset_dir": "./output_datasets",  # 修改输出目录配置
        # 新增列名配置
        "column_names": {
            "instruction": "instruction",
            "output": "output",
            "predict_reason": "predict_reason",
            "predict_content": "predict_content"
        }
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    print(f"Config file '{CONFIG_FILE}' created, please fill in parameters.")
    exit(1)

with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = json.load(f)

model_conf = config['local_model_config']
task = config.get('task', 'rewrite')
mode = config.get('mode', 'batch')
input_path = config['input_path']

# 新增列名配置
column_config = config.get('column_names', {
    "instruction": "instruction",
    "output": "output",
    "predict_reason": "predict_reason",
    "predict_content": "predict_content"
})
# 获取配置的列名
instruction_col = column_config["instruction"]
output_col = column_config["output"]
reason_col = column_config["predict_reason"]
content_col = column_config["predict_content"]

MODEL_WORKERS=2
JSON_INPUT=True
DATA_JSON_FILE="./data/DAPO-Math-17k/DAPO-Math-17k-Processed-en-test-300.json"
MISSION_TYPE="MATH"#MATH or DL

math_solution_start="SOLUTION_START_FLAG"
match_numbers = re.compile(
    math_solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
def extract_numbers(text):
    guess=match_numbers.search(text)
    return guess.group(1) if guess else ""

def rename_columns_DL(example):
    return {
        column_config["instruction"]: example['text'],
        column_config["output"]: example['label'],
        'data_type': example['data_type'],
        'llm_type': example['llm_type']
    }
def rename_columns_MATH(example):
    return {
        column_config["instruction"]: example['prompt'],
        column_config["output"]: example['solution'],
    }

rename_columns=rename_columns_MATH if MISSION_TYPE=="MATH" else rename_columns_DL


class LocalModelHandler:
    def __init__(self):
        self.pipeline = None

    def initialize_pipeline(self):
        try:
            print('model config')
            print(str(model_conf))
            tokenizer = AutoTokenizer.from_pretrained(model_conf["model_path"], trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_conf["model_path"],
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            )

            if model_conf['adapter_path']:
                import peft
                model = peft.PeftModel.from_pretrained(model, model_conf['adapter_path'])
                model = model.merge_and_unload()
                print("Loaded model from adapter path")

            self.pipeline = ChatPipeline(
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=model_conf["max_new_tokens"],
                temperature=model_conf["temperature"],
                top_p=model_conf["top_p"],
                enable_thinking=model_conf["enable_thinking"]
            )
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False

def model_worker(queue_in,queue_out):
    handler = LocalModelHandler()
    if not handler.initialize_pipeline():
        print("模型加载失败")
        exit(1)
    while True:
        input_data = queue_in.get()
        # 清空历史并添加system prompt
        handler.pipeline.clear_history()
        if model_conf["system_prompt"]:
            handler.pipeline.add_to_history("system", model_conf["system_prompt"])
            
        handler.pipeline.add_to_history("user", input_data[instruction_col])
        
        # 生成响应
        response, _, _ = handler.pipeline.generate_response()
        input_data['response']=response
        queue_out.put(input_data)

if __name__ == '__main__':

    # 新增压缩进程全局变量
    compress_queue = mp.Queue()
    compress_process = None

    # 初始化压缩进程
    if compress_process is None:
        compress_process = mp.Process(target=compression_worker, args=(compress_queue,))
        compress_process.start()

    to_model_queue = mp.Queue(maxsize=2*MODEL_WORKERS)
    from_model_queue = mp.Queue()
    for i in range(MODEL_WORKERS):
        p = mp.Process(target=model_worker, args=(to_model_queue, from_model_queue))
        p.start()

    if True:
        # === 修改数据处理逻辑 ===
        from datasets import load_from_disk, Dataset,load_dataset
        import pandas as pd

        output_dataset_dir = config['output_dataset_dir']
        os.makedirs(output_dataset_dir, exist_ok=True)

        

        def process_dataset():
            # 加载数据集
            try:
                if JSON_INPUT:
                    # 定义字段重映射函数
                    # 加载 JSON 文件为 Dataset 对象
                    dataset = load_dataset('json', data_files=DATA_JSON_FILE, split='train')

                    # 添加乱序操作（可指定 seed 保证可重复性）
                    dataset = dataset.shuffle()  # 移除 seed 参数可获得完全随机的排列

                    # 应用字段重映射并删除旧字段
                    dataset = dataset.map(rename_columns)
                else:
                    # 加载数据集
                    dataset = load_from_disk(input_path)
            except Exception as e:
                print(f"加载数据集失败: {str(e)}")
                exit(1)

            # 修改缓存加载逻辑
            processed_data = []
            if os.path.exists('./cache/EvaluateLLM.lastrun.json.lzma'):
                try:
                    # 添加编码参数确保解码一致性
                    with lzma.open('./cache/EvaluateLLM.lastrun.json.lzma', 'rt', encoding='utf-8') as f:
                        processed_data = json.load(f)
                    print(f"检测到未完成的处理缓存，已恢复 {len(processed_data)} 条记录")
                except Exception as e:
                    print(f"缓存加载失败: {str(e)}")

            # 新增已处理指令集合
            processed_instructions = {item[column_config["instruction"]] for item in processed_data}
            
            # 初始化时间预估器
            skip_count = 0
            time_estimator = TimeEstimator()
            total_items = len(dataset)
            time_estimator.update(len(processed_data), total_items,skip_count)

            correct_cnt=0
            all_cnt=0
            idx=0
            while all_cnt<total_items-skip_count:
                if idx<total_items:
                    example = dataset[idx]
                    idx+=1
            #for idx, example in enumerate(dataset):
                # 跳过已处理条目
                current_instruction = example[column_config["instruction"]]
                if current_instruction in processed_instructions:
                    skip_count += 1
                    print(f"[{idx+1}] 跳过已处理条目: {current_instruction[:50]}...")
                    continue
                
                
                
                try:
                    if not to_model_queue.full():
                        to_model_queue.put(example)
                        continue
                    example=from_model_queue.get()
                    response=example['response']
                    
                    # 修改解析逻辑：通过特殊标签判断
                    predict_reason = ""
                    predict_content = response
                    if MISSION_TYPE=="DL":
                        if '<think>' in response and '</think>' in response:
                            split_parts = response.split('<think>')
                            predict_reason = split_parts[1].split('</think>')[0].strip()
                            predict_content = '</think>'.join(split_parts[1].split('</think>')[1:]).strip()
                    elif MISSION_TYPE=="MATH":
                        predict_reason=response#整体都放入reason
                        predict_content=extract_numbers(response)
                    # 删除原<output>标签处理逻辑
                    # if '<output>' in response:
                    #     predict_content = response.split('<output>')[-1].strip()

                    # 新增实时输出
                    print(f"\n=== 推理内容 ===\n{predict_reason}\n=== 生成结果 ===\n{predict_content}\n=== 实际答案 ===\n{example[output_col]}\n")

                    all_cnt+=1
                    if JSON_INPUT:
                        if MISSION_TYPE=="DL":
                            if predict_content.strip() == example[output_col].strip():
                                correct_cnt+=1
                        elif MISSION_TYPE=="MATH":
                            if predict_content.strip().replace(",", "")==example[output_col].strip().replace(",", ""):
                                correct_cnt+=1
                        if all_cnt>0:
                            print(f"累计正确率：{correct_cnt/all_cnt:.2%}")

                    # 构建结果记录
                    processed = {
                        instruction_col: current_instruction,
                        output_col: example[output_col],
                        reason_col: predict_reason,
                        content_col: predict_content
                    }
                    
                    processed_data.append(processed)
                    
                    # 更新处理计数并输出时间预估
                    time_estimator.update(len(processed_data), total_items,skip_count)
                    print(f"[{all_cnt}] 处理成功 | {time_estimator.estimate()}")

                except Exception as e:
                    time_estimator.update(len(processed_data), total_items,skip_count)
                    print(f"[{all_cnt}] 处理失败: {str(e)} | {time_estimator.estimate()}")

                # 修改缓存保存方式
                compress_queue.put(processed_data)

            # 保存处理结果到datasets格式
            if processed_data:
                result_df = pd.DataFrame(processed_data)
                result_dataset = Dataset.from_pandas(result_df)
                result_dataset.save_to_disk(output_dataset_dir)
                print(f"处理完成，结果已保存到 {output_dataset_dir}")
                # 修改缓存清理逻辑
                if os.path.exists('./cache/EvaluateLLM.lastrun.json.lzma'):
                    os.remove('./cache/EvaluateLLM.lastrun.json.lzma')


        process_dataset()  # 替换原来的批处理逻辑


    # 添加程序退出时的清理逻辑
    if compress_process is not None:
        compress_queue.put(None)
        compress_process.join()
