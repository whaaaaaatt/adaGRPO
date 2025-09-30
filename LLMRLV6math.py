from adaGRPO.AdaGRPOTrainer import *
from adaGRPO.AdaGRPOTrainWithLoraManager import AdaGRPOTrainWithLoraManager

model_path_or_name='../../AIRelated/model/Qwen3-1.7B'
check_point_output_dir='../../AIRelated/mymodel/Qwen3-1.7B'
train_dataset_path='./data/DAPO-Math-17k/DAPO-Math-17k-Processed-en-train-1000'

def extract_hash_answer(text):
    # if "####" not in text: return None
    # return text.split("####")[1].strip()
    return text
reasoning_end="</think>"
reasoning_start="<think>"
solution_start="SOLUTION_START_FLAG"
solution_end="SOLUTION_END_FLAG"
def dataset_process_func(dataset,tokenizer):

    #Place it between {reasoning_start} and {reasoning_end}.
    system_prompt = \
    f"""You are given a math problem.
    Think about the problem and solve it.
    Then, the output must include your final number answer between string "{solution_start}" and "{solution_end}". 
    for examle: if the final answer is number x, then the output should include "{solution_start} x {solution_end}".
    """

    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    })
    tokenized = dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
)
    print(tokenizer.decode(tokenized[0]["tokens"]))
    tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

    import numpy as np
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print("Max Length = ", maximum_length)

    # Filter only samples smaller than 90% max length
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    del tokenized
    return dataset

import re


manager=AdaGRPOTrainWithLoraManager()
#init model. 初始化模型。
manager.init_base(
    model_name=model_path_or_name,
    lora_config={
        "r": 16,
        "lora_alpha": 32,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    dataset_path=train_dataset_path,
    dataset_process_func=dataset_process_func,
)


# Add optional EOS token matching
solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
    "(?:" + re.escape(manager.tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end_regex}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)
match_format

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!

        # No need to reward <start_working_out> since we always prepend it!
        # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        # Correct answer gets 5 points!
        if guess == true_answer:
            score += 5.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 1.5
                else: score -= 2.5 # Penalize wrong answers
            except:
                score -= 4.5 # Penalize
        scores.append(score)
    return scores

match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
# print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
# print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
# print(match_numbers.findall("<SOLUTION>  -0.234  </SOLUTION>"))
# print(match_numbers.findall("<SOLUTION>17</SOLUTION>"))

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]
    extracted_responses_after_think=[]
    for r in responses:
        r_elms=r.split(reasoning_end)
        guess=None
        if len(r_elms) > 1:
            guess=match_numbers.search(r_elms[-1])
            guess=guess.group(1) if guess is not None else None
        extracted_responses_after_think.append(guess)

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            '*'*20 + f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}"
        )
    PRINTED_TIMES += 1

    for guess,guess_after_think, true_answer in zip(extracted_responses,extracted_responses_after_think, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            extra_reward=0
            if guess_after_think is not None and guess.strip().replace(",", "")==guess_after_think.strip().replace(",", ""):
                extra_reward=0.5
            # Remove commas like in 123,456
            guess       = float(guess.strip().replace(",", ""))
            scores.append(3.5+extra_reward if guess == true_answer else -1.5+extra_reward)
        except:
            scores.append(0)
            continue
    return scores

def ada_reward_judge(inputs,rewards_per_func,prompts, completions, completion_ids, **reward_kwargs):
        need_retry=False
        right_cnt=0
        for rewards_by_func in rewards_per_func:
            answer_reward=rewards_by_func[-1]
            if answer_reward>0:
                right_cnt+=1
        print("right_cnt:",str(right_cnt))
        print("rewards_per_func:",str(rewards_per_func))
        if right_cnt<1:
            need_retry=True
        return need_retry

#init GRPO params, param num_generations is the minimal group size unit. 
# 初始化GPRPO参数，num_generations参数是最小的生成组大小单元
manager.init_GRPO(
    output_dir=check_point_output_dir,
    num_train_epochs=1,
    learning_rate=1e-5,
    max_completion_length=8192,
    num_generations=2,
    per_device_train_batch_size=2,
    reward_funcs=[
        match_format_approximately,
        match_format_exactly,
        check_answer,
        check_numbers,
        
    ],
    use_ada=True,
    #参数解释：ada_sample_triggers按如下参数输入后，训练时效果为：
    # 1. 训练到第1步时，将adaGRPO的最大重试次数改为4，采样参数固定使用温度1.0
    # 2. 训练到第100步时，将adaGRPO的最大重试次数改为8，采样参数前4次重试使用温度1.0，后四次使用1.1且top_p改为0.99
    # 3. 训练到第300步时，将adaGRPO的最大重试次数改为16，采样参数前8次重试使用温度1.0，后8次使用1.1且top_p改为0.99
    # 4. 训练到第700步时，将adaGRPO的最大重试次数改为32，采样参数前10次重试使用温度1.0，后11次使用1.1且top_p改为0.99，最后11次使用1.3且top_p改为0.97
    #Parameter Explanation: After inputting the parameters according to the following ada_sample_triggers, the effect of training is as follows:
    # 1. At step 1, set the maximum retry times of adaGRPO to 4 and use temperature 1.0 for sampling parameters
    # 2. At step 100, set the maximum retry times of adaGRPO to 8 and use temperature 1.0 for the first 4 retries and temperature 1.1 and top_p 0.99 for the remaining 4 retries
    # 3. At step 300, set the maximum retry times of adaGRPO to 16 and use temperature 1.0 for the first 8 retries and temperature 1.1 and top_p 0.99 for the remaining 8 retries
    # 4. At step 700, set the maximum retry times of adaGRPO to 32 and use temperature 1.0 for the first 10 retries, temperature 1.1 and top_p 0.99 for the next 11 retries,
    ada_sample_triggers=[
        AdaSampleTrigger(1,4,[{"temperature":1.0} for i in range(4)]),
        AdaSampleTrigger(100,8,
                         [{"temperature":1.0} for i in range(4)]+[{"temperature":1.1,"top_p":0.99} for i in range(4)]
                         ),
        AdaSampleTrigger(300,16,
                         [{"temperature":1.0} for i in range(8)]+[{"temperature":1.1,"top_p":0.99} for i in range(8)]
                         ),
        AdaSampleTrigger(700,32,
                        [{"temperature":1.0} for i in range(10)]
                        +[{"temperature":1.1,"top_p":0.99} for i in range(11)]
                        +[{"temperature":1.3,"top_p":0.97} for i in range(11)]
                        )
    ],
    ada_reward_judge_func=ada_reward_judge,
)
manager.train()