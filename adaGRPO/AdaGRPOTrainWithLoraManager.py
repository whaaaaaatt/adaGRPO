from typing import Optional
import torch
from torch import nn
from transformers import AutoTokenizer,GenerationConfig,AutoModelForCausalLM,TrainerCallback
from datasets import load_from_disk,load_dataset
from peft import get_peft_model, LoraConfig, TaskType,PeftModelForCausalLM,PeftModel
from trl import GRPOConfig,GRPOTrainer
from torch.nn import functional as F
import json
import os
import time

from .AdaGRPOTrainer import AdaGRPOTrainer,AdaSampleTrigger
from .AdaLogitsProcessor import AdaLogitsProcessorList

class AdaGRPOTrainWithLoraManager:
    def __init__(self):
        self.model=None
        self.grpo_config=None
    def init_base(self,
                model_name,
                lora_config:Optional[dict|LoraConfig],
                dataset_path:str,
                dataset_process_func,
                lora_adapter_base_path:str=None,
                lora_adapter_specific_steps:int=None,
                
                ):
        self.model_name=model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )
        self.lora_config:LoraConfig=lora_config if type(lora_config)==LoraConfig else LoraConfig(**lora_config)
        self.model:PeftModel=None
        if lora_adapter_base_path:
            lora_adapter_path=lora_adapter_base_path
            if lora_adapter_specific_steps and ("checkpoint-"+str(lora_adapter_specific_steps)) in os.listdir(lora_adapter_base_path):
                lora_adapter_path=os.path.join(lora_adapter_base_path,"checkpoint-"+str(lora_adapter_specific_steps))
            
            self.model = PeftModelForCausalLM.from_pretrained(
                self.base_model,
                lora_adapter_path,
                #lora_config=lora_config,
                trust_remote_code=True,
                device_map="auto",
            )
            print('use adapter')
        else:
            self.model = get_peft_model(self.base_model, self.lora_config)
        self.dataset=load_from_disk(dataset_path)
        self.dataset=dataset_process_func(self.dataset,self.tokenizer)

        #禁用kvcache，打开梯度检查点减小显存占用
        self.model.use_cache = False
        self.model.gradient_checkpointing_enable()
        #不加这个会报错：RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        self.model.enable_input_require_grads()
        pass
    def init_GRPO(self,
                output_dir:str,
                num_train_epochs:int=3,
                save_steps:int=100,
                temperature:float=1.0,
                top_p:float=1.0,
                repetition_penalty:float=1.2,
                per_device_train_batch_size:int=1,
                gradient_accumulation_steps=None,
                learning_rate = 5e-6,
                weight_decay = 0.01,
                warmup_ratio = 0.1,
                max_completion_length=8192,
                num_generations=4,
                epsilon=0.2,
                beta=0.04,
                reward_funcs=None,
                use_ada=False,
                ada_sample_triggers:list[AdaSampleTrigger]=None,
                ada_reward_judge_func=None,
                ada_logits_processor_change_token="<think>",
                ada_logits_processor_change_pamdict=None,#{"temperature":0.7,"top_p":0.8},
                ):
        if self.model is None:
            print("you should init base model first")
            return
        self.grpo_config=GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
            temperature=temperature,
            top_p=top_p,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps if gradient_accumulation_steps is not None else int(num_generations/per_device_train_batch_size),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio = warmup_ratio,
            repetition_penalty=repetition_penalty,
            max_completion_length=max_completion_length,
            num_generations=num_generations,
            epsilon=epsilon,
            beta=beta,
        )
        if not reward_funcs:
            print('you should give reward_funcs')
            exit(1)
        self.trainer:Optional[GRPOTrainer|AdaGRPOTrainer]=None
        if use_ada:
            stop_think_token_ids=self.tokenizer.encode(ada_logits_processor_change_token)
            adaLogitsProcessorList=None
            if ada_logits_processor_change_pamdict is not None:
                adaLogitsProcessorList=AdaLogitsProcessorList()
                adaLogitsProcessorList.init_change_samples(stop_think_token_ids,ada_logits_processor_change_pamdict)
            adaSampleTriggers=ada_sample_triggers
            self.trainer=AdaGRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=self.grpo_config,
                train_dataset=self.dataset,
                reward_funcs = reward_funcs,

                ada_adjust_list=adaSampleTriggers,
                ada_reward_judge_func=ada_reward_judge_func,
                custom_logits_processors=adaLogitsProcessorList,
                change_flag_token_ids=stop_think_token_ids,

            )
        else:
            self.trainer=GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=self.grpo_config,
                train_dataset=self.dataset,
                reward_funcs = reward_funcs,
            )
        loggingDir='./log'
        def LogTo(relPath,s):
            with open(os.path.join(loggingDir,relPath),"a",encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S")+" "+s+"\n")

        class RewardLoggingCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                # logs 是一个 dict，里面包含了当前 step 打印出来的所有 metrics
                if logs is not None and "reward" in logs:
                    avg_reward = logs["reward"]
                    logMsg=f"[Step {state.global_step:6d}] Avg Reward = {avg_reward:.4f}"
                    print(logMsg)
                    LogTo(f'{os.path.split(self.model_name)[-1]}.step.log',logMsg)
            def on_evaluate(self, args, state, control, **kwargs):
                logMsg=f"[Step {state.global_step:6d}] eval result = {str(kwargs)}"
                print(logMsg)
                LogTo(f'{self.model_name}.eval.log',logMsg)

        self.trainer.add_callback(RewardLoggingCallback)
    def train(self):
        if self.model is None or self.trainer is None:
            print("you should init base model and GRPO first")
            return
        self.trainer.train()