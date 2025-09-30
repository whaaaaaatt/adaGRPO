from trl.trainer.grpo_trainer import *
from adaGRPO.AdaLogitsProcessor import AdaLogitsProcessorList
#自定义了GRPO方法
#在_generate_and_score_completions函数的reward的normalize之前，根据每次的reward结果，当满足条件时，重新生成当前组，以防只会解决简单的问题。

class AdaSampleTrigger:
    def __init__(self, ada_trigger_after_step=0, ada_retry_times=0, ada_reward_samplepams_tables_by_times:list[dict]=None):
        self.ada_trigger_after_step=ada_trigger_after_step
        self.ada_retry_times=ada_retry_times
        self.ada_reward_samplepams_tables_by_times=ada_reward_samplepams_tables_by_times

class AdaGRPOTrainer(GRPOTrainer):
    def __init__(self,
                ada_adjust_list:list[AdaSampleTrigger]=None,
                ada_reward_judge_func=None,
                custom_logits_processors: Optional[AdaLogitsProcessorList] = None, 
                change_flag_token_ids:list=None,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        ada_adjust_dict={}
        if ada_adjust_list is None:
            print('AdaGRPOTrainer init ada_adjust_list is None')
            exit(1)
        for t in ada_adjust_list:
            ada_adjust_dict[t.ada_trigger_after_step]=t
        self.ada_adjust_dict=ada_adjust_dict

        self.org_retry_times=0
        self.ada_retry_times = 0

        self.ada_reward_judge_func = ada_reward_judge_func

        self.ada_reward_samplepams_tables_by_times = []

        self.custom_logits_processors=custom_logits_processors
        self.org_custom_logits_processors=custom_logits_processors
        self.change_flag_token_ids=change_flag_token_ids
        self.logits_processor_by_pams_list=[]

        self.stat_step_interval=100
        self.stat_retry_times_by_steps=[]

        #累积reward mean和std用
        self.rewards_per_func_cache=None
        self.actual_run_times=0

    def make_logits_processors(self,ada_reward_samplepams_tables_by_times:list[dict]=None):
        if ada_reward_samplepams_tables_by_times is None:
            ada_reward_samplepams_tables_by_times=self.ada_reward_samplepams_tables_by_times
        self.logits_processor_by_pams_list=[]
        for pamdict in self.ada_reward_samplepams_tables_by_times:
            p=AdaLogitsProcessorList()
            p.init_change_samples(change_flag_token_ids=self.change_flag_token_ids,change_pams_dict=pamdict)
            self.logits_processor_by_pams_list.append(p)
    def parse_generation_config_samplepams(self,samplepams_dict:dict):
        self.generation_config.do_sample=samplepams_dict.get("do_sample",self.generation_config.do_sample)
        self.generation_config.temperature=samplepams_dict.get("temperature",self.generation_config.temperature)
        self.generation_config.top_p=samplepams_dict.get("top_p",self.generation_config.top_p)
        self.generation_config.top_k=samplepams_dict.get("top_k",self.generation_config.top_k)
        self.generation_config.min_p=samplepams_dict.get("min_p",self.generation_config.min_p)
        self.generation_config.repetition_penalty=samplepams_dict.get("repetition_penalty",self.generation_config.repetition_penalty)

    def ada_reward_judge(self,inputs: list[dict[str, Union[torch.Tensor, Any]]],**kwargs):
        print('retry step stat:')
        for idx,cnt in enumerate(self.stat_retry_times_by_steps):
            print(f'step {idx*self.stat_step_interval} - {(idx+1)*self.stat_step_interval}: {cnt} times')
        
        if self.ada_retry_times > 0 and self.ada_reward_judge_func is not None:
            _is_retry=self.ada_reward_judge_func(inputs,**kwargs)
            if _is_retry:
                stat_idx=int(self.state.global_step/self.stat_step_interval)
                while stat_idx+1 > len(self.stat_retry_times_by_steps):
                    self.stat_retry_times_by_steps.append(0)
                self.stat_retry_times_by_steps[stat_idx]+=1

                _ada_reward_samplepams_table = self.ada_reward_samplepams_tables_by_times[-self.ada_retry_times]
                print("Retrying with ADA Reward Sampling...:",str(_ada_reward_samplepams_table))
                self.parse_generation_config_samplepams(_ada_reward_samplepams_table)

                if self.custom_logits_processors is not None:
                    self.custom_logits_processors=self.logits_processor_by_pams_list[-self.ada_retry_times]

                self.ada_retry_times -= 1
                return self._generate_and_score_completions(inputs,is_first_gen=False)
        
        self.ada_retry_times=self.org_retry_times
        if self.state.global_step+1 in self.ada_adjust_dict:
            
            trigger:AdaSampleTrigger=self.ada_adjust_dict[self.state.global_step+1]
            print(f'ada pam change:{str(trigger.ada_reward_samplepams_tables_by_times)}')
            self.org_retry_times=trigger.ada_retry_times
            self.ada_reward_samplepams_tables_by_times=trigger.ada_reward_samplepams_tables_by_times
            self.make_logits_processors()
        #由于后续用到了当前采样参数（在get_per_token_logps中），还原参数放在生成开头
        # self.parse_generation_config_samplepams({
        #     "do_sample": True,
        #     "temperature": self.temperature,
        #     "top_p": self.top_p,
        #     "top_k": self.top_k,
        #     "min_p": self.min_p,
        #     "repetition_penalty": self.repetition_penalty,

        # })
        # self.custom_logits_processors=self.org_custom_logits_processors
        return None

                
            
    #trl.GRPOTrainer original code
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]],
        is_first_gen:bool=True,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        print(type(inputs))
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        #修改：以前是super()调用的Trainer，现在需要用super(GRPOTrainer,self)调用Trainer
        prompt_inputs = super(GRPOTrainer,self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        #参数还原放在这，清空reward累计
        if is_first_gen:
            self.parse_generation_config_samplepams({
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,

            })
            self.custom_logits_processors=self.org_custom_logits_processors

            self.actual_run_times=0
            self.rewards_per_func_cache=None

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,  # vLLM on each GPU generates only 1 in colocate mode
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    #修改：加入可选自定义的processorList，没有针对vllm的情况进行修改
                    if self.custom_logits_processors is None:
                        prompt_completion_ids = unwrapped_model.generate(
                            prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                        )
                    else:
                        prompt_completion_ids = unwrapped_model.generate(
                            prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config,
                            logits_processor=self.custom_logits_processors
                        )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super(GRPOTrainer,self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        #缓存reward以计算累计mean和std
        if self.rewards_per_func_cache is None:
            self.rewards_per_func_cache = rewards_per_func
        else:
            self.rewards_per_func_cache=torch.cat((self.rewards_per_func_cache,rewards_per_func),dim=0)
        self.actual_run_times+=1
        #修改：添加 callback before reward nomormalization
        _ada_reward_judge_res=self.ada_reward_judge(inputs,rewards_per_func=rewards_per_func,prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs)
        if _ada_reward_judge_res!=None:
            return _ada_reward_judge_res

        #计算累计reward和std，不知道batchsize>groupsize时有没有问题(应该没问题)，在本函数开头清空
        def compute_accumulated_reward(rewards_per_func,actual_run_times):
            # Apply weights to each reward function's output and sum
            all_rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

            # Compute grouped-wise rewards
            mean_grouped_rewards = all_rewards.view(-1, self.num_generations*actual_run_times).mean(dim=1)
            std_grouped_rewards = all_rewards.view(-1, self.num_generations*actual_run_times).std(dim=1)
            return mean_grouped_rewards, std_grouped_rewards
        mean_grouped_rewards, std_grouped_rewards = compute_accumulated_reward(self.rewards_per_func_cache, self.actual_run_times)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }
