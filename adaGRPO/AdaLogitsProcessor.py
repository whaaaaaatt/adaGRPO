from transformers.generation.logits_process import *
#generate在加载logitsprocessor时，会构造一堆默认的logitsprocessor，按序对logits处理，
# 如ForcedBOSTokenLogitsProcessor、EncoderRepetitionPenaltyLogitsProcessor等，然后再添加自定义的logitsprocessor，
# 然后再根据sample添加temperatureWrapper、toppWrapper等，
# 生成时会在生成每个token后，调用logitsprocessorList的__call__方法
# 所以将原generation_config的temperature、top_p等置为默认值，然后添加自定义的Wrapper，或者在调用的processorList的时候更换wrapper
# 调用时在generate里将这个list传入

class AdaLogitsProcessorList(LogitsProcessorList):
    no_min_tokens_to_keep_pam_list=["temperature"]
    sample_refer_dict={
        "temperature":TemperatureLogitsWarper,
        "top_p":TopPLogitsWarper,
        "top_k":TopKLogitsWarper,
        "min_p":MinPLogitsWarper,
    }
    extra_refer_dict={
        "repetition_penalty":RepetitionPenaltyLogitsProcessor,
    }
    def init_change_samples(self,change_flag_token_ids:list[int],change_pams_dict:dict,min_tokens_to_keep=1):
        self.change_flag_token_ids=change_flag_token_ids
        if type(self.change_flag_token_ids) is not torch.Tensor:
            self.change_flag_token_ids=torch.tensor(self.change_flag_token_ids)
        self.change_pams_dict=change_pams_dict
        self.min_tokens_to_keep=min_tokens_to_keep
        self.org_processors=None
        self.changed_processors=None
    #具体使用时才加载，因为当generate时才会添加默认的processor
    def make_changed_processors(self):
        if self.changed_processors is None:
            self.org_processors=LogitsProcessorList()
            for processor in self:
                self.org_processors.append(processor)
            
            self.changed_processors=LogitsProcessorList()
            #！里边直接将参数传给对应processor而不用**kwargs的形式，目前看各个processor是没问题的,同时没有用到使用device参数的processor也不用给
            for processor in self:
                changed_processor=processor
                for k in self.extra_refer_dict:
                    if k in self.change_pams_dict:
                        changed_processor=self.extra_refer_dict[k](self.change_pams_dict[k])
                for k in self.sample_refer_dict:
                    if self.change_pams_dict.get('do_sample',True) is False:
                        changed_processor=None
                        break
                    if k in self.change_pams_dict:
                        if k in self.no_min_tokens_to_keep_pam_list:
                            changed_processor=self.sample_refer_dict[k](self.change_pams_dict[k])
                        else:
                            changed_processor=self.sample_refer_dict[k](self.change_pams_dict[k],self.min_tokens_to_keep)
                if changed_processor is not None:
                    self.changed_processors.append(changed_processor)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        if self.changed_processors is None:
            self.make_changed_processors()

        self.change_flag_token_ids=self.change_flag_token_ids.to(input_ids.device)
        B, L = input_ids.shape
        k = self.change_flag_token_ids.numel()
        if k<L:
            #原方法
            scores=self.org_processors(input_ids, scores,**kwargs)
        else:
             # 1) 对比最后 k 个 token
            #    input_ids[:, -k:] 形状 (B, k)，和 flag_token_ids 广播后做逐元素比较
            #    eq_mask: (B, k)，all(dim=1) 后得到 (B,) 的布尔掩码
            eq_mask = (input_ids[:, -k:] == self.change_flag_token_ids.unsqueeze(0)).all(dim=1)  # (B,)
            
            # 2) 用掩码分成两组
            match_input_ids = input_ids[eq_mask]
            not_match_input_ids = input_ids[~eq_mask]
            match_scores     = scores[eq_mask]      # (N_match, V)
            not_match_scores = scores[~eq_mask]     # (N_not,   V)

            # 3) 分别调用用户提供的处理函数
            #    这些函数内部也应尽量使用 torch ops，以保持 GPU 加速
            #！这里放**kwargs给里面的processor不知道会不会出问题因为input_ids和scores都拆分了,目前看都没有用额外的参数
            processed_match_scores     = self.changed_processors(match_input_ids,match_scores,**kwargs)     # (N_match, V)
            processed_not_match_scores = self.org_processors(not_match_input_ids,not_match_scores,**kwargs)  # (N_not, V)

            # 4) 将两部分结果放回原位置
            scores = torch.empty_like(scores)      # (B, V)
            scores[eq_mask]  = processed_match_scores
            scores[~eq_mask] = processed_not_match_scores

        return scores