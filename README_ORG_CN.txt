adaGRPO：更高效、稳定、显存占用更小的自适应GRPO强化训练方法。
！！emoji：星星！！如果觉得很赞，请给本仓库点一个star，谢谢！
！！简体中文与英文版本切换！！
I. 算法介绍
GRPO（！！英文全称！！）算法是DeepSeek研究团队提出的一种PPO（！！英文全称！！）强化学习算法的改进。PPO算法使用一个与策略模型（待优化模型）大小相同的价值模型来表示基准及计算采样奖励R的相对优势A，而GRPO算法去掉了这个价值模型，通过采样一组输出O、计算组内输出的平均奖励R的均值及方差来计算相对优势A，较好的减少了资源的占用。其流程图如下图所示：
！！图片文件：ppo_grpo_architecture.jpg!!
本改进算法adaGRPO（adaptive GRPO）旨在解决以下两个问题：
问题1. 在GRPO中，参数num_generations为对于单个prompt单组内采样生成输出的个数，其较大时训练效果较好单计算量与显存占用也大幅增加。num_generations在训练过程中是静态的。其理想情况是：
对于拟合较好的数据子集（如数学问题中的简单问题），num_generations应较小以加快模型迭代减少计算资源与显存的消耗。
对于拟合较差的数据子集（如数学问题中的困难问题），num_generations应较大以使模型多次尝试增加本组内出现较好问题解的概率。
问题2. 在GRPO中，模型生成时的采样参数也是静态的。与问题1相似，模型在面对简单问题时应使用较保守的采样参数防止模型不稳定；在面对困难问题时应使用较激进的采样参数来寻找不同采样路径解决问题。
adaGRPO提出了自适应的采样个数及采样参数，以实现在更少的计算量及显存占用的情况下达到相比原GRPO算法在较高num_generations参数下训练时更好的训练效果。具体实现方法为：
当本组采样输出不满足某一条件（后续给出的实验中为没有一个输出包含正确答案）时，缓存本组采样输出的奖励值的平均值及方差，重新使用更激进的采样参数进行采样，直至满足前述条件后结合之前缓存的奖励值的平均值及方差计算相对优势A并优化策略模型，并重置采样参数进行下一次迭代。
以下为adaGRPO的流程图、GRPO与adaGRPO使用模型！！链接文字：Qwen3-1.7B；链接地址：https://huggingface.co/Qwen/Qwen3-1.7B。！！在数据集！！链接文字：DAPO-MATH-17k；链接地址：https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k。！！训练条件及效果对比。
！！图片文件：adagrpo_architecture.jpg!!
！！图片文字描述：adaGRPO算法流程图，Dynamic Sample部分参考注2！！
！！图片文件：grpo_train.jpg!!
！！图片文字描述：使用GRPO进行训练时的平均奖励值变化曲线！！
！！图片文件：adagrpo_train.jpg!!
！！图片文字描述：使用adaGRPO进行训练时的平均奖励值变化曲线！！
！！图片文件：train_result_table.jpg!!
！！图片文字描述：训练后其在测试数据集上的准确率对比表！！
注1：训练模型为Qwen3-1.7B，数据集为DAPO-MATH-17k，使用r=16的LoRA适配器在一张5060ti（16g）上进行训练。训练步数为400步，测试数据集数量为300个。两实验设置num_generations为8，最大生成token数量为8192。adaGRPO算法在200以后启用自适应机制，平均奖励较低的第200-300step内重采样了64次，平均奖励较高的300step-400step时重采样了37次，训练占用显存与num_generations相同的GRPO并无明显差异。GRPO算法在近200步开始模型无法正常收敛，由于训练集都采用了相同顺序的相同数据，推测原因是在200步左右模型处理大量困难任务时，原GRPO算法在采样数量固定的情况下无法推理正确的回答，因为优势函数正则化的存在，格式得分等微小差异被放大，模型错方向进行拟合。
注2：流程图中Dynamic Sample部分仅需要关注Adjust Sample Pam，剩余部分为作者在模型需要格式化的/较稳定的输出时做的优化。后续adaLogitsProcessor会进行简短介绍。
II. 文件结构
1. adaGRPO/AdaGRPOTrainer.py：实现adaGRPO算法的训练代码。基于hugging face官方的trl库中GRPOTrainer。
2. adaGRPO/AdaGRPOTrainWithLoraManager.py：简化使用adaGRPO进行训练的代码，整合AdaGRPOTrainer与模型、LoRA的初始化，同时兼容进行原GRPO训练。主要函数为：init_base函数初始化模型、LoRA参数、训练数据集；init_GRPO函数初始化adaGRPO/GRPO的trainer；train函数与原trainer类似进行训练。
3. adaGRPO/AdaLogitsProcessor.py：强化学习在进行采样时，较为激进的采样参数（如Temperature较高）可以使模型探索更多样化的决策路径解决问题，但多样化的输出在一些如分类问题需要约束输出内容、需要结构化输出等方面有较大挑战，制约了强化学习在这类任务中的表现。因此提出adaLogitsProcessor，在模型进行thinking时使用较激进的采样参数，同时在最终输出时使用较保守的采样参数，以加快模型收敛及在分类问题更稳定的模型效果。本模块可以离开AdaGRPOTrainer单独使用。
4. LLMRLV6Math.py： 前述实验训练代码，代码、奖励函数等参考！！链接文字：unsloth的GRPO训练教程；链接地址：https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb。！！，实验条件见注1。模型路径、输出路径、训练数据集路径可在文件开头的几个变量中修改。使用2.中AdaGRPOTrainWithLoraManager进行训练，其他如模型初始化参数、训练参数见2.中描述。
5. EasyChatLLM.py：一个简单的大模型聊天GUI程序，可以加载LoRA checkpoint、保存对话记录、记忆上次参数并生成config.json文件。用于直观的测试模型效果。
6. EvaluateLLM.py，CheckEvalResMath.py：用于评估训练效果的代码，用于参考。
III. 运行
1. 首先安装依赖库：
！！命令行代码：pip install -r requirements.txt！！
也可以只关注本代码修改较多的torch、transformers、trl版本，但不知道有没有什么潜在问题。
2. 准备
准备本实验使用的模型！！链接文字：Qwen3-1.7B；链接地址：https://huggingface.co/Qwen/Qwen3-1.7B。！！在数据集！！链接文字：DAPO-MATH-17k；链接地址：https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k。！！（也可以更换想测试的模型与数据集），在LLMRLV6Math.py文件开头修改模型、输出文件夹、训练数据集位置。参照并修改代码中dataset_process_func函数对数据集进行裁剪或修改。
3. 运行
运行LLMRLV6Math.py即可进行训练。
4. 测试评估
运行EasyChatLLM.py即使用GUI程序简单测试模型效果。Evaluate文件可用于参考评估训练效果。
IV. 参考工作
1. GRPO算法：！！链接文字：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models；链接地址：https://arxiv.org/abs/2402.03300。！！
2. GRPO训练教程：！！链接文字：unsloth的GRPO训练教程；链接地址：https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb。！！
3. 模型：！！链接文字：Qwen3-1.7B；链接地址：https://huggingface.co/Qwen/Qwen3-1.7B。！！
4. 数据集：！！链接文字：DAPO-MATH-17k；链接地址：https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k。！！