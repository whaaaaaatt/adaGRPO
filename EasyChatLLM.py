import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog
import threading
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

CONFIG_FILE_PATH="./configs/EasyChatLLM.config.json"

# 3. 创建对话管道
class ChatPipeline:
    def __init__(self, model, tokenizer, max_new_tokens=32768, temperature=0.7, top_p=0.9, enable_thinking=True):
        self.model:AutoModelForCausalLM = model
        self.tokenizer = tokenizer
        self.chat_history = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking

    def add_to_history(self, role, content):
        self.chat_history.append({"role": role, "content": content})

    def clear_history(self):
        self.chat_history = []

    def generate_response(self, system_prompt=None):
        start_time = time.time()
        
        # Create message copy and add system prompt
        messages = self.chat_history.copy()
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        text = self.tokenizer.apply_chat_template(
            messages,  # Use messages with system prompt
            tokenize=False,
            enable_thinking=self.enable_thinking,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
            'repetition_penalty': 1.2
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        end_time = time.time()
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        tokens_count = len(outputs[0]) - inputs.input_ids.shape[-1]
        generation_time = end_time - start_time
        
        self.add_to_history("assistant", response)
        return response, tokens_count, generation_time


# 4. 创建 GUI 应用
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("llm 聊天助手")
        self.root.geometry("1280x720")
        self.root.minsize(600, 500)
        self.root.configure(bg="#f0f0f0")

        # 初始化模型状态
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.pending_input = None  # 用于缓存用户输入
        self.config_modified = False  # New config modified flag
        
        # Add system_prompt to config section
        self._create_config_section()
        self._setup_config_tracking()
        self._load_config()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        # 对话历史显示区域
        self.chat_area = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            state='disabled',
            bg="#ffffff",
            font=("微软雅黑", 12),
            height=15
        )
        self.chat_area.pack(padx=10, pady=10, expand=True, fill='both')

        # 定义颜色标签
        self.chat_area.tag_configure("user", background="white", foreground="blue")
        self.chat_area.tag_configure("assistant", background="lightgray", foreground="green")
        self.chat_area.tag_configure("generating", background="lightgray", foreground="gray")
        self.chat_area.tag_configure("speed", background="lightblue", foreground="black")
        # Add system prompt tag with yellow background
        self.chat_area.tag_configure("system", background="#FFF3CD", foreground="black")

        # 用户输入区域
        self._create_input_section()

        # 按钮区域（并排显示）
        self._create_button_section()

    def _create_config_section(self):
        config_frame = tk.LabelFrame(self.root, text="模型配置", font=("微软雅黑", 12), padx=5, pady=5)
        config_frame.pack(padx=10, pady=10, fill='x')

        # MODEL_PATH (模型路径)
        tk.Label(config_frame, text="MODEL_PATH (模型路径)", font=("微软雅黑", 12)).grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.model_path_var = tk.StringVar(value="")
        self.model_path_entry = tk.Entry(config_frame, textvariable=self.model_path_var, font=("微软雅黑", 12))
        self.model_path_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.model_path_button = tk.Button(
            config_frame,
            text="浏览...",
            command=self.select_model_path,
            font=("微软雅黑", 12)
        )
        self.model_path_button.grid(row=0, column=2, padx=5, pady=2)

        # 增：ADAPTER_PATH (PEFT Adapter 路径)
        tk.Label(config_frame, text="ADAPTER_PATH (Adapter 路径)", font=("微软雅黑", 12)).grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.adapter_path_var = tk.StringVar(value="")
        self.adapter_path_entry = tk.Entry(config_frame, textvariable=self.adapter_path_var, font=("微软雅黑", 12))
        self.adapter_path_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        self.adapter_path_button = tk.Button(
            config_frame,
            text="浏览...",
            command=self.select_adapter_path,
            font=("微软雅黑", 12)
        )
        self.adapter_path_button.grid(row=1, column=2, padx=5, pady=2)

        # max_new_tokens (最大生成长度)
        tk.Label(config_frame, text="max_new_tokens (最大生成长度)", font=("微软雅黑", 12)).grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.max_new_tokens_var = tk.StringVar(value="32768")
        self.max_new_tokens_entry = tk.Entry(config_frame, textvariable=self.max_new_tokens_var, font=("微软雅黑", 12))
        self.max_new_tokens_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=2)

        # temperature (温度)
        tk.Label(config_frame, text="temperature (温度)", font=("微软雅黑", 12)).grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.temperature_var = tk.StringVar(value="0.7")
        self.temperature_entry = tk.Entry(config_frame, textvariable=self.temperature_var, font=("微软雅黑", 12))
        self.temperature_entry.grid(row=3, column=1, sticky='ew', padx=5, pady=2)

        # top_p (Top-p)
        tk.Label(config_frame, text="top_p (Top-p)", font=("微软雅黑", 12)).grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.top_p_var = tk.StringVar(value="0.9")
        self.top_p_entry = tk.Entry(config_frame, textvariable=self.top_p_var, font=("微软雅黑", 12))
        self.top_p_entry.grid(row=4, column=1, sticky='ew', padx=5, pady=2)

        # enable_thinking (启用思考模式)
        tk.Label(config_frame, text="enable_thinking (启用思考模式)", font=("微软雅黑", 12)).grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.enable_thinking_var = tk.BooleanVar(value=True)
        self.enable_thinking_checkbox = tk.Checkbutton(config_frame, variable=self.enable_thinking_var)
        self.enable_thinking_checkbox.grid(row=5, column=1, sticky='w', padx=5, pady=2)

        # Modify system_prompt entry to ComboBox and add add button
        tk.Label(config_frame, text="system_prompt (系统提示)", font=("微软雅黑", 12)).grid(row=6, column=0, sticky='w', padx=5, pady=2)
        self.system_prompt_var = tk.StringVar(value="")
        self.system_prompt_combo = ttk.Combobox(config_frame, textvariable=self.system_prompt_var, font=("微软雅黑", 12))
        self.system_prompt_combo.grid(row=6, column=1, sticky='ew', padx=5, pady=2)
        
        # Add button to save current prompt to list
        self.add_prompt_btn = tk.Button(
            config_frame,
            text="+",
            command=self._add_system_prompt,
            font=("微软雅黑", 10)
        )
        self.add_prompt_btn.grid(row=6, column=2, padx=5, pady=2)
        
        # Initialize system prompt list
        self.system_prompt_list = []

    def _setup_config_tracking(self):
        def mark_modified(*args):
            self.config_modified = True
        self.model_path_var.trace_add('write', mark_modified)
        self.max_new_tokens_var.trace_add('write', mark_modified)
        self.temperature_var.trace_add('write', mark_modified)
        self.top_p_var.trace_add('write', mark_modified)
        self.enable_thinking_var.trace_add('write', mark_modified)
        self.system_prompt_var.trace_add('write', mark_modified)
        # 增：追踪 adapter 路径修改
        self.adapter_path_var.trace_add('write', mark_modified)

    def _load_config(self):
        config_path = CONFIG_FILE_PATH
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.model_path_var.set(config.get('model_path', ''))
                self.max_new_tokens_var.set(str(config.get('max_new_tokens', '32768')))
                self.temperature_var.set(str(config.get('temperature', '0.7')))
                self.top_p_var.set(str(config.get('top_p', '0.9')))
                self.enable_thinking_var.set(config.get('enable_thinking', True))
                self.system_prompt_list = config.get('system_prompt', [])
                self.system_prompt_combo['values'] = self.system_prompt_list
                self.config_modified = False
            except Exception as e:
                self._update_chat_area(f"加载配置失败: {str(e)}\n", "generating")

    def _save_config(self):
        config = {
            'model_path': self.model_path_var.get().strip(),
            'max_new_tokens': self.max_new_tokens_var.get(),
            'temperature': self.temperature_var.get(),
            'top_p': self.top_p_var.get(),
            'enable_thinking': self.enable_thinking_var.get(),
            'system_prompt': self.system_prompt_list
        }
        config_path = os.path.join(os.path.dirname(__file__), "EasyChatLLM.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.config_modified = False
        except Exception as e:
            self._update_chat_area(f"保存配置失败: {str(e)}\n", "generating")

    def _create_input_section(self):
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(padx=10, pady=10, fill='x')

        self.input_area = tk.Text(self.input_frame, height=5, font=("微软雅黑", 12), wrap=tk.WORD)
        self.input_area.pack(side=tk.LEFT, expand=True, fill='x')

        self.scrollbar = ttk.Scrollbar(self.input_frame, command=self.input_area.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill='y')
        self.input_area.configure(yscrollcommand=self.scrollbar.set)

        # 绑定回车键事件
        self.input_area.bind("<Return>", self.on_enter_key)

    def _create_button_section(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        self.send_button = tk.Button(
            button_frame,
            text="发送",
            command=self.on_send,
            font=("微软雅黑", 12),
            bg="#4CAF50",
            fg="white"
        )
        self.send_button.pack(side='left', padx=5)

        self.clear_button = tk.Button(
            button_frame,
            text="清空",
            command=self.on_clear,
            font=("微软雅黑", 12),
            bg="#FFA726",
            fg="white"
        )
        self.clear_button.pack(side='left', padx=5)

        self.quit_button = tk.Button(
            button_frame,
            text="退出",
            command=self.on_quit,
            font=("微软雅黑", 12),
            bg="#F44336",
            fg="white"
        )
        self.quit_button.pack(side='left', padx=5)

    def on_enter_key(self, event):
        # 区分 Enter（发送）和 Shift+Enter（换行）
        if event.state == 0:  # 无修饰键（纯 Enter）
            self.on_send()
            return "break"
        elif event.state == 4:  # Shift+Enter（换行）
            self.input_area.insert(tk.INSERT, "\n")
            return "break"
        return None

    def on_send(self):
        # 如果模型未加载，则先加载模型
        if self.pipeline is None:
            try:
                model_path = self.model_path_var.get().strip()
                if not model_path:
                    self._update_chat_area("错误：模型路径不能为空。\n", "generating")
                    return

                max_new_tokens = int(self.max_new_tokens_var.get())
                temperature = float(self.temperature_var.get())
                top_p = float(self.top_p_var.get())
                enable_thinking = self.enable_thinking_var.get()

                # 验证参数范围
                if max_new_tokens <= 0:
                    self._update_chat_area("错误：最大生成长度必须为正整数。\n", "generating")
                    return
                if not (0 < temperature <= 1.0):
                    self._update_chat_area("错误：温度应在 (0, 1] 范围内。\n", "generating")
                    return
                if not (0 < top_p <= 1.0):
                    self._update_chat_area("错误：Top-p 应在 (0, 1] 范围内。\n", "generating")
                    return

                # 获取并缓存用户输入
                user_input = self.input_area.get("1.0", tk.END).strip()
                if not user_input:
                    return
                self.pending_input = user_input
                self.input_area.delete("1.0", tk.END)

                # 启动线程加载模型
                self._update_chat_area("正在加载模型，请稍候...\n", "generating")
                self.root.update_idletasks()

                thread = threading.Thread(
                    target=self._load_model_and_pipeline,
                    args=(model_path, max_new_tokens, temperature, top_p, enable_thinking)
                )
                thread.start()

                # 禁用配置输入框
                self._disable_config_inputs()

            except ValueError as e:
                self._update_chat_area(f"参数错误: {str(e)}\n", "generating")
                return
            except Exception as e:
                self._update_chat_area(f"模型加载失败: {str(e)}\n", "generating")
                return
        else:
            # 如果模型已加载，直接处理用户输入
            self._process_user_input()

    def _process_user_input(self, user_input=None):
        if user_input is None:
            user_input = self.input_area.get("1.0", tk.END).strip()
        if not user_input:
            return

        # 添加用户消息到历史和显示
        self.pipeline.add_to_history("user", user_input)
        self._update_chat_area(f"用户: {user_input}\n", "user")

        # 清空输入框
        self.input_area.delete("1.0", tk.END)

        # 显示生成中提示
        self._update_chat_area("助手: 正在生成...\n", "generating")
        self.chat_area.see(tk.END)  # 自动滚动到底部

        # 启动线程生成回答
        self.root.after(100, self._start_generation)

    def _load_model_and_pipeline(self, model_path, max_new_tokens, temperature, top_p, enable_thinking):
        try:
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            )
            # 增：如果指定了 adapter 路径，则加载 PEFT adapter
            adapter_path = self.adapter_path_var.get().strip()
            if adapter_path:
                from peft import PeftModel
                # 将 adapter 加载到主模型上
                self.model = PeftModel.from_pretrained(self.model, adapter_path, torch_dtype="auto")
                self.model = self.model.merge_and_unload()
                print('Adapter loaded successfully!'+adapter_path)
            # 初始化对话管道
            self.pipeline = ChatPipeline(self.model, self.tokenizer, max_new_tokens, temperature, top_p, enable_thinking)
            # 在主线程中更新UI
            self.root.after(0, self._on_model_loaded)
        except Exception as e:
            self.root.after(0, lambda: self._update_chat_area(f"模型加载失败: {str(e)}\n", "generating"))

    def _on_model_loaded(self):
        self._update_chat_area("模型加载完成。\n", "generating")
        self.chat_area.see(tk.END)
        # 处理缓存的用户输入
        if hasattr(self, 'pending_input'):
            self._process_user_input(self.pending_input)
            del self.pending_input

    def _start_generation(self):
        # 使用线程执行生成
        thread = threading.Thread(target=self._generate_and_display_response)
        thread.start()

    def _generate_and_display_response(self):
        system_prompt = self.system_prompt_var.get().strip()
        # Display system prompt with special tag
        if system_prompt:
            self.root.after(0, self._update_chat_area, f"系统提示: {system_prompt}\n", "system")
        response, tokens_count, generation_time = self.pipeline.generate_response(
            system_prompt=system_prompt if system_prompt else None
        )
        self.root.after(0, self._update_chat_area, f"助手: {response}\n", "assistant")
        self.root.after(0, self._update_chat_area, f"[速度: {tokens_count:.0f} tokens in {generation_time:.2f}s] ({tokens_count / generation_time:.2f} tokens/s)\n", "speed")

    def _update_chat_area(self, message, tag):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, message, tag)
        self.chat_area.configure(state='disabled')
        self.chat_area.see(tk.END)  # 自动滚动到底部

    def _disable_config_inputs(self):
        self.model_path_entry.config(state='disabled')
        self.max_new_tokens_entry.config(state='disabled')
        self.temperature_entry.config(state='disabled')
        self.top_p_entry.config(state='disabled')
        self.enable_thinking_checkbox.config(state='disabled')

    def select_model_path(self):
        path = filedialog.askdirectory()
        if path:
            self.model_path_var.set(path)
       # 增：选择 PEFT adapter 路径
    def select_adapter_path(self):
        path = filedialog.askdirectory()
        if path:
            self.adapter_path_var.set(path)

    def on_clear(self):
        self.pipeline.clear_history()
        self.chat_area.configure(state='normal')
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.configure(state='disabled')

    def on_quit(self):
        if self.config_modified:
            self._save_config()
        self.root.destroy()

    def _add_system_prompt(self):
        new_prompt = self.system_prompt_var.get().strip()
        if new_prompt and new_prompt not in self.system_prompt_list:
            self.system_prompt_list.append(new_prompt)
            self.system_prompt_combo['values'] = self.system_prompt_list
            self.system_prompt_var.set('')
            self.config_modified = True
            # Force save when adding new prompt
            self._save_config()


# 启动 GUI 应用
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()