from datasets import load_from_disk
import re,lzma,json

match_numbers = re.compile(
    "boxed{" + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
def extract_numbers(text):
    guess=match_numbers.search(text)
    return guess.group(1) if guess else ""

def load_data(file_path):
    if file_path.endswith(".lzma"):
        with lzma.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        ds = load_from_disk(file_path)
        return [dict(item) for item in ds]

def main():
    # 获取用户输入
    input_path = input("请输入 datasets 数据文件夹路径：").replace('"','')
    #positive_label = input("请输入正类标签（如 'human' 或 'llm'）：")

    # 加载数据集（假设为通过 save_to_disk 保存的文件夹）
    dataset = load_data(input_path)

    # 假设使用 'train' 分割，若不确定可让用户选择或遍历所有 splits
    # if 'train' in dataset:
    #     dataset = dataset['train']
    # else:
    #     raise ValueError("数据集中未找到 'train' 分割，请确认数据结构。")

    # 预定义参考标签（用户需根据实际场景修改）
    #refer_labels = ['human','llm']  # 示例标签，请根据实际需求修改

    # 初始化计数器
    # tp = 0  # 真阳性 (True Positive)
    # fp = 0  # 假阳性 (False Positive)
    # fn = 0  # 假阴性 (False Negative)
    perfect_correct_count = 0#给出答案
    maybe_correct_count = 0#没给出答案但是推理包含（可能格式错误）
    error_label_count = 0  #格式错误计数
    total_chars=0
    total_samples = 0

    # 预定义参考标签（假设为 ['0', '1'] 或 ['A', 'B']）
    #refer_labels = ['0', '1']  # 示例标签，请根据实际需求修改

    # 遍历数据集
    for sample in dataset:
        total_samples += 1
        output = sample.get('output', '').strip()
        predict_output = sample.get('predict_content', '').strip()
        predict_reason=sample.get('predict_reason', '')
        total_chars+=len(predict_reason)
        
        is_error_format=False
        try:
            if predict_output!="":
                if float(predict_output.strip().replace(",",""))==float(output.strip().replace(",","")):
                    perfect_correct_count += 1
                    maybe_correct_count += 1
            else:
                is_error_format=True
        except:
            is_error_format=True
        error_label_count+= 1 if is_error_format else 0
        
        if predict_output=="" and is_error_format:
            maybe_answer = extract_numbers(predict_reason)
            try:
                if maybe_answer!="":
                    if float(maybe_answer.strip().replace(",",""))==float(output.strip().replace(",","")):
                        maybe_correct_count += 1
            except:
                pass


    # 输出结果
    print(f"\n统计结果：")
    print(f"总样本数：{total_samples}")
    print(f"正确数：{perfect_correct_count}")
    print(f"格式错误正确回答数：{maybe_correct_count}")
    print(f"格式错误数：{error_label_count}")
    print(f"\n正确率：{(perfect_correct_count/total_samples):.2%}")
    print(f"格式错误正确回答率： {(maybe_correct_count/total_samples):.2%}")
    print(f"格式错误率： {(error_label_count/total_samples):.2%}")
    print(f"平均字符长度: {(total_chars/total_samples):.2f}")

if __name__ == "__main__":
    main()