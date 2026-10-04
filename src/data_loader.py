import json
import re
from typing import List, Dict, Any, Optional
import random

class DataLoader:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        
    def load_mmlu(self, subset: Optional[str] = None, max_samples: Optional[int] = None) -> List[Dict]:
        """
        加载MMLU数据
        Args:
            subset: 子任务名称，如 'abstract_algebra'
            max_samples: 最大样本数（用于快速测试）
        Returns:
            List of dict: [{"question": "...", "choices": ["A. ...", ...], "answer": 0}]
        """
        with open(f"{self.raw_data_path}/mmlu/mmlu_subset.json", "r") as f:
            data = json.load(f)
        
        samples = []
        for sub, items in data.items():
            if subset and sub != subset:
                continue
            for item in items:
                # MMLU的choices是列表，answer是索引
                samples.append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "subset": sub
                })
        
        if max_samples:
            samples = random.sample(samples, min(max_samples, len(samples)))
        return samples
    
    def load_gsm8k(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        加载GSM8K数据
        Args:
            max_samples: 最大样本数
        Returns:
            List of dict: [{"question": "...", "answer": "The answer is 42"}]
        """
        with open(f"{self.raw_data_path}/gsm8k/gsm8k_test.json", "r") as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            # 提取GSM8K的答案（数字）
            answer_text = item["answer"]
            # 答案格式通常是 "#### 42"，提取数字
            match = re.search(r"####\s*([\d,]+\.?\d*)", answer_text)
            if match:
                numeric_answer = float(match.group(1).replace(",", ""))
            else:
                numeric_answer = None
            
            samples.append({
                "question": item["question"],
                "answer_raw": answer_text,
                "answer_numeric": numeric_answer
            })
        
        if max_samples:
            samples = random.sample(samples, min(max_samples, len(samples)))
        return samples
    
    def clean_question(self, text: str) -> str:
        """清洗问题文本（去除多余空格、特殊字符等）"""
        text = re.sub(r"\s+", " ", text)  # 合并多余空格
        text = text.strip()
        return text
    
    def format_mmlu_prompt(self, question: str, choices: List[str]) -> str:
        """
        将MMLU问题格式化为模型输入
        例如: "Question: xxx\nA. xxx\nB. xxx\nC. xxx\nD. xxx\nAnswer:"
        """
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        return f"Question: {question}\n{choices_str}\nAnswer:"
    
    def format_gsm8k_prompt(self, question: str) -> str:
        """将GSM8K问题格式化为模型输入"""
        return f"Question: {question}\nPlease provide the final answer as a number only."
```

