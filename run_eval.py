"""
Mini-Eval MVP - Single-file closed loop
Usage: python run_eval.py --limit 20


import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- 1. Data loading ----------

PROMPT_TEMPLATE = """Solve the following math problem. Show your reasoning, and put the final answer after "####".

Question: {question}
Answer:"""


def extract_answer(text: str) -> str:
    """Extract the final numeric answer from a GSM8K reference or model output."""
    # Prefer the number after "####"
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: take the last number found
    nums = re.findall(r"\-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else ""


def load_gsm8k(limit: int):
    """Return a list of dicts: {question, gold}"""
    ds = load_dataset("gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for row in ds:
        samples.append({
            "question": row["question"],
            "gold": extract_answer(row["answer"]),
        })
    return samples


# ---------- 2. Model inference ----------

class ModelWrapper:
    def __init__(self, model_name: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
        outputs = []
        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.model.device)
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # greedy decoding for reproducibility
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Strip the prompt tokens
            new_tokens = gen[0][inputs["input_ids"].shape[1]:]
            outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs


# ---------- 3. Evaluation ----------

def evaluate(preds: list[str], golds: list[str]) -> dict:
    results = []
    correct = 0
    for pred, gold in zip(preds, golds):
        pred_ans = extract_answer(pred)
        is_correct = (pred_ans == gold)
        correct += is_correct
        results.append({
            "pred_raw": pred,
            "pred_ans": pred_ans,
            "gold": gold,
            "correct": is_correct,
        })
    return {
        "accuracy": correct / len(results) if results else 0.0,
        "num_correct": correct,
        "num_total": len(results),
        "details": results,
    }


# ---------- 4. Main pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of samples to evaluate; 0 means all")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    # 1) Data
    print(f"[1/4] Loading GSM8K (limit={args.limit}) ...")
    samples = load_gsm8k(args.limit)
    print(f"      Total: {len(samples)} samples")

    # 2) Model
    print(f"[2/4] Loading model {args.model} ...")
    model = ModelWrapper(args.model)

    # 3) Inference
    print(f"[3/4] Running inference ...")
    prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in samples]
    preds = model.generate(prompts, max_new_tokens=args.max_new_tokens)

    # 4) Evaluation
    print(f"[4/4] Evaluating ...")
    golds = [s["gold"] for s in samples]
    report = evaluate(preds, golds)

    print("\n" + "=" * 50)
    print(f"Model:     {args.model}")
    print(f"Dataset:   GSM8K (n={report['num_total']})")
    print(f"Accuracy:  {report['accuracy']:.2%}  ({report['num_correct']}/{report['num_total']})")
    print("=" * 50)

    # Print a few error cases for debugging
    wrong = [d for d in report["details"] if not d["correct"]][:3]
    if wrong:
        print("\n--- Error cases (first 3) ---")
        for i, w in enumerate(wrong, 1):
            print(f"\n[{i}] gold={w['gold']}  pred={w['pred_ans']}")
            print(f"    Model output tail: ...{w['pred_raw'][-200:]}")

    # Save to disk
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()

"""

#########################

def extract_answer(text):
    '''
    调用 re.search(pattern, string, flags=0) 
    方法： pattern：正则表达式模式。 string：需要匹配的目标字符串。 
    flags（可选）：控制匹配方式的标志，例如忽略大小写（re.I）或多行模式（re.M）。
    '''
   m = re.search(r'####\s*([\-\d,\.]+)', text) 
   if m:#如果找到了，提取并清理答案
   '''group(0) 是整个匹配（如 "#### 18"）

group(1) 是第一个捕获组（括号里的内容，如 "18"）replace用来把字符串里的逗号去掉
strip()去掉首位空白'''
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"\-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else ""

def load_gsm8k(limit):
     ds = load_dataset("gsm8k","main", split = "test")#只加载测试集
     ''' if limit:
        ds = ds.select(range(min(limit, len(ds))))
     '''
     samples = []
     for row in ds:
        samples.append({
            "question": row["question"],
            "gold":extract_answer(row["answer"]),
        }

        )
        return samples


class ModelWrapper:
    def __init__(self, model_name, device='auto'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
           # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=512):
        outputs = []
        for p in prompts:#无banch
            #input = self.tokenizer(p, return_tensors = 'pt')
            inputs = self.tokenizer(p, return_tensors="pt").to(self.model.device)
            gen = self.model.generate(
               ''' input_ids = inputs[input_ids],
                attention_mask = inputs.get("attention_mask"),'''
                **inputs,
                max_new_tokens = max_new_tokens,
                '''
                 do_sample=False,
                 pad_token_id=self.tokenizer.eos_token_id,
                '''

            )
            new_tokens = gen[0][inputs["input_ids"].shape[1]:]
            outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens = True))
            return outputs

    
def evaluate(preds, golds):
    results = []
    correct = 0
    for pred, gold in zip(preds, golds):
        pred_ans = extract_answer(pred)
        if(pred_ans == gold) correct++
        results.append({
            "pred": pred, #模型原始输出
            "pred_ans": pred_ans,#提取后的答案
            "gold": gold,#标准答案


        } )



def main():
'''
    ds = load_dataset("gsm8k","main", split = "test")#
    
    #先跑20条 limit
    ds = ds.select(range(20))
但每条数据里，answer 字段是 "...推理过程... #### 42"，我不能拿一整段去比。

→ "我得从里面抽出 42。写个函数。"

到这里，我发现我需要一个工具函数 extract_answer。 于是我开始暂时离开 main，去写这个小函数：
'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", default=100)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    
    args = parser.parse_args()


    print("----Loading GSM8K----")
    samples = load_gsm8k(args.limit)
    print(f"Total: {len(samples)} samples")

    print("----Loading Model----")
    model = ModelWrapper(args.model)


    print("----Running inference----") #正在推理：让模型跑一边，生成输出结果
    #把题目包装成prompt
    prompts =[]
    for s in samples:
         p = PROMPT_TEMPLATE.format(question=s["question"])
         prompts.append(p)
    preds = model.generate(prompts, max_new_tokens = max_new_tokens)
    
    print("----Evaluating----")
    


if __name__ == "__main__":
    main()