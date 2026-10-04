from datasets import load_dataset
import json
import os

def download_mmlu():
    """MULL"""
    print("Downloading MMLU ...")

    subsets = ['abstract_algebra', 'college_biology', 'high_school_math', 'world_religions']
    data = {}
    for subset in subsets:
        dataset = load_dataset("cais/mmlu", subset, split="validation")
        data[subset] = [{"question": item["question"], 
                         "choices": item["choices"], 
                         "answer": item["answer"]} for item in dataset]
    
    os.makedirs("data/raw/mmlu", exist_ok=True)
    with open("data/raw/mmlu/mmlu_subset.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"MMLU finished, {len(subsets)} tasks")

def download_gsm8k():
    """GSM8K"""
    print("Downloading GSM8K ...")
    dataset = load_dataset("gsm8k", "main", split="test")
    data = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
    
    os.makedirs("data/raw/gsm8k", exist_ok=True)
    with open("data/raw/gsm8k/gsm8k_test.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"GSM8K finished,  {len(data)} datas")

if __name__ == "__main__":
    download_mmlu()
    download_gsm8k()
```
