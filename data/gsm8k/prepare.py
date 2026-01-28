# saves the GSM8K dataset to a binary file for training
# GSM8K is a dataset of grade school math word problems
# Available at: https://huggingface.co/datasets/gsm8k

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets

# number of workers in .map() call
num_proc = 8

# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

def format_example(question, answer):
    """
    格式化 GSM8K 範例為訓練格式
    可以選擇不同的格式：
    1. 問題 + 答案（完整推理）
    2. 問題 + 簡短答案
    3. 指令格式
    """
    # 格式1: 完整的問題和推理過程
    return f"Question: {question}\nAnswer: {answer}"
    
    # 格式2: 指令格式（可選）
    # return f"### Instruction:\nSolve the following math problem.\n\n### Question:\n{question}\n\n### Answer:\n{answer}"

if __name__ == '__main__':
    # Load GSM8K dataset from Hugging Face
    print("Loading GSM8K dataset...")
    
    # GSM8K 有 'main' 配置，包含 train 和 test splits
    dataset_train = load_dataset("openai/gsm8k", "main", split="train", num_proc=num_proc_load_dataset)
    dataset_test = load_dataset("openai/gsm8k", "main", split="test", num_proc=num_proc_load_dataset)
    
    # GSM8K 的 train split 有 7473 個樣本，test split 有 1319 個樣本
    # 我們將 train 分為 train/val，並保留官方 test 集
    split_dataset = dataset_train.train_test_split(test_size=0.1, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    
    # 如果想使用官方 test set 作為 validation，可以取消註釋下面這行：
    # split_dataset['val'] = dataset_test

    print("Dataset split created:")
    print(f"  train: {len(split_dataset['train'])} examples")
    print(f"  val: {len(split_dataset['val'])} examples")
    print(f"  official test: {len(dataset_test)} examples (not used in training)")

    # Define tokenization function
    def process(example):
        # GSM8K 的欄位是 'question' 和 'answer'
        formatted_text = format_example(example['question'], example['answer'])
        ids = enc.encode_ordinary(formatted_text)  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token (50256 for gpt2 bpe)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_train = split_dataset['train'].map(
        process, 
        remove_columns=['question', 'answer'], 
        desc="tokenizing train", 
        num_proc=num_proc
    )
    tokenized_val = split_dataset['val'].map(
        process, 
        remove_columns=['question', 'answer'], 
        desc="tokenizing val", 
        num_proc=num_proc
    )

    # More efficient approach: pre-allocate array and fill it
    print("Writing tokens to binary files...")
    
    # Process train split
    train_len = sum(tokenized_train['len'])
    print(f"train will have {train_len:,} tokens")
    train_ids = np.memmap(os.path.join(os.path.dirname(__file__), 'train.bin'), 
                          dtype=np.uint16, mode='w+', shape=(train_len,))
    idx = 0
    for token_ids in tqdm(tokenized_train['ids'], desc='writing train'):
        train_ids[idx : idx + len(token_ids)] = token_ids
        idx += len(token_ids)
    train_ids.flush()
    
    # Process val split
    val_len = sum(tokenized_val['len'])
    print(f"val will have {val_len:,} tokens")
    val_ids = np.memmap(os.path.join(os.path.dirname(__file__), 'val.bin'), 
                        dtype=np.uint16, mode='w+', shape=(val_len,))
    idx = 0
    for token_ids in tqdm(tokenized_val['ids'], desc='writing val'):
        val_ids[idx : idx + len(token_ids)] = token_ids
        idx += len(token_ids)
    val_ids.flush()
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'tokenizer': 'gpt2'
    }
    import pickle
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Done! Files saved to {os.path.dirname(__file__)}")
    print("\nDataset statistics:")
    print(f"  Total training tokens: {train_len:,}")
    print(f"  Total validation tokens: {val_len:,}")
    print(f"  Average tokens per training example: {train_len/len(tokenized_train):.1f}")
    print(f"  Average tokens per validation example: {val_len/len(tokenized_val):.1f}")
