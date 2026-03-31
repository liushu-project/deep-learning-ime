import torch
import os
import json
import numpy as np
import hyperparams as hp
from data_load import load_vocab_json
from model import Graph

def load_latest_checkpoint(model, logdir):
    """自动寻找最新的模型权重"""
    ckpt_files = [f for f in os.listdir(logdir) if f.endswith('.pth')]
    if not ckpt_files:
        return None
    # 按修改时间排序，取最新的一个
    latest_ckpt = sorted(ckpt_files, key=lambda x: os.path.getmtime(os.path.join(logdir, x)))[-1]
    ckpt_path = os.path.join(logdir, latest_ckpt)
    print(f"正在加载模型权重: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return latest_ckpt

def eval_interactive():
    # 1. 加载词表
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab_json()
    pnyn_vocab_size = len(pnyn2idx)
    hanzi_vocab_size = len(hanzi2idx)

    # 2. 初始化模型并设为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = Graph(pnyn_vocab_size, hanzi_vocab_size)
    
    if not load_latest_checkpoint(model, hp.logdir):
        print(f"错误: 在 {hp.logdir} 没找到任何 .pth 模型文件，请先训练模型。")
        return

    model.to(device)
    model.eval()

    print("\n" + "="*30)
    print("拼音输入法模型已就绪！")
    print("输入拼音 (如: 'wo ai bei jing')，按回车推理。")
    print("输入 'q' 或 'quit' 退出程序。")
    print("="*30 + "\n")

    while True:
        line = input("拼音输入 >> ").strip().lower()
        
        if line in ['q', 'quit']:
            break
        if not line:
            continue

        # 3. 预处理输入：拼音 -> 索引 ID
        # 假设你的模型训练时是按空格切分拼音单词的
        words = line.split()
        # 将拼音转换为 ID，不在词表里的用 <UNK> (假设 ID 为 1) 替代
        # 并根据 hp.maxlen 进行填充 (Padding)
        x_ids = [pnyn2idx.get(w, 1) for w in words]
        if len(x_ids) < hp.maxlen:
            x_ids += [0] * (hp.maxlen - len(x_ids))
        else:
            x_ids = x_ids[:hp.maxlen]

        x_tensor = torch.LongTensor([x_ids]).to(device)

        # 4. 模型推理
        with torch.no_grad():
            preds = model(x_tensor) # 返回 (1, T) 的 tensor
        
        # 5. 后处理：ID -> 汉字
        res_ids = preds[0].detach().cpu().numpy()
        res_chars = []
        for i, idx in enumerate(res_ids):
            # 遇到 padding (0) 就停止或者跳过
            if idx == 0: continue 
            # 对应的拼音位置超出了输入长度也停止
            if i >= len(words): break
            
            char = idx2hanzi.get(str(idx), "")
            res_chars.append(char)

        print(f"模型输出 << {''.join(res_chars)}")
        print("-" * 20)

if __name__ == "__main__":
    eval_interactive()
