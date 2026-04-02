import torch
import os
import json
import numpy as np
import hyperparams as hp
from data_load import load_vocab_json
from model import Graph


def load_latest_checkpoint(model, logdir):
    """自动寻找最新的模型权重"""
    ckpt_files = [f for f in os.listdir(logdir) if f.endswith(".pth")]
    if not ckpt_files:
        return None
    # 按修改时间排序，取最新的一个
    latest_ckpt = sorted(
        ckpt_files, key=lambda x: os.path.getmtime(os.path.join(logdir, x))
    )[-1]
    ckpt_path = os.path.join(logdir, latest_ckpt)
    print(f"正在加载模型权重: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return latest_ckpt


def eval_interactive():
    # 1. 加载词表
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab_json()
    pnyn_vocab_size = len(pnyn2idx)
    hanzi_vocab_size = len(hanzi2idx)

    # 2. 初始化模型并设为评估模式
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = Graph(pnyn_vocab_size, hanzi_vocab_size)

    if not load_latest_checkpoint(model, hp.logdir):
        print(f"错误: 在 {hp.logdir} 没找到任何 .pth 模型文件，请先训练模型。")
        return

    model.to(device)
    model.eval()

    print("\n" + "=" * 30)
    print("拼音输入法模型已就绪！")
    print("输入拼音 (如: 'woaibeijing')，按回车推理。")
    print("输入 'q' 或 'quit' 退出程序。")
    print("=" * 30 + "\n")

    while True:
        line = input("拼音输入 >> ").strip().lower()
        if line in ["q", "quit"]:
            break
        if not line:
            continue

        # 将输入字符串转换为单个字母列表（与原 TensorFlow 行为一致）
        pnyn_sent = list(
            line
        )  # e.g. "woaibeijing" -> ['w','o','a','i','b','e','i','j','i','n','g']
        x_ids = [pnyn2idx.get(ch, 1) for ch in pnyn_sent]  # 1 表示 OOV（未知字母）

        # 填充或截断到 maxlen
        if len(x_ids) < hp.maxlen:
            x_ids += [0] * (hp.maxlen - len(x_ids))
        else:
            x_ids = x_ids[: hp.maxlen]

        x_tensor = torch.LongTensor([x_ids]).to(device)

        with torch.no_grad():
            preds = model(x_tensor)  # 假设输出形状 (1, T)

        # 后处理：获取有效长度（非填充的输入字母个数）
        valid_len = np.count_nonzero(x_ids)  # 原 TensorFlow 使用 np.count_nonzero(xx)
        res_ids = preds[0].detach().cpu().numpy()

        # 转换为汉字，遇到 padding (0) 停止，且不超过 valid_len
        got_chars = []
        for i, idx in enumerate(res_ids):
            if idx == 0 or i >= valid_len:
                break
            char = idx2hanzi.get(str(idx), "")  # 注意 key 可能需要字符串形式
            if char == "_":  # 原代码替换 "_" 为空
                continue
            got_chars.append(char)

        got = "".join(got_chars)
        print(f"模型输出 << {got}")


if __name__ == "__main__":
    eval_interactive()
