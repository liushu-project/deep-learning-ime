import re
import os
from xpinyin import Pinyin

pinyin = Pinyin()


def align(sent):
    pnyns = pinyin.get_pinyin(sent, " ").split()

    hanzis = []
    for char, p in zip(sent.replace(" ", ""), pnyns):
        hanzis.extend([char] + ["_"] * (len(p) - 1))

    pnyns = "".join(pnyns)
    hanzis = "".join(hanzis)

    assert len(pnyns) == len(hanzis), (
        "The hanzis and the pinyins must be the same in length."
    )
    return pnyns, hanzis


def clean(text):
    """清理文本，仅保留汉字、空格和指定标点"""
    # 排除包含英文和数字的句子
    if re.search(r"[A-Za-z0-9]", text):
        return ""
    # 仅保留汉字范围 \u4e00-\u9fa5 以及指定的标点符号
    text = re.sub(r"[^\u4e00-\u9fa5。，！？ ]", "", text)
    return text.strip()


def build_corpus():
    os.makedirs("data", exist_ok=True)

    input_file = "data/zho_news_2007-2009_1M-sentences.txt"
    output_file = "data/zh.tsv"

    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    with open(output_file, "w", encoding="utf-8") as fout:
        with open(input_file, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin, 1):
                try:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue

                    idx, sent = parts[0], parts[1]
                    sent = clean(sent)

                    if sent:
                        pnyns, hanzis = align(sent)
                        fout.write(f"{idx}\t{pnyns}\t{hanzis}\n")

                except Exception:
                    print(f"error line {i}")
                    # 语料库较大，跳过处理错误的行
                    continue

                if i % 10000 == 0:
                    print(f"已处理 {i} 行...")


if __name__ == "__main__":
    print("开始处理语料...")
    build_corpus()
    print("处理完成！")
