from collections import Counter
from itertools import chain
import hyperparams as hp
import json


def build_vocab():
    # pinyin
    if hp.isqwerty:
        pnyns = "EUabcdefghijklmnopqrstuvwxyz0123456789。，！？"  # E: Empty, U: Unknown
        pnyn2idx = {pnyn: idx for idx, pnyn in enumerate(pnyns)}
        idx2pnyn = {idx: pnyn for idx, pnyn in enumerate(pnyns)}
    else:
        pnyn2idx, idx2pnyn = dict(), dict()
        pnyns_list = [
            "E",
            "U",
            "abc",
            "def",
            "ghi",
            "jkl",
            "mno",
            "pqrs",
            "tuv",
            "wxyz",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "。",
            "，",
            "！",
            "？",
        ]  # E: Empty, U: Unknown
        for i, pnyns in enumerate(pnyns_list):
            for pnyn in pnyns:
                pnyn2idx[pnyn] = i

    # hanzis
    hanzi_sents = [
        line.split("\t")[2]
        for line in open("data/zh.tsv", "r", encoding="utf-8").read().splitlines()
    ]
    hanzi2cnt = Counter(chain.from_iterable(hanzi_sents))
    hanzis = [
        hanzi for hanzi, cnt in hanzi2cnt.items() if cnt > 5
    ]  # remove long-tail characters

    hanzis.remove("_")
    hanzis = ["E", "U", "_"] + hanzis  # 0: empty, 1: unknown, 2: blank
    hanzi2idx = {hanzi: idx for idx, hanzi in enumerate(hanzis)}
    idx2hanzi = {idx: hanzi for idx, hanzi in enumerate(hanzis)}

    if hp.isqwerty:
        json.dump(
            (pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi),
            open("data/vocab.qwerty.json", "w"),
        )
    else:
        json.dump(
            (pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi),
            open("data/vocab.nine.json", "w"),
        )


if __name__ == "__main__":
    build_vocab()
