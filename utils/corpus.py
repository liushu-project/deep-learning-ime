import random


def split_large_file(input_file, train_file, val_file, val_ratio=0.3, seed=42):
    random.seed(seed)
    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(train_file, "w", encoding="utf-8") as f_train,
        open(val_file, "w", encoding="utf-8") as f_val,
    ):
        for line in f_in:
            if random.random() < val_ratio:
                f_val.write(line)
            else:
                f_train.write(line)

    print(f"Split done: {train_file}, {val_file}")


if __name__ == "__main__":
    split_large_file("data/pinyin-zh.txt", "data/train.txt", "data/val.txt")
