class LazyVocabulary:
    def __init__(self, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = ['<unk>']
        self.reserved_tokens = reserved_tokens

        self.token_to_id = {}
        self.id_to_token = []

        for token in self.reserved_tokens:
            self._add_token(token)

    def _add_token(self, token: str):
        """内部方法：添加 token 到词汇表（不检查重复）"""
        if token not in self.token_to_id:
            idx = len(self.id_to_token)
            self.token_to_id[token] = idx
            self.id_to_token.append(token)

    def append(self, token: str):
        """添加新 token（若已存在则忽略）"""
        if token not in self.token_to_id:
            self._add_token(token)

    def __len__(self):
        return len(self.id_to_token)

    def __contains__(self, token: str):
        return token in self.token_to_id

    def get_id(self, token: str):
        """返回 token 对应的索引，若不存在则返回 <unk> 的索引"""
        return self.token_to_id.get(token, self.token_to_id['<unk>'])

    def get_token(self, idx: int):
        """根据索引返回 token"""
        return self.id_to_token[idx]

    def encode(self, tokens):
        """将 token 列表转换为索引列表（未知词映射为 <unk>）"""
        return [self.get_id(t) for t in tokens]

    def decode(self, indices):
        """将索引列表转换回 token 列表"""
        return [self.get_token(i) for i in indices]

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)}, reserved={self.reserved_tokens})"

class Vocabulary:
    def __init__(self):
        self.id_to_token = ['<pad>']
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id['<pad>']

    def add_token(self, token: str):
        if token not in self.token_to_id:
            idx = len(self.id_to_token)
            self.token_to_id[token] = idx
            self.id_to_token.append(token)

    def __len__(self):
        return len(self.id_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]

