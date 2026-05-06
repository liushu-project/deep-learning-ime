import json
from dataclasses import dataclass, fields

@dataclass
class Config:
    embed_size: int
    hidden_size: int
    num_layers: int
    batch_size: int
    num_steps: int
    num_epochs: int
    learning_rate: float
    dropout: float

    @classmethod
    def from_json(cls, file_path: str = './config.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

