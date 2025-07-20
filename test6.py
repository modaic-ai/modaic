from transformers import PretrainedConfig

class ToyConfig(PretrainedConfig):
    model_type = "toy"

    def __init__(
        self,
        x: int = 256,
        y: int = 4,
        z: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)          # 1️⃣ base class consumes / sets leftover keys
        self.hidden_size       = x
        self.num_hidden_layers = y
        self.dropout           = z
        self.other_attr        = x+ y   # 2️⃣

config = ToyConfig(2, 3, 0.1)
print(config)
# config.save_pretrained("test6.json")
loaded_config = ToyConfig.from_pretrained("test6.json")
print(loaded_config)
