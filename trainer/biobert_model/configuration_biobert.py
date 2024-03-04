from transformers import PretrainedConfig


class BioBertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        hidden_size1: int = 256,
        hidden_size2: int = 32,
        hidden_size3: int = 16,
        hidden3_dropout: float = 0.2,
        unfreeze: bool = False,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        **kwargs,
    ):
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden3_dropout = hidden3_dropout
        self.unfreeze = unfreeze
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        super().__init__(**kwargs)