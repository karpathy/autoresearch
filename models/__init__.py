"""Model registry for autoresearch."""
from models.nanochat import GPT as NanochatGPT, GPTConfig as NanochatConfig, MuonAdamW
from models.gpt2 import GPT2, GPT2Config

REGISTRY = {
    "nanochat": (NanochatGPT, NanochatConfig),
    "gpt2": (GPT2, GPT2Config),
}


def create_model(name, **config_overrides):
    """Create a model by name. Raises KeyError for unknown names."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {list(REGISTRY.keys())}")
    model_cls, config_cls = REGISTRY[name]
    config = config_cls(**config_overrides)
    return model_cls(config), config


def list_models():
    return list(REGISTRY.keys())
