# model_utility.py

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PretrainedConfig
from typing import Union, Optional
from PIL import Image

############################################
# 1) import_model_class_from_model_name_or_path
############################################
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str = None):
    """
    Dynamically determines which text encoder class to load based on
    the model config's 'architectures' field.

    e.g., If the config says "CLIPTextModel", return that class from Transformers.
    """
    # If you store your config in a "subfolder='text_encoder'", it must match here
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        # If this import fails or warns, it might be for alt-diffusion
        try:
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
            return RobertaSeriesModelWithTransformation
        except ImportError:
            raise ValueError("Optional alt_diffusion model class not available.")
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported by import_model_class_from_model_name_or_path.")


############################################
# 2) tokenize_prompt
############################################
def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    """
    Uses the given tokenizer to convert a string (or list of strings) into token IDs.
    If tokenizer_max_length is specified, it overrides the default model_max_length.
    """
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return text_inputs


############################################
# 3) encode_prompt
############################################
def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    """
    Passes the token IDs (and optionally attention_mask) into the text_encoder
    and returns the last hidden states as prompt embeddings (shape [batch, seq_len, hidden_dim]).
    """
    text_input_ids = input_ids.to(text_encoder.device)
    attention_mask = attention_mask.to(text_encoder.device) if text_encoder_use_attention_mask else None

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    # By default, 'prompt_embeds' is a BaseModelOutputWithPooling or similar
    # We typically take [0] to get the last hidden state:
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


############################################
# 4) get_scheduler
############################################
def get_scheduler(
    name: Union[str, "SchedulerType"],
    optimizer: Optimizer,
    step_rules: Optional[str] = None,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Unified API to instantiate a learning rate scheduler by name (e.g. "constant_with_warmup").
    Requires a dict or function dispatch to map name (SchedulerType) to the appropriate
    creation function.

    Example usage:
        scheduler = get_scheduler(
            name="cosine_with_restarts",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
            num_cycles=2
        )
    """
    from accelerate.utils import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

    if not isinstance(name, SchedulerType):
        name = SchedulerType(name)

    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # e.g. 'constant'
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, last_epoch=last_epoch)

    # e.g. 'piecewise_constant'
    if name == SchedulerType.PIECEWISE_CONSTANT:
        return schedule_func(optimizer, step_rules=step_rules, last_epoch=last_epoch)

    # All others require num_warmup_steps
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires num_warmup_steps, please provide it.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, last_epoch=last_epoch)

    # All others also require num_training_steps
    if num_training_steps is None:
        raise ValueError(f"{name} requires num_training_steps, please provide it.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            last_epoch=last_epoch,
        )

    # Fallback for others like 'linear', 'cosine', etc.
    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch
    )