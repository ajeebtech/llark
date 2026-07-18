# Copyright 2024 — LLark M2 PoC
# Licensed under the Apache License, Version 2.0
#
# m2t/models/qwen2.py
#
# Qwen2-based LLark model for M2 MacBook PoC.
# This is a near-identical port of llamav2.py with Qwen2 base classes
# substituted for Llama base classes. The audio injection logic in forward()
# is architecture-independent and is copied verbatim.

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from m2t.models import AudioEncoderConfig
from m2t.special_tokens import (
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
)


class WrappedQwen2Config(Qwen2Config):
    """Config container class for the Qwen2-based LLark model."""

    model_type = "wrapped_qwen2"
    # Default to 512 for CLAP embeddings (vs. 4800 for Jukebox in the paper)
    mm_hidden_size: int = 512


class WrappedQwen2Model(Qwen2Model):
    """
    Qwen2-based LLark model.

    Mirrors WrappedLlamav2Model but uses Qwen2Model as the base.
    The audio injection mechanism (MLP projector + token replacement in
    embedding space) is identical to the LLaMA version.
    """

    config_class = WrappedQwen2Config

    def __init__(self, config: Qwen2Config):
        super(WrappedQwen2Model, self).__init__(config)
        self.audio_encoder_config = AudioEncoderConfig()

    def initialize_adapter_modules(
        self,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=None,
        fsdp: bool = None,
    ):
        """
        Initialize the audio-to-LLM projection MLP.

        Args:
            pretrain_mm_mlp_adapter: optional path to pre-trained projector weights.
            tune_mm_mlp_adapter: unused, kept for API compatibility.
            fsdp: unused, kept for API compatibility.
        """
        del fsdp  # not used on M2
        self.config.use_mm_proj = True

        if not hasattr(self, "mm_projector"):
            # Linear projection: CLAP 512-dim → Qwen2 hidden_size
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            self.mm_projector.load_state_dict(
                {
                    k.split(".")[-1]: v
                    for k, v in mm_projector_weights.items()
                    if "mm_projector" in k
                }
            )

        return dict(audio_config=AudioEncoderConfig())

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_encodings: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with audio injection.

        audio_encodings: Tensor of shape [batch_size, num_frames, audio_dim]
            or [batch_size, audio_dim] for single-frame (CLAP) encodings.
        """
        # HACK: preserve original embeddings for adapter pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if audio_encodings is not None and self.config.use_mm_proj:
            # Cast audio_encodings to match the weight dtype of self.mm_projector (bfloat16)
            proj_dtype = self.mm_projector.weight.dtype
            if isinstance(audio_encodings, list):
                audio_encodings = [x.to(proj_dtype) if x is not None else None for x in audio_encodings]
            else:
                audio_encodings = audio_encodings.to(proj_dtype)

            # Project audio features into LLM embedding space
            if isinstance(audio_encodings, list):
                audio_features = [
                    self.mm_projector(audio_feature) for audio_feature in audio_encodings
                ]
            else:
                audio_features = self.mm_projector(audio_encodings)

            new_input_embeds = []
            cur_audio_idx = 0

            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if self.audio_encoder_config.use_audio_start_end:
                    cur_audio_features = audio_features[cur_audio_idx]
                    num_frames = cur_audio_features.shape[0]

                    if (cur_input_ids == self.audio_encoder_config.audio_start_token).sum() != (
                        cur_input_ids == self.audio_encoder_config.audio_end_token
                    ).sum():
                        raise ValueError(
                            "The number of audio start tokens and audio end tokens must match."
                        )

                    audio_start_tokens = torch.where(
                        cur_input_ids == self.audio_encoder_config.audio_start_token
                    )[0]

                    if not len(audio_start_tokens) and (past_key_values is None):
                        logging.warning(
                            "No audio start tokens detected and no past_key_values; "
                            "this may indicate a data pipeline issue."
                        )

                    if len(audio_start_tokens):
                        for audio_start_token_pos in audio_start_tokens:
                            cur_audio_features = audio_features[cur_audio_idx].to(
                                device=cur_input_embeds.device
                            )
                            num_frames = cur_audio_features.shape[0]
                            if (
                                cur_input_ids[audio_start_token_pos + num_frames + 1]
                                != self.audio_encoder_config.audio_end_token
                            ):
                                raise ValueError(
                                    "The audio end token must immediately follow the audio frames."
                                )
                            if orig_embeds_params is not None:
                                cur_new_input_embeds = torch.cat(
                                    (
                                        cur_input_embeds[:audio_start_token_pos].detach(),
                                        cur_input_embeds[
                                            audio_start_token_pos : audio_start_token_pos + 1
                                        ],
                                        cur_audio_features,
                                        cur_input_embeds[
                                            audio_start_token_pos
                                            + num_frames
                                            + 1 : audio_start_token_pos
                                            + num_frames
                                            + 2
                                        ],
                                        cur_input_embeds[
                                            audio_start_token_pos + num_frames + 2 :
                                        ].detach(),
                                    ),
                                    dim=0,
                                )
                            else:
                                cur_new_input_embeds = torch.cat(
                                    (
                                        cur_input_embeds[: audio_start_token_pos + 1],
                                        cur_audio_features,
                                        cur_input_embeds[audio_start_token_pos + num_frames + 1 :],
                                    ),
                                    dim=0,
                                )
                            cur_audio_idx += 1
                        new_input_embeds.append(cur_new_input_embeds)
                    else:
                        # No audio start tokens: pure text mode (e.g. during generation
                        # with past_key_values already containing audio context).
                        new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError(
                        "audio_encoder_config.use_audio_start_end=False is not implemented."
                    )

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(WrappedQwen2Model, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class WrappedQwen2ForCausalLM(Qwen2ForCausalLM):
    """Qwen2-based wrapper for causal language modeling with audio input."""

    config_class = WrappedQwen2Config
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = WrappedQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, WrappedQwen2Model):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_encodings=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audio_encodings=audio_encodings,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "audio_encodings": kwargs.get("audio_encodings", None),
            }
        )
        return model_inputs

    def initialize_audio_tokenizer(
        self,
        mm_use_audio_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    ):
        """Set up the tokenizer to handle the audio special tokens."""
        del pretrain_mm_mlp_adapter

        audio_encoder_config = self.get_model().audio_encoder_config
        audio_encoder_config.use_audio_start_end = mm_use_audio_start_end
        tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_audio_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN],
                special_tokens=True,
            )
            self.resize_token_embeddings(len(tokenizer))
            (
                audio_encoder_config.audio_start_token,
                audio_encoder_config.audio_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN]
            )

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                # Initialize new token embeddings to the mean of existing ones
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        audio_encoder_config.audio_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_AUDIO_PATCH_TOKEN]
        )[0]


# Register with HuggingFace Auto classes so the model can be loaded with
# AutoModelForCausalLM.from_pretrained() once saved.
AutoConfig.register("wrapped_qwen2", WrappedQwen2Config)
AutoModelForCausalLM.register(WrappedQwen2Config, WrappedQwen2ForCausalLM)
