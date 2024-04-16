from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.clipseg.modeling_clipseg import (
    BaseModelOutputWithPooling,
    CLIPSegImageSegmentationOutput,
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

from src.models.components.hf_clipseg_wrapper import HFCLIPSegWrapper

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from .context_learner.maple_context_learner import MapleContextLearner


class MapleCLIPSeg(HFCLIPSegWrapper):
    def __init__(
        self,
        model_cfg: Mapping[str, Any],
        context_learner: type[MapleContextLearner],
        freeze_all: bool = True,
    ) -> None:
        super().__init__(**model_cfg)

        if freeze_all:
            self.eval()
            self.requires_grad_(False)

        self.context_learner = context_learner(
            visual_dim=self.model.config.vision_config.hidden_size,
            max_network_depth=min(
                self.model.config.text_config.num_hidden_layers,
                self.model.config.vision_config.num_hidden_layers,
            ),
            context_dim=self.model.config.text_config.hidden_size,
            embedding_layer=self.model.clip.text_model.embeddings.token_embedding,
        )

    def embeddings_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        _self = self.model.clip.text_model.embeddings

        if input_ids is None:
            if inputs_embeds is None:
                msg = "You have to specify either input_ids or inputs_embeds"
                raise ValueError(msg)

            seq_length = inputs_embeds.shape[-2]
        else:
            seq_length = input_ids.shape[-1]

        if inputs_embeds is None:
            inputs_embeds = _self.token_embedding(input_ids)

        inputs_embeds = self.context_learner(
            input_embeddings=inputs_embeds,
            max_length=self.model.config.text_config.max_position_embeddings,
            context_vectors=self.context_vectors[0].expand(
                inputs_embeds.size(0),  # type:ignore
                -1,
                -1,
            ),
        )

        seq_length += self.context_learner.num_context

        if position_ids is None:
            position_ids = _self.position_ids[:, :seq_length]

        position_embeddings = _self.position_embedding(position_ids)
        return inputs_embeds + position_embeddings

    def text_transformer_model_forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:
        r"""Args:
        ----
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        _self = self.model.clip.text_model.encoder

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else _self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else _self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        encoder_states = []
        all_attentions = []

        hidden_states = inputs_embeds
        # Start index from 1 since we have already embedded one in the embeddings_forward
        for idx, encoder_layer in enumerate(_self.layers, 1):
            if output_hidden_states:
                encoder_states.append(hidden_states)

            layer_outputs: tuple[torch.FloatTensor, ...] = (
                _self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
                if _self.gradient_checkpointing and _self.training
                else encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )
            )

            hidden_states = layer_outputs[0]

            if idx < self.context_learner.prompt_depth:
                # Overwrite the prompts skipping the BOS Token till the prompt depth
                hidden_states[:, -self.context_learner.num_context :] = (
                    self.context_learner.context_vectors[
                        idx
                    ].expand(hidden_states.size(0), -1, -1)
                )

            if output_attentions:
                all_attentions.append(layer_outputs[1])

        encoder_states = (
            (*encoder_states, hidden_states) if output_hidden_states else None
        )

        all_attentions = tuple(all_attentions) if output_attentions else None

        if not return_dict:
            return tuple(
                v
                for v in (hidden_states, encoder_states, all_attentions)
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

    def text_model_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        _self = self.model.clip.text_model

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else _self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else _self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        if input_ids is None:
            msg = "You have to specify input_ids"
            raise ValueError(msg)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])  # type:ignore

        hidden_states = self.embeddings_forward(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        # CLIPSeg's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIPSeg/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clipseg/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            hidden_states.shape[:-1],
            hidden_states.dtype,
            device=hidden_states.device,
        )

        # expand attention_mask
        if attention_mask is not None:
            attention_mask = self.context_learner.update_attention_mask_for_context(
                attention_mask,
                self.model.config.text_config.max_position_embeddings,
            )

            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        encoder_outputs = self.text_transformer_model_forward(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = _self.final_layer_norm(last_hidden_state)

        first_pooled_indices = torch.arange(
            last_hidden_state.shape[0],
            device=last_hidden_state.device,
        )

        adjusted_input_ids = input_ids.to(  # type:ignore
            dtype=torch.int,
            device=last_hidden_state.device,
        )

        if _self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIPSeg model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pre_argmax_pooled_indices = adjusted_input_ids
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            pre_argmax_pooled_indices = (adjusted_input_ids == _self.eos_token_id).int()

        # Add context number to the argmax indices but limit to the positional embedding size
        second_indices = torch.minimum(
            pre_argmax_pooled_indices.argmax(dim=-1) + self.context_learner.num_context,
            torch.tensor(self.model.config.text_config.max_position_embeddings - 1),
        )
        pooled_output = last_hidden_state[first_pooled_indices, second_indices]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,  # type:ignore
            attentions=encoder_outputs.attentions,  # type:ignore
        )

    def get_text_features(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, ...]]:
        _self = self.model.clip

        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else _self.config.output_attentions
        )
        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        # (last_hidden_state, pooled_output) + encoder_outputs[1:]
        last_hidden_state, pooled_output, *previous_text_outputs = (
            self.text_model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        )

        text_projection = _self.text_projection(pooled_output)

        return text_projection, tuple(last_hidden_state, *previous_text_outputs)

    def vision_transformer_model_forward(
        self,
        inputs_embeds: torch.FloatTensor,
        visual_context_vectors: Sequence[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:
        r"""Args:
        ----
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = []
        all_attentions = []

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers, 1):
            if output_hidden_states:
                encoder_states.append(hidden_states)

            layer_outputs = (
                self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
                if self.gradient_checkpointing and self.training
                else encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )
            )

            hidden_states = layer_outputs[0]

            if idx < self.context_learner.prompt_depth:
                # Overwrite the prompts skipping the BOS Token till the prompt depth
                hidden_states[:, -self.context_learner.num_context :] = (
                    visual_context_vectors[idx].expand(hidden_states.size(0), -1, -1)
                )

            if output_attentions:
                all_attentions.append(layer_outputs[1])

        encoder_states = (
            (*encoder_states, hidden_states) if output_hidden_states else None
        )

        all_attentions = tuple(all_attentions) if output_attentions else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

    def vision_model_forward(
        self,
        pixel_values: torch.FloatTensor,
        text_intermediate_states: Iterable[torch.Tensor],
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        _self = self.model.clip.vision_model

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else _self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else _self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        # shape: (batch_size, sequence_length, hidden_size)
        hidden_states = _self.embeddings(pixel_values)

        # Tuple of tensors of shape (batch_size, num_context, hidden_size)
        visual_context_vectors = self.context_learner.generate_visual_contexts(
            text_intermediate_states,
        )

        # Add te context to the hidden states at the last position
        first_context_vector = visual_context_vectors[0]

        # Concat the context vector with the hidden states
        hidden_states = torch.cat((hidden_states, first_context_vector), dim=1)

        hidden_states = _self.pre_layrnorm(hidden_states)

        encoder_outputs = self.vision_transformer_model_forward(
            inputs_embeds=hidden_states,
            visual_context_vectors=visual_context_vectors,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = _self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,  # type:ignore
            attentions=encoder_outputs.attentions,  # type:ignore
        )

    def get_vision_outputs(
        self,
        pixel_values: torch.FloatTensor,
        text_intermediate_states: Iterable[torch.Tensor],
        output_attentions: bool | None,
        output_hidden_states: bool | None,
    ) -> tuple[
        list[torch.Tensor],
        torch.FloatTensor,
        BaseModelOutputWithPooling,
    ]:
        _self = self.model

        vision_outputs: BaseModelOutputWithPooling = self.vision_model_forward(
            pixel_values=pixel_values,
            text_intermediate_states=text_intermediate_states,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )  # type:ignore
        pooled_output = _self.clip.visual_projection(vision_outputs[1])

        hidden_states = vision_outputs.hidden_states

        if hidden_states is None:
            msg = "You have to specify output_hidden_states=True if you want to use `CLIPSegForImageSegmentation`"
            raise ValueError(msg)

        # we add +1 here as the hidden states also include the initial embeddings
        activations = [hidden_states[i + 1] for i in _self.extract_layers]

        vision_outputs = BaseModelOutputWithPooling(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=vision_outputs.pooler_output,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=vision_outputs.attentions,
        )
        return activations, pooled_output, vision_outputs

    def get_conditional_embeddings(
        self,
        input_ids: torch.LongTensor,
        num_query_images: int,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
    ):
        # compute conditional embeddings from texts
        if len(input_ids) != num_query_images:
            msg = "Make sure to pass as many prompt texts as there are query images"
            raise ValueError(msg)

        return self.get_text_features(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def model_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CLIPSegImageSegmentationOutput:
        _self = self.model

        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        if pixel_values is None:
            msg = (
                "You have to specify pixel_values to use `CLIPSegForImageSegmentation`"
            )
            raise ValueError(msg)

        # step 1: compute conditional embeddings, either from text, images or an own provided embedding
        projected_conditional_embedding, text_intermediate_states = (
            self.get_conditional_embeddings(
                input_ids=input_ids,  # type:ignore
                num_query_images=len(pixel_values),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        )

        # step 2: forward the query images through the frozen CLIP vision encoder
        activations, pooled_output, vision_outputs = self.get_vision_outputs(
            pixel_values=pixel_values,
            text_intermediate_states=text_intermediate_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = _self.decoder(
            activations,
            projected_conditional_embedding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]

        loss: torch.FloatTensor | None = (  # type: ignore
            None
            if labels is None
            else F.binary_cross_entropy_with_logits(logits, labels)
        )

        return CLIPSegImageSegmentationOutput(
            loss=loss,
            logits=logits,
            conditional_embeddings=projected_conditional_embedding,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,  # type:ignore
            decoder_output=decoder_outputs,
        )

    def forward(
        self,
        text_input: Mapping[str, torch.Tensor],
        image_input: torch.Tensor,
    ):
        B, _, H, W = image_input.shape

        # The output is squeezed so it may be (H, W) or (B, H, W)
        outputs = self.model_forward(**text_input, pixel_values=image_input)  # type:ignore

        return outputs.logits.view(B, 1, H, W)
