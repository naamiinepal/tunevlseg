import torch
from torch.nn import functional as F
from transformers.models.clipseg.modeling_clipseg import (
    BaseModelOutputWithPooling,
    CLIPSegImageSegmentationOutput,
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

from src.models.components.hf_clipseg_wrapper import HFCLIPSegWrapper

from .context_learner import ContextLearnerMixin


class COOPCLIPSeg(ContextLearnerMixin, HFCLIPSegWrapper):
    def embeddings_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
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

        # Intercept here, now add the learnable contexts
        first_embed = inputs_embeds[:, :1, :]  # type:ignore
        rest_embed = inputs_embeds[:, 1:, :]  # type:ignore

        inputs_embeds = torch.cat(
            [first_embed, self.context_vectors, rest_embed],
            dim=1,
        )  # type:ignore

        seq_length += self.num_context

        if position_ids is None:
            position_ids = _self.position_ids[:, :seq_length]

        position_embeddings = _self.position_embedding(position_ids)
        return inputs_embeds + position_embeddings

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
            input_shape,
            hidden_states.dtype,
            device=hidden_states.device,
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        encoder_outputs = _self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = _self.final_layer_norm(last_hidden_state)

        if _self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIPSeg model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0],
                    device=last_hidden_state.device,
                ),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(  # type:ignore
                    dim=-1,
                ),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0],
                    device=last_hidden_state.device,
                ),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device)  # type:ignore
                    == _self.eos_token_id
                )
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_text_features(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.FloatTensor:
        _self = self.model.clip

        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
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

        text_outputs = self.text_model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        return self.text_projection(pooled_output)

    def get_vision_outputs(
        self,
        pixel_values: torch.FloatTensor | None,
        output_attentions: bool | None,
        output_hidden_states: bool | None,
    ) -> tuple[
        list[torch.Tensor],
        torch.FloatTensor,
        BaseModelOutputWithPooling,
    ]:
        _self = self.model

        vision_outputs = _self.clip.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )
        pooled_output = _self.clip.visual_projection(vision_outputs[1])

        hidden_states = vision_outputs.hidden_states
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
        conditional_embeddings: torch.FloatTensor | None,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        conditional_pixel_values: torch.FloatTensor | None,
        position_ids: torch.LongTensor | None,
    ):
        _self = self.model
        if conditional_embeddings is None:
            batch_size = pixel_values.shape[0]
            if input_ids is not None:
                # compute conditional embeddings from texts
                if len(input_ids) != batch_size:
                    msg = "Make sure to pass as many prompt texts as there are query images"
                    raise ValueError(msg)
                return self.get_text_features(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

            if conditional_pixel_values is not None:
                # compute conditional embeddings from images
                if len(conditional_pixel_values) != batch_size:
                    msg = "Make sure to pass as many prompt images as there are query images"
                    raise ValueError(msg)
                return _self.clip.get_image_features(conditional_pixel_values)

            msg = "Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`"
            raise ValueError(msg)

        if conditional_embeddings.shape[0] != pixel_values.shape[0]:
            msg = "Make sure to pass as many conditional embeddings as there are query images in the batch"
            raise ValueError(msg)

        if conditional_embeddings.shape[1] != _self.config.projection_dim:
            msg = "Make sure that the feature dimension of the conditional embeddings matches `config.projection_dim`."
            raise ValueError(msg)

        return conditional_embeddings

    def model_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        conditional_pixel_values: torch.FloatTensor | None = None,
        conditional_embeddings: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CLIPSegImageSegmentationOutput:
        _self = self.model

        if pixel_values is None:
            msg = (
                "You have to specify pixel_values to use `CLIPSegForImageSegmentation`"
            )
            raise ValueError(msg)

        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        # step 1: forward the query images through the frozen CLIP vision encoder
        activations, pooled_output, vision_outputs = self.get_vision_outputs(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # step 2: compute conditional embeddings, either from text, images or an own provided embedding
        conditional_embeddings = self.get_conditional_embeddings(
            conditional_embeddings=conditional_embeddings,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            conditional_pixel_values=conditional_pixel_values,
            position_ids=position_ids,
        )

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = _self.decoder(
            activations,
            conditional_embeddings,
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
            conditional_embeddings=conditional_embeddings,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,  # type:ignore
            decoder_output=decoder_outputs,
        )

    def forward(self, *, pixel_values: torch.FloatTensor, **kwargs):
        B, _, H, W = pixel_values.shape

        # The output is squeezed so it may be (H, W) or (B, H, W)
        outputs = self.model_forward(pixel_values=pixel_values, **kwargs)

        return outputs.logits.view(B, 1, H, W)
