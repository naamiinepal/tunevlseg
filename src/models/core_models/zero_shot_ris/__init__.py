import torch
from torch import nn
from torchvision.transforms import functional as TF

from .clip import CustomCLIP
from .freesolo import CustomFreeSOLO


class ZeroShotRIS(nn.Module):
    alpha = 0.95
    beta = 0.5

    def __init__(self, clip_pretrained_path, solo_config, solo_state_dict_path) -> None:
        super().__init__()

        self.clip = CustomCLIP(clip_pretrained_path)
        self.freesolo = CustomFreeSOLO(solo_config, solo_state_dict_path)
        self.vision_config = self.clip.config.vision_config

    def get_cropped_features(self, image_input: torch.Tensor, pred_boxes, pred_masks):
        clip_image_size = self.vision_config.image_size

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)

        cropped_imgs = []

        for pred_box, pred_mask in zip(pred_boxes, pred_masks, strict=True):
            # pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
            masked_image = image_input * pred_mask + ~pred_mask * pixel_mean

            x1, y1, x2, y2 = (
                int(pred_box[0]),
                int(pred_box[1]),
                int(pred_box[2]),
                int(pred_box[3]),
            )

            masked_image = TF.resized_crop(
                masked_image,
                y1,
                x1,
                (y2 - y1),
                (x2 - x1),
                [clip_image_size, clip_image_size],
                antialias=False,
            )

            cropped_imgs.append(masked_image)

        cropped_tensor = torch.stack(cropped_imgs)

        return self.clip.get_image_features(cropped_tensor)

    def get_text_ensemble(self, text_input: dict[str, torch.Tensor]):
        batched_text_features = self.clip.model.get_text_features(
            text_input["input_ids"],
            text_input["attention_mask"],
        )

        phrase_features, class_features = batched_text_features

        return self.beta * phrase_features + (1 - self.beta) * class_features

    def get_max_index(self, text_ensemble: torch.Tensor, visual_feature: torch.Tensor):
        # normalized features
        image_embeds = visual_feature / visual_feature.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_ensemble / text_ensemble.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())

        return logits_per_image.argmax()

    def get_mask_features(self, image_input: torch.Tensor, pred_masks):
        clip_image_size = self.vision_config.image_size

        resized_image = TF.resize(
            image_input,
            [clip_image_size, clip_image_size],
            antialias=False,
        )

        patch_size = self.vision_config.patch_size
        mask_size = clip_image_size // patch_size

        # Add channel dimension to the pred_masks
        resized_masks = torch.stack(
            [
                TF.resize(mask, [mask_size, mask_size], antialias=False)
                for mask in pred_masks.unsqueeze(1)
            ],
        ).squeeze(1)

        return self.clip.get_image_features(
            pixel_values=resized_image.unsqueeze(0),
            pred_masks=resized_masks,
            masking_block_idx=-3,
        )

    def get_visual_feature(self, image_input: torch.Tensor, pred_boxes, pred_masks):
        print("Prediction Mask Shape:", pred_masks.shape)

        mask_features = self.get_mask_features(image_input, pred_masks)

        crop_features = self.get_cropped_features(image_input, pred_boxes, pred_masks)

        return self.alpha * mask_features + (1 - self.alpha) * crop_features

    def forward(self, image_input: torch.Tensor, text_input: dict[str, torch.Tensor]):
        """Steps to implement:

        1. image = Load an image in the original size (without normalization).
        2. freesolo_resized = Resize for freesolo, if it doesn't resize itself
        3. mask_resize = resize the to the size of sqrt(num_patches)
        4. Crop and resize the images using masks.
        5. clip_resize = Resize and normalize for original image for clip.
        6. Get classname and description to pass to the clip_text encoder.
        """
        if image_input.ndim == 4:
            if image_input.size(0) != 1:
                print("The image input should be of a single batch size.")
            image_input = image_input[0]

        pred_boxes, pred_masks = self.freesolo(image_input)

        if len(pred_masks) == 0:
            print("No pred masks given. No image features are returned.")
            return torch.zeros(
                (1, *image_input.shape[1:]),
                dtype=image_input.dtype,
                device=image_input.device,
            )

        visual_feature = self.get_visual_feature(image_input, pred_boxes, pred_masks)

        text_ensemble = self.get_text_ensemble(text_input)

        max_index = self.get_max_index(text_ensemble, visual_feature)

        return pred_masks[max_index].unsqueeze(0)
