from typing import List
import os
import json
import copy
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

import datasets
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModel, AutoImageProcessor
from infermodels_local import *


class CLIPDINOEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)


    def get_score(self, image_list: List[Image.Image], text_list: List[str]):
        """
        get CLIP/DINO-based metrics values

        Args:
            image_list (`List[Image.Image]`): list of images - input, output, ground-truth
            text_list (`List[str]`): list of captions - input, target

        Returns:
            `dict`: metric values
        """

        # get CLIP features
        inputs = self.clip_processor(text=text_list, images=image_list, return_tensors="pt", padding=True).to(self.device)
        image_features = self.clip_model.get_image_features(inputs.pixel_values)
        text_features = self.clip_model.get_text_features(inputs.input_ids, inputs.attention_mask)

        input_image_feature = image_features[:1]
        output_image_feature = image_features[1:2]
        gt_image_feature = image_features[2:]
        input_text_feature = text_features[:1]
        target_text_feature = text_features[1:]
        ret = {}

        # clip similarity with gt
        ret["clip_sim"] = {}
        for name, img_feat in [
            ("input", input_image_feature),
            ("output", output_image_feature),
        ]:
            ret["clip_sim"][name] = F.cosine_similarity(img_feat, gt_image_feature)[0].item()

        # clip score with target prompt
        ret["clip_text"] = {}
        for name, img_feat in [
            ("input", input_image_feature),
            ("output", output_image_feature),
            ("gt", gt_image_feature),
        ]:
            ret["clip_text"][name] = F.cosine_similarity(img_feat, target_text_feature)[0].item()

        # clip directional similarity
        if (input_text_feature - target_text_feature).abs().max() > 0:
            ret["clip_dir"] = {}
            for name, img_feat in [
                ("output", output_image_feature),
                ("gt", gt_image_feature),
            ]:
                ret["clip_dir"][name] = \
                F.cosine_similarity(
                    target_text_feature - input_text_feature,
                    img_feat - input_image_feature,
                )[0].item()

        # get DINO features
        inputs = self.dino_processor(images=image_list, return_tensors="pt").to(self.device)
        image_features = self.dino_model(**inputs).pooler_output
        input_image_feature = image_features[:1]
        output_image_feature = image_features[1:2]
        gt_image_feature = image_features[2:]

        # dino similarity with gt
        ret["dino_sim"] = {}
        for name, img_feat in [
            ("input", input_image_feature),
            ("output", output_image_feature),
        ]:
            ret["dino_sim"][name] = F.cosine_similarity(img_feat, gt_image_feature)[0].item()
        return ret


class PromptPairEstimator:
    def __init__(self):
        # load fine-tuned captioning model
        # TODO: update checkpoint path
        model_id = "./ckpts/qwen2_vl_lora_sft"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def get_captions(self, source_image: Image.Image, instruction: str):
        """
        Function to get source/target image captions for prompt-guided editing models

        Args:
            source_image (`Image.Image`): source image to be edited
            instruction: (`str`): edit instruction

        Returns:
            `dict`: dictionary with source and target captions
        """

        # template to generate source caption
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Describe this image based on the edit \"{instruction}\" we want to perform on it later"},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = source_image.convert("RGB").resize((1024, 1024))
        inputs = self.processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to("cuda")
        output_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        source_caption = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]

        # template to generate target caption
        conversation += [
            {
                "role": "assistant",
                "content": source_caption,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the edited image with respect to this image and the edit instruction."},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to("cuda")
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        target_caption = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        return {
            "source_caption": source_caption,
            "target_caption": target_caption
        }


if __name__ == "__main__":
    """
    Generate samples from text-guided image editing models
    with respect to MagicBrush images
    """
    # change split to dev or emu to generate other samples
    split = "train"
    ds = datasets.load_dataset("osunlp/MagicBrush")[split]
    
    base_resolution = 512
    batch_size = 4
    
    out_dir = "./results/tie_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # list of editing models to generate samples with
    model_dict = {
        "cyclediff": CycleDiffusion(), # no batch process
        "pr2pr": Prompt2prompt(),   # no batch process
        "t2l": Text2Live(), # no batch process
        "diffedit": DiffEdit(),
        "ip2p": InstructPix2Pix(),
        "magicbrush": MagicBrush(),
        "aurora": AURORA(),
        "p2p0": Pix2PixZero(),
        "sdedit": SDEdit(),
    }

    # captioning model to get source/target image captions
    caption_model = PromptPairEstimator()
    # evaluator to get clip and dino metrics
    evaluator = CLIPDINOEvaluator()

    for batch_index in tqdm(range(0, len(ds), batch_size)):
        # get the list of samples for the current batch
        sample_list = [ds[i] for i in range(batch_index, min(batch_index+batch_size, len(ds)))]
        out_path_list = [
            f"{sample['img_id']}_{sample['turn_index']}" 
            for sample in sample_list
        ]
        source_list = [
            sample["source_img"].convert("RGB") for sample in sample_list
        ]
        target_list = [
            sample["target_img"].convert("RGB") for sample in sample_list
        ]
        instruction_list = [
            sample["instruction"] for sample in sample_list
        ]

        # generate caption pairs for prompt-guided editing models
        if "source_caption" not in sample_list[0] or "target_caption" not in sample_list[0]:
            prompt_dict_list = [
                caption_model.get_captions(source_image, instruction) for source_image, instruction in
                zip(source_list, instruction_list)
            ]
            source_prompt_list = [
                d["source_caption"] for d in prompt_dict_list
            ]
            target_prompt_list = [
                d["target_caption"] for d in prompt_dict_list
            ]

        # check if the samples have been generated before, if so, skip the batch
        skip = True
        for model in model_dict:
            for filename in out_path_list:
                out_path = os.path.join(out_dir, model, filename + f"_{model}.jpg")
                if not os.path.exists(out_path):
                    skip = False
                    break
            if not skip:
                break
        if skip:
            continue

        for model_name, model in model_dict.items():
            # run batch inference for each editing model
            # for editing model with no batch process, each output is generated separately
            output_list = model.infer_batch(
                src_image=[x.resize((base_resolution, base_resolution)) for x in copy.deepcopy(source_list)], 
                src_prompt=source_prompt_list, 
                target_prompt=target_prompt_list, 
                instruct_prompt=instruction_list,
                seed=42
            )
            for out_path, output_image, source_image, target_image, instruction, source_caption, target_caption in zip(out_path_list, output_list, source_list, target_list, instruction_list, source_prompt_list, target_prompt_list):

                # resize image to match source image size
                output_image = output_image.resize(source_image.size)

                # compute clip and dino metrics
                evaluator_image_list = [source_image, output_image, target_image]
                evaluator_text_list = [source_caption, target_caption]
                metric_dict = evaluator.get_score(image_list=evaluator_image_list, text_list=evaluator_text_list)
                metric_dict["source_caption"] = source_caption
                metric_dict["target_caption"] = target_caption
                metric_dict["instruction"] = instruction

                # save image and metrics
                full_out_path = os.path.join(out_dir, out_path + f"_{model_name}")
                output_image.save(full_out_path + ".jpg")
                with open(full_out_path + ".json", "w") as out:
                    json.dump(metric_dict, out, indent=4)
