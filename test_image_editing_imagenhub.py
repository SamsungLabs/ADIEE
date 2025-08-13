import argparse
import os
from tqdm import tqdm
import json
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import AutoProcessor
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from adiee import ADIEEScorer
from dataset import SCORE_TOKEN
from utils import *

if __name__ == "__main__":
    # evaluate image editing model on ImagenHub
    parser = argparse.ArgumentParser(description="Image editing testing on ImagenHub")
    parser.add_argument(
        "--image_editing_model_id", 
        required=True, 
        type=str,
        help="path to image editing model checkpoint"
    )
    parser.add_argument(
        "--add_reward_prompt", 
        action="store_true",
        help="whether to add prompt with score to the edit instruction"
    )
    parser.add_argument(
        "--scorer_ckpt_path", 
        required=True, 
        type=str,  
        help="path to ADIEE evaluation model checkpoint",
    )
    parser.add_argument(
        "--decoder_type", 
        default="classifier", 
        choices=["score_prediction", "classifier"], 
        type=str,
        help="whether the ADIEE score decoder is a scorer or classifier"
    )
    parser.add_argument(
        "--dataset_dir", 
        default="adiee_datasets", 
        type=str,
        help="path to testing data"
    )
    parser.add_argument(
        "--output_dir", 
        default="results", 
        type=str,
        help="where to store outputs",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load evaluation model and processor
    weight_dtype = torch.bfloat16
    device = "cuda"
    scorer_processor = AutoProcessor.from_pretrained(args.scorer_ckpt_path)

    # check if score token is added to model and processor correctedly
    assert SCORE_TOKEN in scorer_processor.tokenizer.get_vocab()
    score_token_idx = scorer_processor.tokenizer(SCORE_TOKEN, add_special_tokens=False).input_ids[0]
    scorer = ADIEEScorer.from_pretrained(
        args.scorer_ckpt_path,
        score_token_idx=score_token_idx,
        out_channels=1 if args.decoder_type=="score_prediction" else 3,
        torch_dtype=weight_dtype,
        device_map=device,
    )
    assert scorer.score_token_idx == score_token_idx, f"{scorer.score_token_idx} != {score_token_idx}"

    # suppress pad_token_id set at runtime warning
    scorer.vlm.generation_config.pad_token_id = scorer.vlm.generation_config.eos_token_id

    # load image editing model
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.image_editing_model_id, 
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # test on imagenhub 
    imagehub_root = f"{args.dataset_dir}/ImagenHub/results/ImagenHub_Text-Guided_IE"
    instruction_dict = json.load(open(os.path.join(imagehub_root, "dataset_lookup.json")))
    output_list = []

    for basename, prompt_dict in tqdm(instruction_dict.items(), total=len(instruction_dict)):
        input_path = os.path.join(imagehub_root, "input", basename)
        magicbrush_path = os.path.join(imagehub_root, "MagicBrush", basename)
        gt_path = os.path.join(imagehub_root, "GroundTruth", basename)
        input_image = Image.open(input_path)
        magicbrush_output = Image.open(magicbrush_path)
        gt_image = Image.open(gt_path)
        instruction = prompt_dict["instruction"]

        header = ["input", "magicbrush", "ours", "gt"]
        resolution = 512
        font_size = resolution // 20
        space = resolution // 50

        # blank image
        full_width = (resolution + space) * len(header) - space # width = resolution * num_col + space in between
        full_height = font_size + space + resolution + font_size + space

        visual_out_path = os.path.join(args.output_dir, basename)
        output = pipe(
            instruction + ". The image quality is five out of five" if args.add_reward_prompt else instruction, 
            image=input_image, 
            num_inference_steps=100,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            generator=torch.manual_seed(42)
        ).images[0]
        
        image_list = [input_image, magicbrush_output, output, gt_image]
        full_image = Image.new("RGB", (full_width, full_height), "white")
        draw = ImageDraw.Draw(full_image)
        font = ImageFont.truetype("assets/arial.ttf", font_size)
    
        # add header
        for k in range(len(header)): # k = column index
            draw.text(
                (k * (resolution + space), 0), 
                header[k], 
                fill="black", 
                font=font
            )
            full_image.paste(
                image_list[k], 
                ((resolution + space) * k, font_size + space)
            )
        draw.text(
            (0, font_size + space + resolution + space), 
            instruction, 
            fill="black", 
            font=font
        )
        full_image.save(visual_out_path)
        score = prepare_data_and_evaluate(
            scorer=scorer,
            scorer_processor=scorer_processor,
            input_image=input_image,
            output_image=output,
            instruction=instruction,
            weight_dtype=torch.bfloat16,
            device=device,
        )
        output_list.append({
            "filename": basename,
            "instruction": instruction,
            "score": score,
        })

    with open(os.path.join(args.output_dir, "imagenhub.json"), "w") as out:
        mean_score = np.array([x["score"] for x in output_list]).mean()
        print("Avg evaluation score", mean_score)
        json.dump({"metric": mean_score, "outputs": output_list}, out, indent=4)

