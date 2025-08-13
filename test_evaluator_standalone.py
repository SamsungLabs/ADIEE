import argparse
from transformers import AutoProcessor
import torch
from PIL import Image

from adiee import ADIEEScorer
from dataset import SCORE_TOKEN, prepare_question_answer
from utils import prepare_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADIEE Model testing")
    parser.add_argument(
        "--model_id", 
        required=True, 
        type=str,
        help="path to ADIEE evaluation model checkpoint"
    )
    parser.add_argument(
        "--decoder_type", 
        default="classifier", 
        choices=["score_prediction", "classifier"], 
        type=str,
        help="whether the ADIEE score decoder is a scorer or classifier"
    )
    parser.add_argument(
        "--input_image_path", 
        required=True, 
        type=str,
        help="input image path"
    )
    parser.add_argument(
        "--output_image_path", 
        required=True, 
        type=str,
        help="edited image path"
    )
    parser.add_argument(
        "--instruction", 
        required=True, 
        type=str,
        help="edit instruction"
    )
    args = parser.parse_args()

    # load model and processor
    weight_dtype = torch.bfloat16
    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model_id)

    # check if score token is added to model and processor correctedly
    assert SCORE_TOKEN in processor.tokenizer.get_vocab()
    score_token_idx = processor.tokenizer(SCORE_TOKEN, add_special_tokens=False).input_ids[0]
    model = ADIEEScorer.from_pretrained(
        args.model_id,
        score_token_idx=score_token_idx,
        out_channels=1 if args.decoder_type=="score_prediction" else 3,
        torch_dtype=weight_dtype,
        device_map=device,
    )
    assert model.score_token_idx == score_token_idx, f"{model.score_token_idx} != {score_token_idx}"

    # suppress pad_token_id set at runtime warning
    model.vlm.generation_config.pad_token_id = model.vlm.generation_config.eos_token_id

    # prepare text input, also see prepare_question_answer in datasets.py
    image_list = [args.input_image_path, args.output_image_path]
    image_list = [Image.open(f).convert("RGB") for f in image_list]
    inputs = prepare_question_answer(
        processor=processor,
        split="test",
        input_image=image_list[0],
        output_image=image_list[1],
        instruction=args.instruction,
    )

    # move model to GPU and convert it to the right data type
    inputs = prepare_batch(inputs, weight_dtype, device)

    # run model inference
    with torch.no_grad():
        ret = model.inference(**inputs)

    # post-process outputs, also see run_model in utils.py
    ret_dict = {}
    for k, v in ret.items():
        if k == "class_prob": # for inference, convert evaluation class labels 0, 1, 2 to scores 0-10
            score = (torch.nn.Softmax()(v[0]) * torch.tensor([0, 1, 2]).to(v.dtype).to(v.device)).sum().item()
            score = torch.tensor([max(min(score, 2), 0) * 5])
            ret_dict["score"] = score[0].item()
        elif k == "score":
            ret_dict["score"] = v[0].item()
        elif k == "output_ids": # decode output to text
            output_ids = processor.decode(v[0], skip_special_tokens=True).strip()
            ret_dict["output_ids"] = output_ids
    print(ret_dict)


