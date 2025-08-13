import argparse
import os
import copy
import json

import torch
from transformers import AutoProcessor

from adiee import ADIEEScorer
from dataset import SCORE_TOKEN
from utils import *

if __name__ == "__main__":
    # evaluate image editing evaluaton model on 4 benchmarks
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

    # datasets to run evaluation on
    eval_list = [
        "imagenhub",
        "aurorabench",
        "genaibench",
        "aurorabenchpairwise",
    ]
    for benchmark_name in eval_list:
        benchmark_func = eval(f"run_{benchmark_name}")
        ret = benchmark_func(
            args,
            model, 
            copy.deepcopy(processor), 
            weight_dtype, 
            device,
        )

        # save results
        ret["visual"].save(os.path.join(args.output_dir, benchmark_name + ".jpg"))
        with open(os.path.join(args.output_dir, benchmark_name + ".json"), "w") as out:
            json.dump({"metric": ret["metric"], "outputs": ret["outputs"]}, out, indent=4)
            print(benchmark_name, ret["metric"])
