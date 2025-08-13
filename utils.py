from typing import List
import argparse
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import spearmanr
import math
import numpy as np
import io
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from transformers import AutoProcessor
from transformers.image_utils import ChannelDimension, get_image_size
from transformers.image_processing_utils import select_best_resolution
from transformers.models.llava_next.image_processing_llava_next import _get_patch_output_size
from diffusers.utils.torch_utils import is_compiled_module

from dataset import (
    ImagenHubDataset, 
    GenAIBenchDataset, 
    AURORABenchDataset, 
    AURORABenchPairwiseDataset,
    prepare_question_answer
)
from adiee import ADIEEScorer


def unwrap_model(model, accelerator):
    """
    unwrap model with accelerator
    """
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def prepare_batch(batch: dict, weight_dtype: str, device: str):
    """
    Function to move data to GPU with proper data type

    Args:
        batch (`dict`): 
            dictionary with a batch of training samples
        weight_dtype (`str`):
             data type to convert the samples to
        device (`str`): 
            GPU IDs to load the samples onto

    Returns:
        `dict`: a dictionary of processed samples
    """
    for k, v in batch.items():
        if "pixel_values" in k:
            batch[k] = v.to(weight_dtype)
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
    return batch


def run_model(model: ADIEEScorer, batch: dict, is_train: bool):
    """
    Perform one model forward function call

    Args:
        model (`ADIEEScorer`): 
            training model
        batch (`dict`): 
            dictionary with a batch of training samples
        is_train (`bool`): 
            True for training stage, False for validation stage

    Returns:
        `dict`: results from model
    """
    ret_dict = defaultdict(list)

    # one model run per image inputs in the batch
    # for point-wise experiment, the count is 1; for pair-wise experiment, the count is 2
    count = len([k for k in batch if "pixel_values_" in k])
    for i in range(count):
        kwargs = {
            "input_ids": batch[f"input_ids_{i+1}"],
            "attention_mask": batch[f"attention_mask_{i+1}"],
            "pixel_values": batch[f"pixel_values_{i+1}"],
            "image_sizes": batch[f"image_sizes_{i+1}"],
            "labels": batch[f"input_ids_{i+1}"] if is_train else None,
        }
        ret = model(**kwargs)
        for k, v in ret.items():
            if k == "class_prob": # for inference, convert evaluation class labels 0, 1, 2 to scores 0-10
                score = (torch.nn.Softmax(dim=1)(v) * torch.tensor([0, 1, 2]).to(v.dtype).to(v.device)).sum(dim=1)
                score = score.clamp(0, 2) * 5
                ret_dict["score"].append(score)
            else:
                ret_dict[k].append(v)
    return ret_dict


def run_imagenhub(
    args: argparse.Namespace, 
    model: ADIEEScorer, 
    processor: AutoProcessor, 
    weight_dtype: str, 
    device: str,
):
    """
    Evaluation function for ImagenHub Text-guided Image Editing 
    """
    test_dataset = ImagenHubDataset(args, processor)

    score_dict = defaultdict(list)
    for batch in tqdm(test_dataset, desc="eval on imagenhub"):
        batch = prepare_batch(batch, weight_dtype=weight_dtype, device=device)
        with torch.no_grad():
            ret_dict = run_model(
                model=model, 
                batch=batch, 
                is_train=False
            )
        
        # get scorer input and output
        input_ids = processor.decode(batch["input_ids_1"][0], skip_special_tokens=True)
        input_ids = inputids_cleanup(input_ids)
        output_ids = processor.decode(ret_dict["output_ids"][0][0], skip_special_tokens=True).strip()

        score_dict[batch["model"]].append({
            "filename": batch["filename"],
            "model": batch["model"],
            "input_ids": input_ids,
            "output_ids": output_ids,
            "gt": batch["score"],
            "pred": ret_dict["score"][0].item(),
        })

    z_score_list = []
    visual_list = []
    for model, gt_pred_list in score_dict.items():
        # compute score correlation with human rating per editing model
        gt_list = [x["gt"] for x in gt_pred_list]
        pred_list = [x["pred"] for x in gt_pred_list]
        r, _ = spearmanr(pred_list, gt_list)
        z_score_list.append(r)

        # visualize the scatter plot
        visual = plot_scatter_plot(
            y_true=gt_list,
            y_pred=pred_list,
            title=model + f" ({round(r, 3)})"
        )
        visual_list.append(visual)
    
    # Averages a list of Fisher Z-transformed correlation scores and converts it back to a correlation coefficient
    z_avg = sum(z_score_list) / len(z_score_list)
    r_avg = (math.exp(2 * z_avg) - 1) / (math.exp(2 * z_avg) + 1)

    # concate all visualization together
    rows = 2
    cols = math.ceil(len(visual_list) / rows)
    img_width, img_height = visual_list[0].size
    combined_width = cols * img_width
    combined_height = rows * img_height
    combined_image = Image.new("RGB", (combined_width, combined_height))
    for idx, img in enumerate(visual_list):
        row, col = divmod(idx, cols)
        x_offset = col * img_width
        y_offset = row * img_height
        combined_image.paste(img, (x_offset, y_offset))
    # print("\nImagenHub r", r_avg)

    output_list = []
    for model, model_output_list in score_dict.items():
        output_list += model_output_list
    return {
        "metric": r_avg,
        "visual": combined_image,
        "outputs": output_list,
    }


def run_genaibench(
    args: argparse.Namespace, 
    model: ADIEEScorer, 
    processor: AutoProcessor, 
    weight_dtype: str, 
    device: str,
):
    """
    Evaluation function for GenAI-Bench Text-guided Image Editing 
    """
    test_dataset = GenAIBenchDataset(args, processor)
    output_list = []
    for batch in tqdm(test_dataset, desc="eval on genaibench"):
        batch = prepare_batch(batch, weight_dtype=weight_dtype, device=device)
        with torch.no_grad():
            ret_dict = run_model(
                model=model, 
                batch=batch, 
                is_train=False
            )
        score_1 = ret_dict["score"][0].item()
        score_2 = ret_dict["score"][1].item()

        # normalize score to type 0, 1, 2 for failed, partial successful, and successful edit
        type_1 = round(score_1 / 5)
        type_2 = round(score_2 / 5)

        # one image is preferred when their type are different
        # if both are successful edit, they are equally good
        # otherwize they are equally bad
        if type_1 > type_2:
            pred_label = 0
        elif type_1 < type_2:
            pred_label = 1
        elif score_1 >= 9 and score_2 >= 9:
            pred_label = 2
        else:
            pred_label = 3

        # get scorer input and output for each image
        input_ids_1 = processor.decode(batch["input_ids_1"][0], skip_special_tokens=True).replace("assistant", "").strip()
        input_ids_2 = processor.decode(batch["input_ids_2"][0], skip_special_tokens=True).replace("assistant", "").strip()
        input_ids_1 = inputids_cleanup(input_ids_1)
        input_ids_2 = inputids_cleanup(input_ids_2)
        output_ids_1 = processor.decode(ret_dict["output_ids"][0][0], skip_special_tokens=True).strip()
        output_ids_2 = processor.decode(ret_dict["output_ids"][1][0], skip_special_tokens=True).strip()

        output_list.append({
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            "output_ids_1": output_ids_1,
            "output_ids_2": output_ids_2,
            "score_1": score_1,
            "score_2": score_2,
            "gt": batch["label"],
            "pred": pred_label,
        })

    # get accuracy and confusion matrix
    pred_list = [output["pred"] for output in output_list]
    gt_list = [output["gt"] for output in output_list]
    acc = sum([1 for p, g in zip(pred_list, gt_list) if p == g]) / len(test_dataset)
    visual = plot_confusion_matrix(
        y_true=gt_list, 
        y_pred=pred_list, 
        labels=test_dataset.labels,
        title="GenAI-Bench",
    )
    # print("\nGenAI-Bench acc", acc)
    return {
        "metric": acc,
        "visual": visual,
        "outputs": output_list,
    }


def run_aurorabench(
    args: argparse.Namespace, 
    model: ADIEEScorer, 
    processor: AutoProcessor, 
    weight_dtype: str, 
    device: str,
):
    """
    Evaluation function for AURORA-Bench point-wise samples 
    """
    test_dataset = AURORABenchDataset(args, processor)
    output_list = []
    for batch in tqdm(test_dataset, desc="eval on aurorabench"):
        batch = prepare_batch(batch, weight_dtype=weight_dtype, device=device)
        with torch.no_grad():
            ret_dict = run_model(
                model=model, 
                batch=batch, 
                is_train=False
            )

        # get scorer input and output
        input_ids = processor.decode(batch["input_ids_1"][0], skip_special_tokens=True)
        input_ids = inputids_cleanup(input_ids)
        output_ids = processor.decode(ret_dict["output_ids"][0][0], skip_special_tokens=True).strip()
            
        output_list.append({
            "input_path": batch["input_path"],
            "output_path": batch["output_path"],
            "input_ids": input_ids,
            "output_ids": output_ids,
            "pred": ret_dict["score"][0].item(),
            "gt": batch["score"]
        })

    # get correlation and confusion matrix
    pred_list = [output["pred"] for output in output_list]
    gt_list = [output["gt"] for output in output_list]
    r, _ = spearmanr(pred_list, gt_list)
    visual = plot_scatter_plot(
        y_true=gt_list, 
        y_pred=pred_list, 
        title="AURORA-Bench point-wise"
    )
    # print("\nAURORA-Bench point-wise r", r)
    return {
        "metric": r,
        "visual": visual,
        "outputs": output_list,
    }


def run_aurorabenchpairwise(
    args: argparse.Namespace, 
    model: ADIEEScorer, 
    processor: AutoProcessor, 
    weight_dtype: str, 
    device: str,
):
    """
    Evaluation function for AURORA-Bench pair-wise samples 
    """
    test_dataset = AURORABenchPairwiseDataset(args, processor)
    output_list = []

    for batch in tqdm(test_dataset, desc="eval on aurorabenchpairwise"):
        batch = prepare_batch(batch, weight_dtype=weight_dtype, device=device)
        with torch.no_grad():
            ret_dict = run_model(
                model=model, 
                batch=batch, 
                is_train=False
            )
        score_1 = ret_dict["score"][0].item()
        score_2 = ret_dict["score"][1].item()

        # normalize score to type 0, 1, 2 for failed, partial successful, and successful edit
        type_1 = round(score_1 / 5)
        type_2 = round(score_2 / 5)

        # one image is preferred when their type are different
        # else they are tied
        if type_1 > type_2:
            pred_label = 0
        elif type_1 < type_2:
            pred_label = 1
        else:
            pred_label = 2
    
        # get scorer input and output for each image
        input_ids_1 = processor.decode(batch["input_ids_1"][0], skip_special_tokens=True)
        input_ids_2 = processor.decode(batch["input_ids_2"][0], skip_special_tokens=True)
        input_ids_1 = inputids_cleanup(input_ids_1)
        input_ids_2 = inputids_cleanup(input_ids_2)
        output_ids_1 = processor.decode(ret_dict["output_ids"][0][0], skip_special_tokens=True).strip()
        output_ids_2 = processor.decode(ret_dict["output_ids"][1][0], skip_special_tokens=True).strip()

        output_list.append({
            "input_path": batch["input_path"],
            "output_1_path": batch["output_1_path"],
            "output_2_path": batch["output_2_path"],
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            "output_ids_1": output_ids_1,
            "output_ids_2": output_ids_2,
            "score_1": score_1,
            "score_2": score_2,
            "gt": batch["label"],
            "pred": pred_label,
        })

    # get accuracy and confusion matrix
    pred_list = [output["pred"] for output in output_list]
    gt_list = [output["gt"] for output in output_list]
    acc = sum([1 for p, g in zip(pred_list, gt_list) if p == g]) / len(test_dataset)
    visual = plot_confusion_matrix(
        y_true=gt_list, 
        y_pred=pred_list, 
        labels=test_dataset.labels,
        title="AURORA-Bench pair-wise"
    )
    # print("\nAURORA-Bench pair-wise acc", acc)
    return {
        "metric": acc,
        "visual": visual,
        "outputs": output_list,
    }


def inputids_cleanup(input_ids: str):
    """
    Clean up special user/assistant token in the vlm prompt
    """
    input_ids = input_ids.strip().replace("\n", " ")
    if input_ids.startswith("user "):
        input_ids = input_ids[len("user "):]
    if input_ids.endswith("assistant"):
        input_ids = input_ids[:-len("assistant")]
    return input_ids.strip()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[str], title: str):
    """
    Plot classification confusion matrix

    Args:
        y_true (`List[int]`): 
            list of ground-truth values
        y_pred (`List[int]`):  
            list of predicted values
        labels (`List[str]`): 
            classification labels
        title (`str`): title for the plot

    Returns:
        `Image.Image`: confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert the BytesIO object to a PIL image
    image = Image.fromarray(np.array(Image.open(buf))).convert("RGB")
    plt.clf()
    plt.close()
    buf.close()

    return image


def plot_scatter_plot(y_true: List[float], y_pred: List[float], title: str):
    """
    Plot scaltter plot between predicted and ground-truth scores
    
    Args:
        y_true (`List[int]`): 
            list of ground-truth values
        y_pred (`List[int]`):  
            list of predicted values
        title (`str`): title for the plot

    Returns:
        `Image.Image`: scatter plot
    """
    plt.scatter(y_pred, y_true)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert the BytesIO object to a PIL image
    image = Image.fromarray(np.array(Image.open(buf))).convert("RGB")
    plt.clf()
    plt.close()
    buf.close()

    return image


def create_image_editing_visual(result_dict):
    # create image editing visualization from model output
    image_width, image_height = result_dict['input'][0].size
    font_size = image_width // 20
    font = ImageFont.truetype("assets/arial.ttf", font_size)
    image_space = font_size // 2
    header_height = int(font_size * 1.3)
    sentence_height = int(font_size * 1.5)
    
    width = (len(result_dict) - 1) * (image_width + image_space)
    height = header_height + len(result_dict['input']) * (image_height + sentence_height)
    full_image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(full_image)
    
    x = image_space
    y = image_space
    for header in result_dict:
        if 'prompt' in header:
            continue
        draw.text((x, y), header, font=font, fill='black')
        x += (image_width + image_space)
    
    y += header_height
    for i in range(len(result_dict['input'])):
        x = 0
        for k in result_dict:
            if 'prompt' in k:
                continue
            image = result_dict[k][i].resize((image_width, image_height))
            full_image.paste(image, (x, y))
            x += (image_width + image_space)
        y += image_height
        draw.text((image_space, y), result_dict['prompt'][i], font=font, fill='black')
        y += sentence_height
        
    return full_image


def prepare_data_and_evaluate(
    scorer: ADIEEScorer, 
    scorer_processor: AutoProcessor, 
    input_image: Image.Image, 
    output_image: Image.Image, 
    instruction: str, 
    weight_dtype: torch.dtype, 
    device: str
):
    """
    Function to prepare data and run evaluation scorer

    Args:
        scorer (`ADIEEScorer`):
            model to get image editing evaluation score
        scorer_processor (`AutoProcessor`):
            input processor for the scorer
        input_image (`Image.Image`):
            original image before editing
        output_image (`Image.Image`):
            edited image
        instruction (`str`):
            edit instruction
        weight_dtype (`torch.dtype`):
            data type to convert weight to
        device (`str`):
            GPU device to run experiment on

    Return:
        `float`: evaluation score
    """
    data = {}
    inputs = prepare_question_answer(
        processor=scorer_processor,
        split="test",
        input_image=input_image,
        output_image=output_image,
        instruction=instruction,
    )
    for k, v in inputs.items():
        data[k + f"_1"] = v
    batch = prepare_batch(data, weight_dtype=weight_dtype, device=device)
    with torch.no_grad():
        ret_dict = run_model(
            model=scorer, 
            batch=batch, 
            is_train=False
        )
    return ret_dict["score"][0].item()
