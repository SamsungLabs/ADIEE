from typing import List
import os
import argparse
import random
import json
from PIL import Image
from collections import defaultdict
import glob
import copy
import math
import csv
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvF
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoTokenizer
import datasets

# LLaVA family padding token index
IGNORE_INDEX = -100

# Question template to prompt model
QUESTION_LIST = [
    "Can you rate how successful the edit instruction \"{instruction}\" has been executed from the first image to the second image with a score from 0 to 10?",
    "Please rate how successful the edit instruction \"{instruction}\" has been executed from the first image to the second image with a score from 0 to 10.",
    "How successful the edit instruction \"{instruction}\" has been executed from the first image to the second image? Please respond with a score from 0 to 10.",
    "How successful the edit instruction \"{instruction}\" has been executed from the first image to the second image? Please output a score from 0 to 10.",
]

# Answer template of expected model output
SCORE_TOKEN = "[SCORE]"
ANSWER_LIST = [
    f"It is {SCORE_TOKEN}.",
    f"Sure, {SCORE_TOKEN}.",
    f"Sure, it is {SCORE_TOKEN}.",
    f"Sure, the score is {SCORE_TOKEN}.",
    f"{SCORE_TOKEN}.",
]

# evaluation score weights for balanced score sampling
WEIGHT_DICT = {0: 4.11415770609319, 1: 6.060774064100533, 2: 9.254615818753527, 3: 12.355758880516685, 8: 14.027251619210558, 7: 14.299862962501557, 6: 14.62788326749076, 4: 14.981075437222657, 5: 16.24239422668742, 9: 16.41661899313501, 10: 241.65263157894736}

def pad_tensor(t: torch.Tensor, pad_len: int, pad_val:int):
    """
    Function to pad tensor to a given length with specified values

    Args:
        t (`torch.Tensor`): 
            tensor to be padded. The expected shape of the tensor is batch_size x length
        pad_len (`int`): 
            length to pad tensor to
        pad_val (`int`): 
            value to pad tensor with
    
    Returns:
        `torch.Tensor`: a padded tensor
    """
    assert pad_len >= t.shape[1], \
        f"The final length of the tensor should be larger than its current length: {pad_len} < {t.shape[1]}"
    pad_tensor = torch.ones((t.shape[0], pad_len-t.shape[1])).to(t.dtype)
    return torch.cat((t, pad_tensor*pad_val), dim=-1)


def collate_fn(batch: list, pad_token_id: int):
    """
    Custom collate function for dataloader that pad language model inputs (input_ids, attention_mask) to to fixed length
    
    Args:
        batch (`list`): 
            list of samples
        pad_token_ids (`int`): 
            value to pad tensor with

    Returns:
        `dict`: a data sample batch in a dictionary
    """

    # create dictionary to store data
    batch_dict = defaultdict(list)
    for data in batch:
        for k, v in data.items():
            batch_dict[k].append(v)
    
    # pad input_ids and attention_mask across samples to fixed length
    for k, v in batch_dict.items():
        if "input_ids_" in k:
            input_ids_list = v
            attention_mask_list = batch_dict[k.replace("input_ids", "attention_mask")]
            
            max_len = max([x.shape[1] for x in input_ids_list])
            input_ids_list = [
                pad_tensor(x, max_len, pad_token_id) for x in input_ids_list
            ]
            attention_mask_list = [
                pad_tensor(x, max_len, 0) for x in attention_mask_list
            ]
            batch_dict[k] = input_ids_list
            batch_dict[k.replace("input_ids", "attention_mask")] = attention_mask_list
    
    # concatenate tensors across samples
    for k, v in batch_dict.items():
        batch_dict[k] = torch.cat(v, dim=0)

    # convert defaultdict to dict to avoid accelerator.prepare error
    return dict(batch_dict)


def hconcat_image(image1: Image.Image, image2: Image.Image):
    """
    Functional to horizontally concatenate two images together
    """
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)

    new_image = Image.new("RGB", (total_width, max_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    return new_image


def get_edit_index(dict_key: str):
    """
    Helper function to parse raw data key and get the edit turn index
    """
    return int(dict_key.split("_")[-1])


def prepare_question_answer(
    processor: AutoProcessor, 
    split: str, 
    input_image: Image.Image, 
    output_image: Image.Image, 
    instruction: str
):
    """
    Function to prepare input arguments for the model

    Args:
        processor (`AutoProccesor`):
            module to preprocess image and text input with a chat template
        split (`str`): 
            define the experiment stage. if == train, the assistant answer is appended after the question to the assistant, otherwise, the answer is omitted
        input_image (`Image.Image`): 
            the original image before edit
        output_image (`Image.Image`): 
            the edited image
        instruction (`str`): 
            the edit instruction

    Returns:
        `dict`: a dictionary of processed values as VLM inputs
    """

    # randomly sample a question and answer template to avoid over-fitting
    question = random.choice(QUESTION_LIST).format(instruction=instruction)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True
    )

    # append the answer template for training
    if split == "train":
        answer = random.choice(ANSWER_LIST)
        prompt += " " + answer
    
    # resize and horizontally concatenate images
    image = hconcat_image(input_image.resize((512, 512)), output_image.resize((512, 512)))
    inputs = processor(
        text=prompt, 
        images=image,
        return_tensors="pt"
    )
    ret = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "pixel_values": inputs.pixel_values,
        "image_sizes": inputs.image_sizes
    }
    return ret


def augment(image_list: List[Image.Image]):
    """
    Function to augment a list of images through flipping and rotattion
    """
    if random.random() > 0.5:
        image_list = [tvF.hflip(x) for x in image_list]
    if random.random() > 0.5:
        image_list = [tvF.vflip(x) for x in image_list]
    rotation_angle = random.choice([i * 90 for i in range(4)])
    if rotation_angle > 0:
        image_list = [tvF.rotate(x, rotation_angle) for x in image_list]
    return image_list


def score_heuristics(model_name: str, metrics_dict: dict):
    """
    Heuristics to automatically assign evaluation scores to images
    (0 - failed, 1 - partial success, 2 - success)
    In case where we are not sure what score to assign, return -1
    """
    if metrics_dict["clip_dir"] == float("nan") and (model_name not in ["ip2p", "magicbrush", "aurora"]):
        # if the editing method is not instruction-guided, thus prompt-guided,
        # then the absence of CLIP directional score means that 
        # the source/target image captions are the same,
        # the prompt-guided editing methods will produce failed edits
        return 0
    if metrics_dict["clip_dir"] != float("nan") and metrics_dict["clip_dir"] <= 0:
        # the CLIP directional score of the output is worse than the case where the output == input
        return 0
    if metrics_dict["clip_sim"] <= 0.83 and metrics_dict["dino_sim"] <= 0.4:
        # the CLIP/DINO similarity scores are below the pre-computed 5th percentile of scores of all input images in the dataset
        return 0
    if model_name in ["diffedit", "p2p0", "sdedit", "t2l"]:
        # we identified 4 editing methods that produce bad results overall
        return 0
    if (model_name in ["magicbrush", "aurora"]) and metrics_dict["clip_dir"] != float("nan") and metrics_dict["clip_dir"] <= 0.2:
        # we identified 2 editing methods that produce decent results overall
        # if the CLIP directional scores are below InstructPix2Pix CLIP filter threshold
        # assign them partial success label
        return 1
    return -1


class ADIEEDataset(Dataset):
    def __init__(self, args: argparse.Namespace, processor: AutoProcessor, split: str):
        if args.seed:
            random.seed(args.seed)
        self.args = args
        self.root = args.dataset_dir
        self.processor = processor

        self.split = split
        if split == "train":
            start = 0
            end = 0.9
        else:
            start = 0.9
            end = 1

        self.annotation_list = []

        # add samples from multi-turn edit sequences
        real_data_list = self.get_real_data_list(self.root)
        self.real_data_list = real_data_list[int(start*len(real_data_list)):int(end*len(real_data_list))]
        self.annotation_list += self.real_data_list
        
        # add samples generated by editing models
        syn_data_list = self.get_syn_data_list(self.root)
        self.annotation_list += syn_data_list[int(start*len(syn_data_list)):int(end*len(syn_data_list))]

    def get_real_data_list(self, root: str):
        """
        get samples from multi-turn edit sequences
        """
        self.real_data_root = f"{root}/SEED-Data-Edit-Part2-3/multi_turn_editing"
        json_list = glob.glob(f"{self.real_data_root}/annotations/split_*.jsonl")
        json_list.sort()
        annotation_list = []
        for json_path in json_list:
            annotation_list += [json.loads(l) for l in open(json_path).readlines()]

        # some samples in the sequence needs to be omitted due to corrupted image files
        annotation_list = [
            annotation for annotation in annotation_list if \
            not any(annotation["source_image"].startswith(f"20240330_315P_1323turns/Data/{x}") for x in range(208, 213))
        ]
        return annotation_list
    
    def get_syn_data_list(self, root: str):
        """
        get samples generated by the editing models
        """
        self.syn_data_root = f"{root}/ADIEE-MagicBrush-Data"
        self.syn_ds = datasets.load_from_disk(self.syn_data_root)
        annotation_list = []
        for split in self.syn_ds:
            annotation_list += [(split, i) for i in range(len(self.syn_ds[split]))]
        random.shuffle(annotation_list) # otherwise most global edit will be in the validation split
        return annotation_list
            
    def __len__(self):
        return len(self.annotation_list)
    
    def get_real_item(self, annotation: dict):
        """
        function to select and preprocess sample from mult-turn edit sequences
        """
        # need to make a copy to avoid pre-processing carry over across function call
        annotation = copy.deepcopy(annotation) 
        
        # add source image as edit_image_0 to sample it later
        annotation["edit_image_0"] = annotation["source_image"] 

        # randomly choose an image pair as the original image and the ground-truth edited image
        input_image_key = gt_image_key = "edit_image_0"
        # the ground-truth needs to be further down the edit sequence than the original 
        while get_edit_index(input_image_key) >= get_edit_index(gt_image_key):
            input_image_key, gt_image_key = random.sample([k for k in annotation if k.startswith("edit_image_")], k=2)

        # concatenate instructions between the original and the ground-truth image in random order
        instruction_key_list = [f"instruction_{i+1}" for i in range(get_edit_index(input_image_key), get_edit_index(gt_image_key))]
        random.shuffle(instruction_key_list)
        instruction = ". ".join([annotation[k] for k in instruction_key_list])
        instruction.replace("..", ".")

        # add random image which should get the lowest score when treated as the edited output
        random_image_key = "edit_image_-1"
        annotation[random_image_key] = random.choice(self.real_data_list)["source_image"]

        # category images to different evaluation scores and randomly sample one 
        output_key_list = [k for k in annotation if k.startswith("edit_image_")]
        output_key = random.choice(output_key_list)
        
        if get_edit_index(output_key) <= get_edit_index(input_image_key):
            # the image is in an eariler edit turn than the original image
            # the evaluation score should be the lowest
            # this includes the random image 
            score = 0
        elif get_edit_index(output_key) > get_edit_index(gt_image_key):
            # the image is in a latter edit turn than the ground-truth image
            # the evaluation score should be in the middle due to over-editting
            score = 0.5
        else:
            score = (get_edit_index(output_key) - get_edit_index(input_image_key)) \
                / (get_edit_index(gt_image_key) - get_edit_index(input_image_key))

        # normalize the score to 0-10
        score *= 10

        return {
            "instruction": instruction,
            "input_image": Image.open(f"{self.real_data_root}/images/{annotation[input_image_key]}"),
            "output_image": Image.open(f"{self.real_data_root}/images/{annotation[output_key]}"),
            "score": score,
            "annotation": f"in: {get_edit_index(input_image_key)}, gt: {get_edit_index(gt_image_key)}, o1: {get_edit_index(output_key)}, ",
        }

    def get_syn_item(self, annotation):
        """
        function to select and preprocess sample generated by the editing models
        """
        instruction = annotation["instruction"]
        source_image = annotation["source_img"]

        model_list = [
            "source",
            "target",
            "cyclediff",
            "diffedit", 
            "pr2pr",
            "p2p0", 
            "sdedit",
            "t2l", 
            "ip2p", 
            "magicbrush", 
            "aurora",
        ]

        # whether to use our scoring heuristics
        if self.args.use_heuristics:
            score_dict = {
                "source": 0,
                "target": 2,
            }
            for model_name in model_list[2:]:
                metrics_dict = {
                    metric_name: annotation[model_name + f"_{metric_name.replace('_', '')}"] 
                    for metric_name in ["clip_dir", "clip_sim", "dino_sim"]
                }
                model_score = score_heuristics(model_name, metrics_dict)
                if model_score != -1:
                    score_dict[model_name] = model_score
            output_key = random.choice(list(score_dict.keys()))
            score = score_dict[output_key] * 5 # map 0, 1, 2 to range 0-10
        else:
            score_list = [0] + [annotation[model + "_score"] for model in model_list[1:]]
            weight_list = [WEIGHT_DICT[round(score)] for score in score_list]

            output_key, score = random.choices(list(zip(model_list, score_list)), weights=weight_list, k=1)[0]
        
        output_image = annotation[output_key + "_img"]
        return {
            "instruction": instruction,
            "input_image": source_image,
            "output_image": output_image,
            "score": score,
            "annotation": f"o1: {output_key}, "
        }

    def __getitem__(self, index: int):
        """
        function to get samples from two datasets and preprocess it for model training
        """
        annotation = self.annotation_list[index]
        if type(annotation) == tuple:
            # get sample generated by editing models
            syn_split, syn_index = annotation
            ret = self.get_syn_item(self.syn_ds[syn_split][syn_index])
        else:
            # get samples from multi-turn edit sequences
            ret = self.get_real_item(annotation)
            
        instruction = ret["instruction"]
        image_list = [ret["input_image"].convert("RGB"), ret["output_image"].convert("RGB")]

        # augment images for training
        if self.split == "train":
            image_list = augment(image_list)
        
        # prepare model input and ouput
        data = {}
        inputs = prepare_question_answer(
            processor=self.processor,
            split=self.split,
            input_image=image_list[0],
            output_image=image_list[1],
            instruction=instruction,
        )
        for k, v in inputs.items():
            data[k + f"_1"] = v

        data["score"] = torch.tensor([ret["score"]])

        # since validation is done per sample, store raw image and text for visualization
        if self.split != "train":
            data["input"] = image_list[0]
            data["output_1"] = image_list[1]
            data["instruction"] = ret["annotation"] + instruction
        return data


class ImagenHubDataset(Dataset):
    def __init__(self, args: argparse.Namespace, processor: AutoProcessor):
        self.root = f"{args.dataset_dir}/ImagenHub/results/ImagenHub_Text-Guided_IE"
        self.instruction_dict = json.load(open(os.path.join(self.root, "dataset_lookup.json")))
        rater_list = glob.glob(f"{args.dataset_dir}/ImagenHub/eval/human_ratings/Text-Guided_IE/*.tsv")
        rater_list.sort()
        
        # load human rating
        rater_dict = defaultdict(list)
        for rater_path in rater_list:
            header = None
            tsv_file = csv.reader(open(rater_path), delimiter="\t")
            for l in tsv_file:
                if header is None:
                    header = l[1:]
                    continue
                else:
                    filename = l[0]
                    scores = l[1:]
                    for model_name, score in zip(header, scores):
                        if model_name == "Imagic":
                            continue
                        score_list = eval(score)
                        rater_dict[filename + ", " + model_name].append(math.sqrt(score_list[0] * score_list[1]))
        self.rater_dict = rater_dict
        self.processor = processor

    def __len__(self):
        return len(self.rater_dict)
    
    def __getitem__(self, index: int):
        filename_modelname = list(self.rater_dict.keys())[index]
        filename, model_name = tuple(filename_modelname.split(", "))
        input_path = os.path.join(self.root, "input", filename)
        out_path = os.path.join(self.root, model_name, filename)
        instruction = self.instruction_dict[filename]["instruction"]
        images = [input_path,out_path]
        images = [Image.open(x).convert("RGB") for x in images]
        
        data = {}
        for i, output in enumerate(images[1:]):
            inputs = prepare_question_answer(
                processor=self.processor,
                split="test",
                input_image=images[0],
                output_image=output,
                instruction=instruction,
            )
            for k, v in inputs.items():
                data[k + f"_{i+1}"] = v

        score = np.array(self.rater_dict[filename_modelname]).mean()
        data["score"] = score
        data["filename"] = filename
        data["model"] = model_name
        data["input"] = images[0]
        data["output_1"] = images[1]
        data["instruction"] = instruction
        return data
    

class GenAIBenchDataset(Dataset):
    def __init__(self, args: argparse.Namespace, processor: AutoProcessor):
        self.root = f"{args.dataset_dir}/GenAI-Bench"
        self.dataset = datasets.load_from_disk(self.root)["test_v1"]
        self.processor = processor
        self.labels = ["leftvote", "rightvote", "tievote", "bothbad_vote"]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        data = self.dataset[index]
        instruction = data["instruct_prompt"]
        input_image = data["source_image"]
        output_1 = data["left_output_image"]
        output_2 = data["right_output_image"]
        label = data["vote_type"]
        label = self.labels.index(label)
        images = [input_image, output_1, output_2]

        data = {}
        for i, output in enumerate(images[1:]):
            inputs = prepare_question_answer(
                processor=self.processor,
                split="test",
                input_image=images[0],
                output_image=output,
                instruction=instruction,
            )
            for k, v in inputs.items():
                data[k + f"_{i+1}"] = v

        data["label"] = label
        data["input"] = images[0]
        data["output_1"] = images[1]
        data["output_2"] = images[2]
        data["instruction"] = instruction
        return data


class AURORABenchDataset(Dataset):
    def __init__(self, args: argparse.Namespace, processor: AutoProcessor):
        self.root = f"{args.dataset_dir}/AURORA"
        self.human_ratings = json.load(open(os.path.join(self.root, "human_ratings_updated_prompt.json")))
        self.processor = processor

    def __len__(self):
        return len(self.human_ratings)
    
    def __getitem__(self, index: int):
        human_rating = self.human_ratings[index]
        input_path = os.path.join(self.root, human_rating["input"])
        output_path = os.path.join(self.root, human_rating["gen"])
        instruction = human_rating["prompt"]
        score = human_rating["score"]
        images = [input_path, output_path]
        images = [Image.open(x).convert("RGB") for x in images]

        data = {}
        for i, output in enumerate(images[1:]):
            inputs = prepare_question_answer(
                processor=self.processor,
                split="test",
                input_image=images[0],
                output_image=output,
                instruction=instruction,
            )
            for k, v in inputs.items():
                data[k + f"_{i+1}"] = v

        data["score"] = score
        data["input"] = images[0]
        data["output_1"] = images[1]
        data["instruction"] = instruction
        data["input_path"] = human_rating["input"]
        data["output_path"] = human_rating["gen"]
        return data


class AURORABenchPairwiseDataset(Dataset):
    def __init__(self, args: argparse.Namespace, processor: AutoProcessor):
        self.root = f"{args.dataset_dir}/AURORA"
        self.human_ratings = json.load(open(os.path.join(self.root, "human_ratings_pairwise.json")))
        self.processor = processor
        self.labels = ["leftvote", "rightvote", "tievote"]

    def __len__(self):
        return len(self.human_ratings)
    
    def __getitem__(self, index: int):
        human_rating = self.human_ratings[index]
        input_path = os.path.join(self.root, human_rating["input"])
        output_path_list = [os.path.join(self.root, x) for x in human_rating["gens"]]
        instruction = human_rating["prompt"]
        label = human_rating["winner"]
        models = human_rating["models"]
        if label == "tie":
            label = "tievote"
        else:
            assert label in models, f"{label} not in {models}"
            if label == models[0]:
                label = "leftvote"
            else:
                label = "rightvote"
        label = self.labels.index(label)

        images = [input_path] + output_path_list
        images = [Image.open(x).convert("RGB") for x in images]

        data = {}
        for i, output in enumerate(images[1:]):
            inputs = prepare_question_answer(
                processor=self.processor,
                split="test",
                input_image=images[0],
                output_image=output,
                instruction=instruction,
            )
            for k, v in inputs.items():
                data[k + f"_{i+1}"] = v

        data["label"] = label
        data["input"] = images[0]
        data["output_1"] = images[1]
        data["output_2"] = images[2]
        data["instruction"] = instruction
        data["input_path"] = human_rating["input"],
        data["output_1_path"] = human_rating["gens"][0]
        data["output_2_path"] = human_rating["gens"][1]
        return data
    

class MagicBrushDataset(Dataset):
    def __init__(self, args: argparse.Namespace, tokenizer: AutoTokenizer, split: str, processor: AutoProcessor = None):
        self.tokenizer = tokenizer

        # for image editing with reinforcement learning, 
        # we need to processor to process inputs for the evaluation scorer
        self.processor = processor 

        if split == "train":
            self.root = f"{args.dataset_dir}/ADIEE-MagicBrush-Data"
            self.ds = datasets.load_from_disk(self.root)[split]
            self.resolution = args.resolution
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.root = f"{args.dataset_dir}/ImagenHub/results/ImagenHub_Text-Guided_IE"
            self.instruction_dict = json.load(open(os.path.join(self.root, "dataset_lookup.json")))
            self.ds = [{
                "source_img": Image.open(os.path.join(self.root, "input", k)),
                "target_img": Image.open(os.path.join(self.root, "GroundTruth", k)),
                "instruction": v["instruction"]
            } for k,v in self.instruction_dict.items()]
            self.resolution = 512
        
        self.split = split
        self.score_list = ["one", "two", "three", "four", "five"]

    def __len__(self):
        return len(self.ds)

    def tokenize_captions(self, caption: str, padding: str = "max_length", truncation: bool = True):
        """
        function to tokenize text caption
        """
        inputs = self.tokenizer(
            [caption], 
            max_length=self.tokenizer.model_max_length, 
            padding=padding, 
            truncation=truncation, 
            return_tensors="pt"
        )
        return inputs.input_ids[0]
    
    def __getitem__(self, index: int):
        data = self.ds[index]
        input_image_pil = data["source_img"].convert("RGB")
        instruction_text = data["instruction"]

        if self.processor is None: # reward-condition as prompt
            if self.split == "train":
                model_list = [
                    "source",
                    "target",
                    "cyclediff",
                    "diffedit", 
                    "pr2pr",
                    "p2p0", 
                    "sdedit",
                    "t2l", 
                    "ip2p", 
                    "magicbrush", 
                    "aurora" ,
                ]
                score_list = [0] + [data[model + "_score"] for model in model_list[1:]]
                weight_list = [WEIGHT_DICT[round(score)] for score in score_list]
                model_name = random.choices(model_list, weights=weight_list, k=1)[0]

                # get the output evaluation score
                if model_name == "source":
                    score = 0
                else:
                    score = round(data[f"{model_name}_score"])
                if score <= 2:
                    score_index = 0 # one
                elif score <= 4:
                    score_index = 1 # two
                elif score <= 6:
                    score_index = 2 # three
                elif score <= 8:
                    score_index = 3 # four
                elif score <= 10:
                    score_index = 4 # five
                score = self.score_list[score_index]
            else:
                model_name = "target"
                score = self.score_list[-1]

            reward_instruction = f". The image quality is {score} out of {self.score_list[-1]}."
            instruction_text = instruction_text + reward_instruction
            instruction_text = instruction_text.replace("..", ".")
        else:
            model_name = "target"

        instruction = self.tokenize_captions(instruction_text) if self.split == "train" else instruction_text
        
        # proprocess and augment (for training) images
        output_image_pil = data[f"{model_name}_img"].convert("RGB")
        image_list = [input_image_pil, output_image_pil]
        image_list = [x.resize((self.resolution, self.resolution)) for x in image_list]
        
        # augment images for training
        if self.split == "train":
            image_list = augment(image_list)
            image_list = [self.transform(x) for x in image_list]
            image_list[0] = image_list[0].unsqueeze(0)
            image_list[1] = image_list[1].unsqueeze(0)
            instruction = instruction.unsqueeze(0)

        data_dict = {
            "original_pixel_values": image_list[0],
            "edited_pixel_values": image_list[1],
            "input_ids": instruction,
        }

        # add inputs for evaluation model
        if self.processor is not None:
            inputs = prepare_question_answer(
                processor=self.processor,
                split="test",
                input_image=input_image_pil,
                output_image=data["target_img"].convert("RGB"), # placeholder for edited image later
                instruction=instruction_text,
            )
            data_dict.update({
                k + "_evaluation": v for k,v in inputs.items()
            })
        return data_dict
