from typing import List
import os
import argparse
from functools import partial
import math
import copy
from tqdm import tqdm
import logging
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
import transformers
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

from adiee import ADIEEScorer
from dataset import (
    ADIEEDataset, 
    collate_fn,
    SCORE_TOKEN
)
from utils import *


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="ADIEE Model Training")
    parser.add_argument(
        "--model_id", 
        default="llava-hf/llama3-llava-next-8b-hf",
        type=str,
        help="baseline VLM to train the ADIEE scorer from"
    )
    parser.add_argument(
        "--ce_loss_weight", 
        default=1.0, 
        type=float,
        help="weight for the scorer text output cross-entropy loss"
    )
    parser.add_argument(
        "--score_loss_weight", 
        default=10.0, 
        type=float,
        help="weight for the predicted score l1 loss"
    )
    parser.add_argument(
        "--dataset_dir", 
        default="./adiee_datasets", 
        type=str,
        help="path to training data"
    )
    parser.add_argument(
        "--use_heuristics", 
        action="store_true", 
        help="whether to use automatic scoring heuristics for training"
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ckpts/evaluation_model",
        help="The output directory where the checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=5,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=32,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        default=8,
        type=int,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha", 
        default=16, 
        type=int, 
        help="The scale of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_dropout", 
        default=0.05, 
        type=float, 
        help="The dropout ratio of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_target_modules", 
        default="q_proj,v_proj", 
        type=str,  
        help="The modules to add the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_amd",
        action="store_true",
        help="Whether to use AMD GPUs for training",
    )
    
    return parser.parse_args()


def save_model(
    model: ADIEEScorer, 
    processor: AutoProcessor, 
    save_path: str
):
    """
    save model and processor

    Args:
        model: `ADIEEScorer`: scorer model 
        processor: `AutoProcessor`: input processor corresponding to the model
        save_path: `str`: where the model and the processor should be saved 
    """
    # merge lora
    model.vlm = model.vlm.merge_and_unload()

    # save model and processor
    state_dict = model.state_dict()
    model.save_pretrained(save_path, state_dict=state_dict)
    processor.save_pretrained(save_path)


def find_linear_layers(model: LlavaNextForConditionalGeneration, lora_target_modules: List[str]):
    """
    Define LoRA metrics

    Args:
        model (`LlavaNextForConditionalGeneration`): 
            LLaVA-Next base model
        lora_target_modules (`List[str]`): 
            list of module types to add LoRA metrics with

    Returns:
        `list`: list of model layer names to add LoRA metrics 
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                [
                    x not in name
                    for x in [
                        "visual_model",
                        "vision_tower",
                        "multi_modal_projector",
                    ]
                ]
            )
            and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))


def init_model(args: argparse.Namespace, weight_dtype: str, device: str):
    """
    function to initialize the model with score prediction decoder and LoRA
    
    Args:
        args (`argparse.Namespace`): 
            training configuration arguments
        weight_dtype (`str`): 
            data type to load model weights with
        device (`str`): 
            GPU device ID
        is_train (`bool`):
            whether this is in training or testing stage

    Returns:
        `LlavaNextForConditionalGeneration`: initialized training model
        `AutoProcessor`: processor with score token 
    """
    if args.use_amd:
        vlm = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=weight_dtype,
            device_map=device,
            attn_implementation="eager" # without this, the training takes double the time
        )
    else:
        vlm = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=weight_dtype,
            device_map=device,
        )
    # define pad_token_id to suppress warning
    if vlm.generation_config.pad_token_id is None and vlm.generation_config.eos_token_id is not None:
        vlm.generation_config.pad_token_id = vlm.generation_config.eos_token_id

    # freeze weights for vision tower and multi_modal_projector
    vlm.vision_tower.requires_grad_(False)
    vlm.multi_modal_projector.requires_grad_(False)

    # emable input_embedding grad, needed to finetune vlm
    vlm.enable_input_require_grads()

    # kwargs needed for ddp training
    if args.gradient_checkpointing:
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # define lora
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_target_modules = find_linear_layers(
        vlm, args.lora_target_modules.split(",")
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    vlm = get_peft_model(vlm, lora_config)

    # add score_token
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.add_tokens(SCORE_TOKEN)
    score_token_idx = processor.tokenizer(SCORE_TOKEN, add_special_tokens=False).input_ids[0]
    vlm.resize_token_embeddings(len(processor.tokenizer))
    
    # create model
    config = vlm.config
    model = ADIEEScorer(
        config, 
        score_token_idx=score_token_idx,
        vlm=vlm, # for faster initialization
        out_channels=3 if args.use_heuristics else 1 # if using scoring heuristics, define the model as a classifier, else define it as a score predictor
    )
    return model, processor


def main(args):
    """
    Main training function
    """
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # load accelerator based on the GPU type
    if args.use_amd:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            project_config=accelerator_project_config,
            kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)]
        )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # finetune embed_tokens, lm_head, score decoder
    model, processor = init_model(args, weight_dtype, accelerator.device)
    model.vlm.language_model.model.embed_tokens.requires_grad_(True)
    model.vlm.language_model.lm_head.requires_grad_(True)
    model.score_decoder.requires_grad_(True)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision != "no":
        models = [model]
        # only upcast traintable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # model loss function
    if args.use_heuristics:
        criterion = torch.nn.CrossEntropyLoss() 
        loss_func = lambda ret_dict, batch: criterion(ret_dict["score"][0].float(), (batch["score"] / 5).long()) # map score 0-10 to 0, 1, 2
    else:
        loss_func = lambda ret_dict, batch: F.l1_loss(ret_dict["score"][0].float(), batch["score"].unsqueeze(1).float())

    # load training data
    train_dataset = ADIEEDataset(
        args=args,
        processor=copy.deepcopy(processor),
        split="train"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            pad_token_id=processor.tokenizer.pad_token_id,
        ),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_name = args.output_dir[:-1] if args.output_dir[-1] == "/" else args.output_dir
        tracker_name = os.path.basename(tracker_name)
        accelerator.init_trackers(tracker_name, config=tracker_config)

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            first_epoch = int(path.split("-")[1])
            initial_global_step = global_step = first_epoch * num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # define process bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # training
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                batch = prepare_batch(
                    batch=batch, 
                    weight_dtype=weight_dtype, 
                    device=accelerator.device,
                )
                ret_dict = run_model(
                    model=model,
                    batch=batch,
                    is_train=True
                )

                # get loss
                score_loss = loss_func(ret_dict, batch)
                ce_loss_list = ret_dict["ce_loss"]
                ce_loss = sum(ce_loss_list) / len(ce_loss_list)
                loss = args.score_loss_weight * score_loss + args.ce_loss_weight * ce_loss

                # back-propagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # log loss values
            logs = {
                "lr": lr_scheduler.get_last_lr()[0], 
                "loss": loss.detach().item(),
                "score_loss": score_loss.detach().item(),
                "ce_loss": ce_loss.detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        # save training state and model checkpoint
        if accelerator.sync_gradients and accelerator.is_main_process:
            if (epoch+1) % args.checkpointing_epochs == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                save_model(
                    unwrap_model(copy.deepcopy(model), accelerator), 
                    copy.deepcopy(processor), 
                    os.path.join(args.output_dir, str(epoch+1))
                )
            
        # run evaluation
        if (epoch+1) % args.validation_epochs == 0:
            run_eval(
                args=args,
                model=model,
                processor=processor,
                weight_dtype=weight_dtype,
                accelerator=accelerator,
                epoch=epoch
            )

    # save the final checkpoint
    if accelerator.sync_gradients and accelerator.is_main_process:
        save_model(
            unwrap_model(model, accelerator), 
            processor, 
            os.path.join(args.output_dir, "final")
        )


def run_eval(
    args: argparse.Namespace,
    model: ADIEEScorer, 
    processor: AutoProcessor, 
    weight_dtype: str, 
    accelerator: Accelerator, 
    epoch: int,
):
    """
    Evaluation function

    Args:
        args (`argparse.Namespace`): 
            training configuration
        model (`ADIEEScorer`): 
            scorer model to be trained
        weight_dtype (`str`): 
            data type to convert the samples to
        accelerator (`Accelerator`): 
            accelerator to preprocess the data
        epoch (`int`): 
            Current training epoch, used for logging
    """
    model.eval()
    device = accelerator.device
    
    # datasets to run evaluation on
    eval_dict = {
        "val_split": {"type": "loss"},
    }
    for benchmark_name in eval_dict:
        benchmark_func = eval(f"run_{benchmark_name}")
        ret = benchmark_func(
            args,
            model, 
            copy.deepcopy(processor), 
            weight_dtype, 
            device,
        )
        eval_dict[benchmark_name]["metric"] = ret["metric"]
        eval_dict[benchmark_name]["visual"] = ret["visual"]
    
    # save results to tensorboard
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for name in eval_dict:
                tracker.writer.add_images(
                    name, 
                    np.array(eval_dict[name]["visual"])[None,...], 
                    epoch+1, 
                    dataformats="NHWC"
                ) 
                tracker.writer.add_scalar(
                    f'{name} {eval_dict[name]["type"]}', 
                    eval_dict[name]["metric"], 
                    epoch+1
                )
    torch.cuda.empty_cache()


def run_val_split(
    args: argparse.Namespace, 
    model: ADIEEScorer, 
    processor: AutoProcessor, 
    weight_dtype: str, 
    device: str
):
    """
    Evaluation function for validation split 
    """
    val_dataset = ADIEEDataset(
        args=args,
        processor=processor,
        split="val"
    )

    dist = 0
    visual = None
    index_list = list(range(len(val_dataset)))
    for order, index in tqdm(enumerate(index_list), desc="eval on val split", total=len(val_dataset)):
        batch = prepare_batch(
            batch=val_dataset[index], 
            weight_dtype=weight_dtype, 
            device=device,
        )
        with torch.no_grad():
            ret_dict = run_model(
                model=model, # otherwise program will stuck on multi-gpu
                batch=batch,
                is_train=False,
            )
        pred_score = ret_dict["score"][0].item()
        gt_score = batch["score"][0]
        dist += abs(pred_score - gt_score)

        if order == 0:
            output_ids_1 = val_dataset.processor.decode(ret_dict["output_ids"][0][0], skip_special_tokens=True)
            result_dict = {
                "input": batch["input"],
                "output": batch["output_1"],
                "instruction": batch["instruction"],
                "output_ids": output_ids_1,
                "score": gt_score,
                "pred_score": pred_score,
            }
            visual = create_val_visual(result_dict)
    loss = dist / len(val_dataset)
    return {
        "metric": loss,
        "visual": visual,
    }


def create_val_visual(result_dict: dict):
    """
    Visualize one validation split sample and the predicted score
    """
    image_width, image_height = result_dict["input"].size
    font_size = image_width // 20
    font = ImageFont.truetype("assets/arial.ttf", font_size)
    image_space = font_size // 2
    header_height = int(font_size * 1.3)
    sentence_height = int(font_size * 1.5)
    
    width = 2 * (image_width + image_space) + image_space
    height = header_height + image_height + sentence_height * 4
    full_image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(full_image)
    
    x = image_space
    y = image_space
    for header in ["input", "output"]:
        draw.text((x, y), header, font=font, fill='black')
        image = result_dict[header].resize((image_width, image_height))
        full_image.paste(image, (x, y+header_height))
        x += (image_width + image_space)
    y += header_height + image_height
    draw.text((image_space, y), result_dict["instruction"], font=font, fill='black')
    
    y += sentence_height
    text = f"pred_score: {round(result_dict['pred_score'], 3)}, gt_score: {result_dict['score']}"
    draw.text((image_space, y), text, font=font, fill='black')

    y += sentence_height
    draw.text((image_space, y), "output_ids: " + result_dict["output_ids"], font=font, fill='black')
    return full_image


if __name__ == "__main__":
    args = parse_args()
    main(args)
