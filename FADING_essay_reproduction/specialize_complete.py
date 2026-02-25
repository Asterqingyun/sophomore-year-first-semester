# code adapted from https://github.com/huggingface/diffusers/blob/v0.10.0/examples/dreambooth/train_dreambooth.py

import os
import argparse
import json
import hashlib
import itertools
import logging
import math
import numpy as np
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler

# from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

# %%

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.10.0.dev0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",  # 配置类的基类
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]  # 看配置里的architecture的模型类

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(  # 添加参数，包括名称 默认 值 帮助信息 是否需要
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of training images.",
    )
    parser.add_argument(
        "--instance_age_path",
        type=str,
        default=None,
        required=True,
        help="A numpy array that contains the initial ages of training images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="photo of a person",
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="finetune_double_prompt",
        required=False,
        help="Specialization mode, 'finetune_double_prompt'|'finetune_single_prompt'",
    )

    #####
    parser.add_argument(
        "--revision",
        type=str,  # 值的类型
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=150,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
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
        default=5e-6,
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
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# %%
class DreamBoothDatasetAge(Dataset):
    """
    return instance (img, token of 'photo of a xx year old person')
    and (img, token of 'photo of a person')
    """

    def __init__(
        self,
        instance_data_root,
        instance_age_path,  # numpy array that stores the age of training images
        tokenizer,
        instance_prompt="photo of a person",
        size=512,
        center_crop=False,
    ):

        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        if self.instance_data_root.is_file():
            self.instance_images_path = [Path(instance_data_root)]  # 文件直接加入列表
        else:
            self.instance_images_path = [
                os.path.join(instance_data_root, filename)
                for filename in os.listdir(instance_data_root)
            ]
        # path就去找，然后加入列表

        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        self.age_labels = np.load(instance_age_path)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),  # 调整图片大小
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),  # 裁剪
                transforms.ToTensor(),  # H,W,C的图片变成C,H,W的tensor，并且归一化到0-1之间
                transforms.Normalize([0.5], [0.5]),  # mean and std 的归一化
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(
        self, index
    ):  # 对对象进行切片操作时的实际返回值，返回的是这样的一个包含prompt,age,两个prompt的token的字典，index选择的是
        example = {}

        instance_image_path = self.instance_images_path[
            index % self.num_instance_images
        ]
        # 常见于把数据集当成“可循环”的，防止某些情况下 index 超出实际图片数时报错。
        instance_image = Image.open(instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(
            instance_image
        )  # 转变成tensor后放进字典里

        example["instance_prompt"] = self.instance_prompt  # 加入prompt
        example["instance_prompt_ids"] = self.tokenizer(  # token(photo of a person)
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids  # tokenizer的结果

        instance_image_name = instance_image_path.split("\\")[-1]
        instance_image_age = int(
            self.age_labels[self.age_labels[:, 0] == instance_image_name][0, 1]
        )  # age-labels的第一列是名字，第二列是年龄,哪些名字和当前的图片名相同的一个bool array
        # 然后True的那一个age_labels选中，然后取第一个label，拿出第1列年龄
        # age_labels:至少是个形如 (N, 2) 的 numpy 数组，第 0 列是文件名，第 1 列是年龄，
        example["instance_image_age"] = instance_image_age
        instance_age_prompt = self.instance_prompt.replace(
            " a ", f" a {instance_image_age} year old "
        )
        # add 年级
        example["instance_age_prompt"] = instance_age_prompt
        example["instance_age_prompt_ids"] = (
            self.tokenizer(  # token(photo of a xx year old person)
                instance_age_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        )

        example["blank_prompt_ids"] = self.tokenizer(  # token("")
            "",
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example


# %


def collate_fn(examples, finetune_mode="finetune_double_prompt"):
    if len(examples) != 1:
        raise ValueError("batchsize can only be 1..")

    if finetune_mode == "finetune_double_prompt":
        input_ids = [example["instance_prompt_ids"] for example in examples]
        input_ids += [example["instance_age_prompt_ids"] for example in examples]
    # 如果是两个prompt，那么就直接加入，否则就进行2遍
    elif finetune_mode == "finetune_single_prompt":
        input_ids = [example["instance_age_prompt_ids"] for example in examples]
        input_ids += [example["instance_age_prompt_ids"] for example in examples]

    # elif finetune_mode == "finetune_single_prompt_no_age":
    #     input_ids = [example["instance_prompt_ids"] for example in examples]
    #     input_ids += [example["instance_prompt_ids"] for example in examples]
    #
    # elif finetune_mode == "finetune_no_prompt":
    #     input_ids = [example["blank_prompt_ids"] for example in examples]
    #     input_ids += [example["blank_prompt_ids"] for example in examples]

    else:
        raise ValueError("invalid finetune_mode")

    pixel_values = [example["instance_images"] for example in examples]
    pixel_values += [example["instance_images"] for example in examples]
    # 加入两遍初始的图像
    pixel_values_ages = [example["instance_image_age"] for example in examples]
    # 加入一遍age
    pixel_values = torch.stack(pixel_values)  # 生成一个新的维度，把这些图片们堆叠起来
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # 转换成连续内存
    input_ids = torch.cat(input_ids, dim=0)  # 将prompt_ids直接拼接起来

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "pixel_values_ages": pixel_values_ages,
    }

    return batch  # 获得训练batch


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


# %%
def main(args):
    argparse_dict = vars(args)  # 转成dict

    argparse_json = json.dumps(
        argparse_dict, indent=4
    )  # 字典转成json字符串，缩进4个空格
    print("model configuration:\n", argparse_json)
    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as outfile:
        outfile.write(argparse_json)
    # 把配置存成json文件
    logging_dir = Path(
        args.output_dir, args.logging_dir
    )  # 日志目录进行拼接(path的用处)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,  # 这个是用于多卡的时候的加速的，防止每一步都对齐，其实只用在更新的前一步对齐
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )  # 所有 rank，所有参数，在同一个 backward step 必须同时参与梯度同步，但是text_encoder很多时候没有参与

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if (
        accelerator.is_local_main_process
    ):  # 判断主进程，仅仅主进程进行打印，只显示警告及以上级别
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()  # 设置级别
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()  # 非主进程只有error才会打印
        diffusers.utils.logging.set_verbosity_error()  # 非主进程只有error才会打印

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            repo = Repository(
                args.output_dir, clone_from=repo_name
            )  # 创建or打开这个repo，然后定期push

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")  # 写入一些进入gitignore的文件
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,  # 有名字直接按名字
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(  # 没有名字就从之前训好的来进行。从这个path/name的子目录中来获取
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path,
        args.revision,  # 根据模型选择text encoder类
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"  # 从子目录加载，并且实例化对象
        
        """DDPMscheduler定义了：
前向过程：如何逐步给图片加噪声（训练时用）
公式：noisy_image = sqrt(alpha_t) * image + sqrt(1 - alpha_t) * noise
反向过程：如何从噪声恢复图片（推理时用）
时间步调度：1000 步的 alpha/beta 参数如何变化
执行流程：
""",
    )

    def pred_original_samples(  # 根据noise获得初始图像
        samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ):
        sqrt_alpha_prod = noise_scheduler.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = (
            sqrt_alpha_prod.flatten()
        )  # 将维度展平，变成一维的向量,变成(batch_size,)
        while len(sqrt_alpha_prod.shape) < len(samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(
                -1
            )  # 因为sample是(B,C,H,W)的，所以要扩展维度，扩展成(B,1,1,1)

        sqrt_one_minus_alpha_prod = (
            1 - noise_scheduler.alphas_cumprod[timesteps]
        ) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(
                -1
            )  # 因为sample是(B,C,H,W)的，所以要扩展维度，扩展成(B,1,1,1)

        original_samples = (
            samples - sqrt_one_minus_alpha_prod * noise  # 预测的x_0
        ) / sqrt_alpha_prod
        return original_samples

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,  # 从对应的类中加载text_encoder
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(  # 从预训练模型中加载vae
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(  # 从预训练模型中加载unet
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)  # 如果不训练train_text_encoder，那么就冻结它

    if args.enable_xformers_memory_efficient_attention:  # xformer可以省显存
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()  # 启用梯度检查点，节省显存，但是会增加计算时间
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    # 用于控制在 Ampere 或更新的 GPU 上是否允许使用 TensorFloat-32 张量核心进行矩阵乘法运算。allow_tf32 即将弃用。
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps  # 放大学习率，但是对 text encoder 极其危险；大 batch：梯度被平均了，update 变“温柔”
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:  # 使用8bit Adam优化器
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW  # 否则就使用一般的Adamw

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters())
        if args.train_text_encoder  # 如果要训练文本encoder的话，那么把两个参数串起来训练，否则只训练u-net
        else unet.parameters()
    )
    optimizer = optimizer_class(  # 设置学习率 (lr)、Adam 算法的 Beta 参数、权重衰减 (weight_decay) 和数值稳定性项 (eps)。
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,  # Adam 需要学习
        eps=args.adam_epsilon,
    )

    train_dataset = DreamBoothDatasetAge(
        instance_data_root=args.instance_data_dir,
        instance_age_path=args.instance_age_path,
        tokenizer=tokenizer,
        instance_prompt=args.instance_prompt,
        size=args.resolution,
        center_crop=args.center_crop,  # 加载dataset
    )
    train_dataloader = torch.utils.data.DataLoader(  # 累积成一个一个batch
        train_dataset,
        batch_size=args.train_batch_size,  # 从dataset中加载dataloader
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, args.finetune_mode),
        num_workers=0,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.gradient_accumulation_steps  # 总 batch 数除以梯度累积步数，并向上取整就是需要更新多少次参数;梯度累计步数就是多少个batch更新一次
    )
    if args.max_train_steps is None:
        args.max_train_steps = (
            args.num_train_epochs * num_update_steps_per_epoch
        )  # 如果没有设置最大训练步数，那么就用 epoch 数乘以每个 epoch 的更新步数
        overrode_max_train_steps = True
    # scheduler.step() 是在每一个 micro step 被调用的
    lr_scheduler = get_scheduler(  # 创建学习率调度器（如 linear, cosine 等）。
        # 它控制学习率如何随训练过程变化（例如预热 warmup 阶段和随后的衰减）。
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps
        * args.gradient_accumulation_steps,  # 预热步数，一共更新多少次*我所认为的预热步数
        num_training_steps=args.max_train_steps
        * args.gradient_accumulation_steps,  # 总训练步数，一次更新中多少个batch*我所认为的总训练步数
        # num_cycles=args.lr_num_cycles,
        # power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = (
            accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        )
        """
        
        将所有相关的 PyTorch 训练对象（优化器、模型、数据加载器、学习率调度器）在创建后立即传递给 prepare() 方法。该方法会将模型包装在一个为您的分布式设置优化的容器中，使用 Accelerate 版本的优化器和调度器，并为您的数据加载器创建一个分片版本，以便在 GPU 或 TPU 之间分发
        """
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unet.dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {unet.dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and text_encoder.dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {text_encoder.dtype}. {low_precision_error_string}"
        )
    # ccelerator.prepare 可能会改变 train_dataloader 的长度（例如在多 GPU 并行时拆分数据），因此需要重新计算每个 epoch 的步数、总步数和总 epoch 数，以确保准确。
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))
    # 加载跟踪器，看主进程
    # Train!
    total_batch_size = (
        args.train_batch_size  # 总共处理器处理的样本数量
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(
                dirs, key=lambda x: int(x.split("-")[1])
            )  # global_step 最大的 checkpoint
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = (
            global_step * args.gradient_accumulation_steps
        )  # 一共训练过的batch数
        first_epoch = (
            resume_global_step // num_update_steps_per_epoch
        )  # 训练了几个epoch
        resume_step = (
            resume_global_step % num_update_steps_per_epoch
        )  # 当前epoch中第几个batch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(
        first_epoch, args.num_train_epochs
    ):  # 从训练了几个epoch开始往后训练
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()  # 训练文本

        for step, batch in enumerate(
            train_dataloader
        ):  # 取出来一个个batch,step是batch数
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch
                == first_epoch  # 如果是从checkpoint恢复，并且是第一个epoch并且步数小于恢复的步数,跳过这些步数
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(
                        1
                    )  # bar在已经更新过的地方的每一个参数更新的开头+1
                continue

            with accelerator.accumulate(unet):
                # 如果有梯度累积（accumulation steps > 1），accelerator 会自动处理何时进行真正的反向传播和参数更新。
                # Convert images to latent space，编码而后缩放
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                noise_single = torch.randn_like(
                    latents[0]
                )  # latents是(B,C,H,W),取第0个样本，生成和它一样shape的噪声
                bsz = latents.shape[0]
                # 多少个batch
                noise = torch.cat([noise_single.unsqueeze(0)] * bsz, dim=0)
                # 在第0维度上把noise_single复制bsz次，变成和latents一样的shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )  # 随机采样时间步长，(bsz,)表示每个样本一个时间步长，也就是每个图都有一个时间步
                timesteps = timesteps.long()  # 转成long类型

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # 直接用调度器里面的alpha/beta参数来加噪声，得到有噪声的潜在表示
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                # 使用文本编码器将输入的提示词 (input_ids) 转换为文本嵌入 (encoder_hidden_states,是模型的第一个输出)，作为 UNet 的条件输入。
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,  # t,z_t,embedding得到
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                pred_denoised_latents = pred_original_samples(
                    samples=noisy_latents, noise=model_pred, timesteps=timesteps
                )  # 由噪声 和时间步长 x_t预测的去噪潜在表示

                # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                pred_noisy_latents = noise_scheduler.add_noise(
                    latents, model_pred.to(dtype=weight_dtype), timesteps
                )  # 由预测的噪声 和时间步长 重新加噪，得到预测的有噪潜在表示

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # 让预测出来的噪声和真实噪声进行mse计算
                instance_loss = loss
                accelerator.backward(loss)
                instance_loss = loss
                optimizer.step()                   # 尝试更新参数（accelerator 会自动限流）
                lr_scheduler.step()                # 更新学习率
                optimizer.zero_grad()              # 清空梯度
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if (
                    global_step % args.checkpointing_steps == 0
                ):  # 每隔多少步保存一次模型
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(
                            save_path
                        )  # 保存当前状态，包括模型权重、优化器状态等，以便后续恢复训练
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "instance_loss": instance_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            progress_bar.set_postfix(**logs)  # 在进度条上显示日志信息
            accelerator.log(logs, step=global_step)  # 记录日志

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()  # 同步之后
    if accelerator.is_main_process:  # 主进程
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
            safety_checker=None,      # 关键：不加载安全检查器
        feature_extractor=None    # 关键：不加载特征提取器
        )
        # 在主进程中，将训练好的 UNet 和 Text Encoder 组装回完整的 DiffusionPipeline。
        # 保存最终的 Pipeline 到输出目录（这就是你可以在 webui 中加载的模型）。
        # 如果有配置，推送到 Hugging Face Hub。
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            repo.push_to_hub(
                commit_message="End of training", blocking=False, auto_lfs_prune=True
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()

    # args.instance_data_dir = 'specialization_data/training_images'
    # args.instance_age_path = 'specialization_data/training_ages.npy'
    #
    # args.output_dir = 'specialized_models/tmp'

    main(args)
