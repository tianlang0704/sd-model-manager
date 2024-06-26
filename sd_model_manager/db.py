from http import server
import os
import os.path
import io
import sys
import glob
import tqdm
import asyncio
import simplejson
from ast import literal_eval as make_tuple
from PIL import Image
from datetime import datetime
from sqlalchemy import select, delete, func, and_
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
import yaml

from sd_model_manager.utils.common import PATH, find_image, is_comfyui
from sd_model_manager.utils import safetensors_hack
from sd_model_manager.models.sd_models import Base, PreviewImage, SDModel, LoRAModel


DATABASE_NAME = os.getenv("DATABASE_NAME", "model_database")


def to_bool(s):
    if s is None or s == "None":
        return None
    return bool(s)


def to_int(s):
    if s is None or s == "None":
        return None
    return int(float(s))


def to_float(s):
    if s is None or s == "None":
        return None
    return float(s)


def to_str(s):
    if s is None or s == "None":
        return None
    return str(s)


def to_datetime(s):
    if s is None or s == "None":
        return None
    return datetime.fromtimestamp(float(s))


def to_json(s):
    if s is None or s == "None":
        return None
    try:
        return simplejson.loads(s)
    except Exception as ex:
        return None


def to_unique_tags(s):
    if s is None or s == "None":
        return None

    tags = set()

    d = simplejson.loads(s)
    for k, v in d.items():
        for t in v:
            tags.add(t)

    return len(tags)


def format_resolution(tuple_str, idx):
    try:
        t = make_tuple(tuple_str)
        if isinstance(t, (int, float)):
            return int(t)
        elif isinstance(t, tuple):
            return t[idx]
    except Exception as ex:
        return None


def can_load_image(path):
    try:
        image = Image.open(path)
        image.load()
        return True
    except Exception:
        return False


def find_preview_images(basepath):
    i = 0
    images = []
    path = basepath + ".png"
    if os.path.isfile(path):
        images.append(path)

    while True:
        if i == 0:
            path = f"{basepath}.preview.png"
        else:
            path = f"{basepath}.preview.{i}.png"
        if not os.path.isfile(path):
            break
        images.append(path)
        i += 1

    images = [os.path.normpath(i) for i in images if can_load_image(i)]
    return images


MODEL_TYPES = {
    "networks.lora": "LoRA",
    "sd_scripts.networks.lora": "LoRA",
    "networks.dylora": "DyLoRA",
}


MODEL_ALGOS = {
    "lora": "LoRA",
    "locon": "LoCon",
    "lokr": "LoKR",
    "loha": "LoHa",
    "ia3": "(IA)^3",
}


def format_module_name(m):
    module = m.get("ss_network_module", None)

    if module in MODEL_TYPES:
        return MODEL_TYPES[module]

    if module == "lycoris.kohya":
        args = simplejson.loads(m.get("ss_network_args") or "{}")
        algo = args.get("algo")
        return MODEL_ALGOS.get(algo, module)

    return module


class DB:
    def __init__(self):
        self.engine = None
        pass

    async def init(self, model_paths):
        path = os.path.join(PATH, DATABASE_NAME)
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{path}.db")

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self.AsyncSession = async_sessionmaker(bind=self.engine)
        
        async with self.AsyncSession() as session:
            stmt = select(func.count()).select_from(SDModel)
            count = (await session.execute(stmt)).scalar()

        print(f"Database is at {path}.db.")

        await self.scan(model_paths)
    
    def merge_comfy_paths(self, paths):
        if not is_comfyui():
            return paths
        if not paths:
            paths = []
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        comfyui_root_path = os.path.abspath(os.path.join(current_file_path, "..", "..", ".."))
        comfyui_model_path = os.path.join(comfyui_root_path, "models")
        if comfyui_model_path not in paths:
            paths.append(comfyui_model_path)

        try:
            extra_conf_path = os.path.join(comfyui_root_path, "extra_model_paths.yaml")
            extra_conf = yaml.safe_load(open(extra_conf_path, "r"))
            for key, value in extra_conf.items():
                base_path = value.get("base_path")
                if base_path is None:
                    continue
                if base_path not in paths:
                    paths.append(base_path)
        except:
            pass
        return paths
    
    async def scan(self, paths):
        paths = self.merge_comfy_paths(paths)
        if not paths:
            return
        for p in paths:
            if not os.path.isdir(p):
                raise RuntimeError(f"Invalid path: {p}")

        print("Building model database...")

        async with self.AsyncSession() as session:
            for path in paths:
                path = os.path.normpath(path)
                files = list(glob.iglob(f"{path}/**/*.safetensors", recursive=True))
                files += list(glob.iglob(f"{path}/**/*.ckpt", recursive=True))
                files += list(glob.iglob(f"{path}/**/*.pt", recursive=True))

                for f in tqdm.tqdm(files):
                    f = os.path.normpath(f)
                    filepath = f
                    query = select(LoRAModel).filter(LoRAModel.filepath == filepath)
                    row = (await session.execute(query)).one_or_none()
                    if row is not None:
                        continue
                    try:
                        metadata = safetensors_hack.read_metadata(f)
                    except:
                        metadata = {}

                    display_name = metadata.get("ssmd_display_name", None)
                    author = metadata.get("ssmd_author", None)
                    source = metadata.get("ssmd_source", None)
                    keywords = metadata.get("ssmd_keywords", None)
                    negative_keywords = metadata.get("ssmd_negative_keywords", None)
                    version = metadata.get("ssmd_version", None)
                    description = metadata.get("ssmd_description", None)
                    rating = to_int(metadata.get("ssmd_rating", None))
                    tags = metadata.get("ssmd_tags", None)

                    id = None
                    # if "ss_lr_scheduler" not in metadata:
                    #     continue
                    lora_model = LoRAModel(
                        id = id,
                        root_path = path,
                        filepath = filepath,
                        filename = os.path.basename(f),
                        last_embedded = datetime.min,
                        display_name = display_name,
                        author = author,
                        source = source,
                        keywords = keywords,
                        negative_keywords = negative_keywords,
                        version = version,
                        description = description,
                        rating = rating,
                        tags = tags,
                        model_hash = metadata.get("sshs_model_hash", None),
                        legacy_hash = metadata.get("sshs_legacy_hash", None),
                        session_id = to_int(metadata.get("ss_session_id", None)),
                        training_started_at = to_datetime(metadata.get("ss_training_started_at", None)),
                        output_name = metadata.get("ss_output_name", None),
                        learning_rate = to_float(metadata.get("ss_learning_rate", None)),
                        text_encoder_lr = to_float(metadata.get("ss_text_encoder_lr", None)),
                        unet_lr = to_float(metadata.get("ss_unet_lr", None)),
                        num_train_images = to_int(metadata.get("ss_num_train_images", None)),
                        num_reg_images = to_int(metadata.get("ss_num_reg_images", None)),
                        num_batches_per_epoch = to_int(metadata.get("ss_num_batches_per_epoch", None)),
                        num_epochs = to_int(metadata.get("ss_num_epochs", None)),
                        epoch = to_int(metadata.get("ss_epoch", None)),
                        batch_size_per_device = to_int(metadata.get("ss_batch_size_per_device", None)),
                        total_batch_size = to_int(metadata.get("ss_total_batch_size", None)),
                        gradient_checkpointing = to_bool(metadata.get("ss_gradient_checkpointing", None)),
                        gradient_accumulation_steps = to_int(metadata.get("ss_gradient_accumulation_steps", None)),
                        max_train_steps = to_int(metadata.get("ss_max_train_steps", None)),
                        lr_warmup_steps = to_int(metadata.get("ss_lr_warmup_steps", None)),
                        lr_scheduler = metadata.get("ss_lr_scheduler", None),
                        network_module = metadata.get("ss_network_module", None),
                        module_name = format_module_name(metadata),
                        network_dim = metadata.get("ss_network_dim", None),
                        network_alpha = metadata.get("ss_network_alpha", None),
                        network_args = to_json(metadata.get("ss_network_args", None)),
                        mixed_precision = to_bool(metadata.get("ss_mixed_precision", None)),
                        full_fp16 = to_bool(metadata.get("ss_full_fp16", None)),
                        v2 = to_bool(metadata.get("ss_v2", None)),
                        resolution_width = format_resolution(metadata.get("ss_resolution", None), 0),
                        resolution_height = format_resolution(metadata.get("ss_resolution", None), 1),
                        clip_skip = to_int(metadata.get("ss_clip_skip", None)),
                        max_token_length = to_int(metadata.get("ss_max_token_length", None)),
                        color_aug = to_bool(metadata.get("ss_color_aug", None)),
                        flip_aug = to_bool(metadata.get("ss_flip_aug", None)),
                        random_crop = to_bool(metadata.get("ss_random_crop", None)),
                        shuffle_caption = to_bool(metadata.get("ss_shuffle_caption", None)),
                        cache_latents = to_bool(metadata.get("ss_cache_latents", None)),
                        enable_bucket = to_bool(metadata.get("ss_enable_bucket", None)),
                        min_bucket_reso = to_int(metadata.get("ss_min_bucket_reso", None)),
                        max_bucket_reso = to_int(metadata.get("ss_max_bucket_reso", None)),
                        seed = to_int(metadata.get("ss_seed", None)),
                        keep_tokens = to_bool(metadata.get("ss_keep_tokens", None)),
                        dataset_dirs = to_json(metadata.get("ss_dataset_dirs", None)),
                        reg_dataset_dirs = to_json(metadata.get("ss_reg_dataset_dirs", None)),
                        tag_frequency = to_json(metadata.get("ss_tag_frequency", None)),
                        unique_tags = to_unique_tags(metadata.get("ss_tag_frequency", None)),
                        sd_model_name = metadata.get("ss_sd_model_name", None),
                        sd_model_hash = metadata.get("ss_sd_model_hash", None),
                        new_sd_model_hash = metadata.get("ss_new_sd_model_hash", None),
                        vae_name = metadata.get("ss_vae_name", None),
                        vae_hash = metadata.get("ss_vae_hash", None),
                        new_vae_hash = metadata.get("ss_new_vae_hash", None),
                        training_comment = to_str(metadata.get("ss_training_comment", None)),
                        bucket_info = to_json(metadata.get("ss_bucket_info", None)),
                        sd_scripts_commit_hash = metadata.get("ss_sd_scripts_commit_hash", None),
                        noise_offset = to_float(metadata.get("ss_noise_offset", None)),
                        optimizer = metadata.get("ss_optimizer", None),
                        max_grad_norm = to_float(metadata.get("ss_max_grad_norm", None)),
                        caption_dropout_rate = to_float(metadata.get("ss_caption_dropout_rate", None)),
                        caption_dropout_every_n_epochs = to_int(metadata.get("ss_caption_dropout_every_n_epochs", None)),
                        caption_tag_dropout_rate = to_float(metadata.get("ss_caption_tag_dropout_rate", None)),
                        face_crop_aug_range = metadata.get("ss_face_crop_aug_range", None),
                        prior_loss_weight = to_float(metadata.get("ss_prior_loss_weight", None)),
                        min_snr_gamma = to_float(metadata.get("ss_min_snr_gamma", None)),
                        scale_weight_norms = to_float(metadata.get("ss_scale_weight_norms", None)),
                    )
                    session.add(lora_model)
                    await session.flush()

                    # TODO dedup
                    image_paths = find_preview_images(os.path.splitext(f)[0])
                    for image_path in image_paths:
                        preview_image = PreviewImage(filepath=image_path, is_autogenerated=False, model_id=lora_model.id)
                        session.add(preview_image)

                await session.commit()
