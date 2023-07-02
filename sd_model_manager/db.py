import os
import os.path
import io
import sys
import glob
import tqdm
import asyncio
import simplejson
from PIL import Image
from datetime import datetime
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

from sd_model_manager.utils.common import PATH
from sd_model_manager.utils import safetensors_hack
from sd_model_manager.models.sd_models import Base, LoRAModel


DATABASE_NAME = os.getenv("DATABASE_NAME", "lora_db")


def to_bool(s):
    if s is None or s == "None":
        return None
    return bool(s)


def to_int(s):
    if s is None or s == "None":
        return None
    return int(s)


def to_float(s):
    if s is None or s == "None":
        return None
    return float(s)


def to_datetime(s):
    if s is None or s == "None":
        return None
    return datetime.fromtimestamp(float(s))


def to_json(s):
    if s is None or s == "None":
        return None
    return simplejson.loads(s)


def to_unique_tags(s):
    if s is None or s == "None":
        return None

    tags = set()

    d = simplejson.loads(s)
    for k, v in d.items():
        for t in v:
            tags.add(t)

    return len(tags)


PREVIEW_EXTS = [".preview.png", ".png"]
def get_preview_image(path):
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]

    for ext in PREVIEW_EXTS:
        file = os.path.join(dirname, f"{basename}{ext}")
        if os.path.isfile(file):
            image = Image.open(file)
            with io.BytesIO() as out:
                image.thumbnail((512,512), Image.LANCZOS)
                image = image.convert("RGB")
                image.save(out, "JPEG", quality=70)
                return out.getvalue()

    return None


class DB:
    def __init__(self):
        self.engine = None
        self.Session = None
        pass

    async def init(self):
        path = os.path.join(PATH, DATABASE_NAME)
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{DATABASE_NAME}.db")

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self.AsyncSession = async_sessionmaker(bind=self.engine)


    async def scan(self, paths):
        for p in paths:
            if not os.path.isdir(p):
                raise RuntimeError(f"Invalid path: {p}")

        print("Building model database...")

        async with self.AsyncSession() as session:
            query = delete(LoRAModel)
            await session.execute(query)
            await session.commit()

            for path in paths:
                files = list(glob.iglob(f"{path}/**/*.safetensors", recursive=True))

                for f in tqdm.tqdm(files):
                    try:
                        metadata = safetensors_hack.read_metadata(f)
                    except:
                        continue
                    # if "ss_lr_scheduler" not in metadata:
                    #     continue
                    lora_model = LoRAModel(
                        root_path=path,
                        filepath=os.path.relpath(f, path),
                        # preview_image=get_preview_image(f),
                        display_name=metadata.get("ssmd_display_name", None),
                        author=metadata.get("ssmd_author", None),
                        source=metadata.get("ssmd_source", None),
                        keywords=metadata.get("ssmd_keywords", None),
                        description=metadata.get("ssmd_description", None),
                        rating=to_int(metadata.get("ssmd_rating", None)),
                        tags=metadata.get("ssmd_tags", None),
                        model_hash=metadata.get("sshs_model_hash", None),
                        legacy_hash=metadata.get("sshs_legacy_hash", None),
                        session_id=to_int(metadata.get("ss_session_id", None)),
                        training_started_at=to_datetime(metadata.get("ss_training_started_at", None)),
                        output_name=metadata.get("ss_output_name", None),
                        learning_rate=to_float(metadata.get("ss_learning_rate", None)),
                        text_encoder_lr=to_float(metadata.get("ss_text_encoder_lr", None)),
                        unet_lr=to_float(metadata.get("ss_unet_lr", None)),
                        num_train_images=to_int(metadata.get("ss_num_train_images", None)),
                        num_reg_images=to_int(metadata.get("ss_num_reg_images", None)),
                        num_batches_per_epoch=to_int(metadata.get("ss_num_batches_per_epoch", None)),
                        num_epochs=to_int(metadata.get("ss_num_epochs", None)),
                        epoch=to_int(metadata.get("ss_epoch", None)),
                        batch_size_per_device=to_int(metadata.get("ss_batch_size_per_device", None)),
                        total_batch_size=to_int(metadata.get("ss_total_batch_size", None)),
                        gradient_checkpointing=to_bool(metadata.get("ss_gradient_checkpointing", None)),
                        gradient_accumulation_steps=to_int(metadata.get("ss_gradient_accumulation_steps", None)),
                        max_train_steps=to_int(metadata.get("ss_max_train_steps", None)),
                        lr_warmup_steps=to_int(metadata.get("ss_lr_warmup_steps", None)),
                        lr_scheduler=metadata.get("ss_lr_scheduler", None),
                        network_module=metadata.get("ss_network_module", None),
                        network_dim=to_int(metadata.get("ss_network_dim", None)),
                        network_alpha=to_float(metadata.get("ss_network_alpha", None)),
                        mixed_precision=to_bool(metadata.get("ss_mixed_precision", None)),
                        full_fp16=to_bool(metadata.get("ss_full_fp16", None)),
                        v2=to_bool(metadata.get("ss_v2", None)),
                        resolution=metadata.get("ss_resolution", None),
                        clip_skip=to_int(metadata.get("ss_clip_skip", None)),
                        max_token_length=to_int(metadata.get("ss_max_token_length", None)),
                        color_aug=to_bool(metadata.get("ss_color_aug", None)),
                        flip_aug=to_bool(metadata.get("ss_flip_aug", None)),
                        random_crop=to_bool(metadata.get("ss_random_crop", None)),
                        shuffle_caption=to_bool(metadata.get("ss_shuffle_caption", None)),
                        cache_latents=to_bool(metadata.get("ss_cache_latents", None)),
                        enable_bucket=to_bool(metadata.get("ss_enable_bucket", None)),
                        min_bucket_reso=to_int(metadata.get("ss_min_bucket_reso", None)),
                        max_bucket_reso=to_int(metadata.get("ss_max_bucket_reso", None)),
                        seed=to_int(metadata.get("ss_seed", None)),
                        keep_tokens=to_bool(metadata.get("ss_keep_tokens", None)),
                        dataset_dirs=to_json(metadata.get("ss_dataset_dirs", None)),
                        reg_dataset_dirs=to_json(metadata.get("ss_reg_dataset_dirs", None)),
                        tag_frequency=to_json(metadata.get("ss_tag_frequency", None)),
                        unique_tags=to_unique_tags(metadata.get("ss_tag_frequency", None)),
                        sd_model_name=metadata.get("ss_sd_model_name", None),
                        sd_model_hash=metadata.get("ss_sd_model_hash", None),
                        sd_new_model_hash=metadata.get("ss_sd_new_model_hash", None),
                        sd_vae_name=metadata.get("ss_sd_vae_name", None),
                        sd_vae_hash=metadata.get("ss_sd_vae_hash", None),
                        sd_new_vae_hash=metadata.get("ss_sd_new_vae_hash", None),
                        vae_name=metadata.get("ss_vae_name", None),
                        training_comment=metadata.get("ss_training_comment", None),
                        bucket_info=to_json(metadata.get("ss_bucket_info", None)),
                        sd_scripts_commit_hash=metadata.get("ss_sd_scripts_commit_hash", None),
                        noise_offset=to_float(metadata.get("ss_noise_offset", None))
                    )
                    session.add(lora_model)

            await session.commit()