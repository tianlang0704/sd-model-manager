import copy
import io
import os
import re
import wx
import time
import shutil
import struct
import urllib
import random
import wxasync
import tempfile
import traceback
import simplejson
from wx.lib.agw import floatspin
from dataclasses import dataclass
from asyncio.locks import Event
from aiopubsub import Key

from sd_model_manager.utils.common import try_load_image
from gui import ids, utils
from gui.api import ComfyAPI, ModelManagerAPI
from gui.utils import PROGRAM_ROOT, combine_tag_freq
from gui.comfy_executor import ComfyExecutor
from gui.image_panel import ImagePanel
from gui.async_utils import AsyncShowDialog, AsyncShowDialogModal, on_close

CHECKPOINTS = [
    "Based64Mix-v3",
    "Based64",
    "AbyssOrangeMix2_nsfw",
    "animefull",
    "animefull",
    "v1-5-",
]
VAES = ["animefull-latest", "kl-f8-anime", "vae-ft-mse"]

OVERRIDE_KEYWORD = "override:"
DEFAULT_POSITIVE = "masterpiece, 1girl, solo"
DEFAULT_NEGATIVE = "(worst quality, low quality:1.2)"
DEFAULT_SEED = -1
DEFAULT_DENOISE = 1.0
DEFAULT_UPSCALE_DENOISE = 0.6
DEFAULT_CFG = 8
DEFAULT_STEPS = 20
DEFAULT_CLIP = -1
DEFAULT_UPSCALE_FACTOR = 1.5
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_SCHEDULER = "normal"
DEFAULT_LORA_BASE = "AOM3.safetensors"

POSITIVE_REGEX = r"positive:(.+)\n*"
NEGATIVE_REGEX = r"negative:(.+)\n*"
SEED_REGEX = r"seed:\s*(\-?\s*\d+)\n*"
DENOISE_REGEX = r"(upscale)?.*denoise:\s*(\d+\s*.?\s*\d*)\n*" #workaround for detecting upscale denoise
UPSCALE_DENOISE_REGEX = r"upscale.*denoise:\s*(\d+\s*.?\s*\d*)\n*"
CFG_REGEX = r"cfg:\s*(\d+)\n*"
STEPS_REGEX = r"steps:\s*(\d+)\n*"
CLIP_REGEX = r"clip:\s*(-?\s*\d+)\n*"
UPSCALE_FACTOR_REGEX = r"upscale.*factor:\s*(\d+\s*.?\s*\d*)\n*"
SAMPLER_REGEX = r"sampler:\s*([\w\.\-_]+)\n*"
SCHEDULER_REGEX = r"scheduler:\s*([\w\.\-_]+)\n*"
LORA_BASE_REGEX = r"lora.*base:\s*([\w\.\-_ ]+)\n*"

MODEL_SD_15_TAG = "sd-1.5"
MODEL_SD_XL_TAG = "sd-xl"
MODEL_SD_TURBO_TAG = "sd-turbo"
MODEL_SD_MERGE_TURBO_TAG = "sd-merge-turbo"
MODEL_SD_LORA_TAG = "sd-lora"

SEED_RANDOM_MAX = 2**32

def load_prompt(name):
    with open(
        os.path.join(PROGRAM_ROOT, "gui/prompts", name), "r", encoding="utf-8"
    ) as f:
        return simplejson.load(f)


@dataclass
class GeneratePreviewsOptions:
    prompt_before: str
    prompt_after: str
    seed: int
    denoise: float
    upscale_denoise: float
    cfg: int
    steps: int
    clip: int
    upscale_factor: float
    sampler: str
    scheduler: str
    lora_base: str


@dataclass
class PreviewPromptData:
    seed: int
    denoise: float
    upscale_denoise: float
    checkpoint: str
    positive: str
    negative: str
    cfg: int
    steps: int
    clip: int
    upscale_factor: float
    sampler: str
    scheduler: str
    lora_base: str
    tags: str

    def to_prompt(self):
        firstTag = ""
        if self.tags:
            firstTag = self.tags.split(",")[0].strip()
        keywordToFunc = {
            MODEL_SD_15_TAG: self.to_prompt_default,
            MODEL_SD_XL_TAG: self.to_prompt_xl,
            MODEL_SD_TURBO_TAG: self.to_prompt_turbo,
            MODEL_SD_MERGE_TURBO_TAG: self.to_prompt_merge_turbo,
            MODEL_SD_LORA_TAG: self.to_prompt_lora,
        }
        for keyword in keywordToFunc:
            if keyword == firstTag:
                return keywordToFunc[keyword]()
        return self.to_prompt_default()
    
    def to_prompt_lora(self):
        prompt = load_prompt("lora.json")
        prompt["3"]["inputs"]["seed"] = self.seed
        prompt["3"]["inputs"]["denoise"] = self.denoise
        prompt["3"]["inputs"]["cfg"] = self.cfg
        prompt["3"]["inputs"]["steps"] = self.steps
        prompt["3"]["inputs"]["sampler_name"] = self.sampler
        prompt["3"]["inputs"]["scheduler"] = self.scheduler
        prompt["230"]["inputs"]["lora_name"] = self.checkpoint
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["7"]["inputs"]["text"] = self.negative
        prompt["231"]["inputs"]["stop_at_clip_layer"] = self.clip
        prompt["4"]["inputs"]["ckpt_name"] = self.lora_base
        return prompt
    
    def to_prompt_merge_turbo(self):
        prompt = load_prompt("merge_turbo.json")
        prompt["5"]["inputs"]["seed"] = self.seed
        prompt["5"]["inputs"]["denoise"] = self.denoise
        prompt["5"]["inputs"]["cfg"] = self.cfg
        prompt["5"]["inputs"]["steps"] = self.steps
        prompt["5"]["inputs"]["sampler_name"] = self.sampler
        prompt["5"]["inputs"]["scheduler"] = self.scheduler
        prompt["2"]["inputs"]["stop_at_clip_layer"] = self.clip
        prompt["1"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["3"]["inputs"]["text"] = self.positive
        prompt["4"]["inputs"]["text"] = self.negative
        return prompt

    def to_prompt_turbo(self):
        prompt = load_prompt("turbo.json")
        prompt["13"]["inputs"]["cfg"] = self.cfg
        prompt["13"]["inputs"]["noise_seed"] = self.seed
        prompt["20"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["7"]["inputs"]["text"] = self.negative
        prompt["22"]["inputs"]["steps"] = self.steps
        prompt["22"]["inputs"]["denoise"] = self.denoise
        return prompt

    def to_prompt_xl(self):
        prompt = load_prompt("xl.json")
        prompt["10"]["inputs"]["noise_seed"] = self.seed
        prompt["10"]["inputs"]["cfg"] = self.cfg
        prompt["11"]["inputs"]["cfg"] = self.cfg
        prompt["10"]["inputs"]["sampler_name"] = self.sampler
        prompt["11"]["inputs"]["sampler_name"] = self.sampler
        prompt["10"]["inputs"]["scheduler"] = self.scheduler
        prompt["11"]["inputs"]["scheduler"] = self.scheduler
        prompt["4"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["12"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["7"]["inputs"]["text"] = self.negative
        prompt["53"]["inputs"]["value"] = self.steps
        prompt["54"]["inputs"]["value"] = int(self.steps * self.denoise)
        return prompt

    def to_prompt_default(self):
        prompt = load_prompt("default.json")
        prompt["3"]["inputs"]["seed"] = self.seed
        prompt["3"]["inputs"]["denoise"] = self.denoise
        prompt["3"]["inputs"]["cfg"] = self.cfg
        prompt["3"]["inputs"]["steps"] = self.steps
        prompt["3"]["inputs"]["sampler_name"] = self.sampler
        prompt["3"]["inputs"]["scheduler"] = self.scheduler
        prompt["4"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["7"]["inputs"]["text"] = self.negative
        prompt["31"]["inputs"]["stop_at_clip_layer"] = self.clip
        return prompt

    def to_hr_prompt(self, image):
        firstTag = ""
        if self.tags:
            firstTag = self.tags.split(",")[0].strip()
        filename = ""
        if isinstance(image, str):
            filename = image
        else:
            filename = image["filename"]
        keywordToFunc = {
            MODEL_SD_15_TAG: self.to_prompt_default_hr,
            MODEL_SD_XL_TAG: self.to_prompt_xl_hr,
            MODEL_SD_TURBO_TAG: self.to_prompt_turbo_hr,
            MODEL_SD_MERGE_TURBO_TAG: self.to_prompt_merge_turbo_hr,
            MODEL_SD_LORA_TAG: self.to_prompt_lora_hr,
        }
        for keyword in keywordToFunc:
            if keyword == firstTag:
                return keywordToFunc[keyword](filename)
        return self.to_prompt_default_hr(filename)

    def to_prompt_default_hr(self, filename):
        prompt = load_prompt("default-hr.json")
        prompt["11"]["inputs"]["seed"] = self.seed
        prompt["11"]["inputs"]["denoise"] = self.upscale_denoise
        prompt["11"]["inputs"]["cfg"] = self.cfg
        prompt["11"]["inputs"]["steps"] = self.steps
        prompt["11"]["inputs"]["sampler_name"] = self.sampler
        prompt["11"]["inputs"]["scheduler"] = self.scheduler
        prompt["16"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["7"]["inputs"]["text"] = self.negative
        prompt["20"]["inputs"]["scale_by"] = self.upscale_factor
        prompt["18"]["inputs"]["image"] = f"{filename} [output]"
        return prompt
    
    def to_prompt_lora_hr(self, filename):
        prompt = load_prompt("lora-hr.json")
        prompt["11"]["inputs"]["seed"] = self.seed
        prompt["11"]["inputs"]["denoise"] = self.upscale_denoise
        prompt["11"]["inputs"]["cfg"] = self.cfg
        prompt["11"]["inputs"]["steps"] = self.steps
        prompt["11"]["inputs"]["sampler_name"] = self.sampler
        prompt["11"]["inputs"]["scheduler"] = self.scheduler
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["7"]["inputs"]["text"] = self.negative
        prompt["16"]["inputs"]["ckpt_name"] = self.lora_base
        prompt["22"]["inputs"]["lora_name"] = self.checkpoint
        prompt["20"]["inputs"]["scale_by"] = self.upscale_factor
        prompt["18"]["inputs"]["image"] = f"{filename} [output]"
        return prompt

    def to_prompt_xl_hr(self, filename):
        prompt = load_prompt("xl-hr.json")
        prompt["27"]["inputs"]["seed"] = self.seed
        prompt["27"]["inputs"]["start_at_step"] = int(self.steps * (1 - self.upscale_denoise))
        prompt["27"]["inputs"]["cfg"] = self.cfg
        prompt["27"]["inputs"]["steps"] = self.steps
        prompt["27"]["inputs"]["sampler_name"] = self.sampler
        prompt["27"]["inputs"]["scheduler"] = self.scheduler
        prompt["12"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["15"]["inputs"]["text"] = self.positive
        prompt["16"]["inputs"]["text"] = self.negative
        prompt["52"]["inputs"]["scale_by"] = self.upscale_factor
        prompt["24"]["inputs"]["image"] = f"{filename} [output]"
        return prompt
    
    def to_prompt_turbo_hr(self, filename):
        prompt = load_prompt("turbo-hr.json")
        prompt["13"]["inputs"]["noise_seed"] = self.seed
        prompt["13"]["inputs"]["cfg"] = self.cfg
        prompt["22"]["inputs"]["denoise"] = self.upscale_denoise
        prompt["22"]["inputs"]["steps"] = self.steps
        prompt["14"]["inputs"]["sampler_name"] = self.sampler
        prompt["20"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["7"]["inputs"]["text"] = self.negative
        prompt["42"]["inputs"]["scale_by"] = self.upscale_factor
        prompt["38"]["inputs"]["image"] = f"{filename} [output]"
        return prompt
    
    def to_prompt_merge_turbo_hr(self, filename):
        prompt = load_prompt("merge_turbo-hr.json")
        prompt["5"]["inputs"]["seed"] = self.seed
        prompt["5"]["inputs"]["denoise"] = self.upscale_denoise
        prompt["5"]["inputs"]["cfg"] = self.cfg
        prompt["5"]["inputs"]["steps"] = self.steps
        prompt["5"]["inputs"]["sampler_name"] = self.sampler
        prompt["5"]["inputs"]["scheduler"] = self.scheduler
        prompt["2"]["inputs"]["stop_at_clip_layer"] = self.clip
        prompt["1"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["3"]["inputs"]["text"] = self.positive
        prompt["4"]["inputs"]["text"] = self.negative
        prompt["13"]["inputs"]["scale_by"] = self.upscale_factor
        prompt["10"]["inputs"]["image"] = f"{filename} [output]"
        return prompt
class CancelException(Exception):
    pass


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

    return images

KEY_POSITIVE = "processed_positive"
KEY_NEGATIVE = "processed_negative"

#file drop target
class FileDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, filenames):
        return self.window.OnDropFiles(x, y, filenames)

class PreviewGeneratorDialog(wx.Dialog):
    def __init__(self, parent, app, items, duplicate_op):
        main_name = items[0]["filename"] if items and len(items) > 0 else ""
        suffix = f" and other {len(items) - 1} model(s)" if len(items) > 1 else ""
        super(PreviewGeneratorDialog, self).__init__(
            parent, -1, f"Preview Generator: {main_name}{suffix}", size=app.FromDIP(700, 500)
        )
        self.app = app
        self.comfy_api = ComfyAPI()
        self.duplicate_op = duplicate_op  # "replace", "append"
        self.items = items
        self.preview_options = self.item_to_preview_options(items)
        self.result = None
        self.last_data = None
        self.last_output = None
        self.executing_node_id = None
        self.upscaled = False
        self.last_seed = -1
        self.last_upscale_seed = -1
        self.node_text = ""

        utils.set_icons(self)
        self.autogen = False

        self.SetDropTarget(FileDropTarget(self))

        # Parameter controls
        self.text_prompt_before = wx.TextCtrl(
            self,
            id=wx.ID_ANY,
            value=self.preview_options.prompt_before,
            size=self.Parent.FromDIP(wx.Size(250, 100)),
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
        )
        self.text_prompt_after = wx.TextCtrl(
            self,
            id=wx.ID_ANY,
            value=self.preview_options.prompt_after,
            size=self.Parent.FromDIP(wx.Size(250, 100)),
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
        )

        # Status controls/buttons
        self.status_text = wx.StaticText(self, -1, "Ready")
        self.models_text = wx.StaticText(self, wx.ID_ANY, label=f"Selected models: {len(self.items)}")
        self.gauge = wx.Gauge(self, -1, 100, size=app.FromDIP(800, 32))
        self.image_panel = ImagePanel(self, style=wx.SUNKEN_BORDER, size=app.FromDIP(512, 512))
        main_item = items[0] if items and len(items) > 0 else None
        preview_image = None
        if main_item is not None:
            filepath = main_item["filepath"]
            basepath = os.path.splitext(filepath)[0]
            images = find_preview_images(basepath)
            if len(images) > 0:
                preview_image = images[0]
                self.set_preview_image(preview_image)

        self.button_save_notes = wx.Button(self, wx.ID_SAVE, "Save Notes")
        self.button_regenerate = wx.Button(self, wx.ID_HELP, "Generate")
        self.button_upscale = wx.Button(self, wx.ID_APPLY, "Upscale")
        if not preview_image:
            self.button_upscale.Disable()
        self.button_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        self.button_ok = wx.Button(self, wx.ID_OK, "OK")
        self.button_ok.Disable()

        self.Bind(wx.EVT_BUTTON, self.OnRegenerate, id=wx.ID_HELP)
        self.Bind(wx.EVT_BUTTON, self.OnUpscale, id=wx.ID_APPLY)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, id=wx.ID_CANCEL)
        wxasync.AsyncBind(wx.EVT_BUTTON, self.OnSave, self.button_save_notes, id=wx.ID_SAVE)
        wxasync.AsyncBind(wx.EVT_BUTTON, self.OnOK, self.button_ok, id=wx.ID_OK)
        wxasync.AsyncBind(wx.EVT_CLOSE, self.OnClose, self)

        sizerB = wx.BoxSizer(wx.HORIZONTAL)
        sizerB.Add(self.button_ok, 0, wx.ALL, 5)
        sizerB.Add(self.button_cancel, 0, wx.ALL, 5)
        sizerB.Add(self.button_upscale, 0, wx.ALL, 5)
        sizerB.Add(self.button_regenerate, 0, wx.ALL, 5)
        sizerB.Add(self.button_save_notes, 0, wx.ALL, 5)

        sizerLeft = wx.BoxSizer(wx.VERTICAL)
        sizerLeft.Add(self.image_panel)
        sizerLeft.AddSpacer(8)
        sizerLeft.Add(self.status_text)

        sizerRightAfter = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfter.Add(
            wx.StaticText(self, wx.ID_ANY, label="Seed"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        self.spinner_seed = wx.TextCtrl(
            self, 
            wx.ID_ANY, 
            value = str(self.preview_options.seed), 
            size = self.Parent.FromDIP(wx.Size(90, 25))
        )
        sizerRightAfter.Add(self.spinner_seed, proportion=1, flag=wx.ALL, border=5)

        self.button_random_seed = wx.Button(self, wx.ID_CDROM, "Random", size = self.Parent.FromDIP(wx.Size(70, 25)))
        self.Bind(wx.EVT_BUTTON, lambda e:self.spinner_seed.SetValue(str(random.randint(0, SEED_RANDOM_MAX))), id=wx.ID_CDROM)
        sizerRightAfter.Add(self.button_random_seed, proportion=0, flag=wx.ALL, border=5)

        self.button_last_seed = wx.Button(self, wx.ID_ADD, "Last", size = self.Parent.FromDIP(wx.Size(50, 25)))
        def on_last_seed(e):
            value = self.last_upscale_seed if self.upscaled else self.last_seed
            if value == -1:
                return
            self.spinner_seed.SetValue(str(value))
        self.Bind(wx.EVT_BUTTON, on_last_seed, id=wx.ID_ADD)
        sizerRightAfter.Add(self.button_last_seed, proportion=0, flag=wx.ALL, border=5)

        sizerRightAfter2 = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfter2.Add(
            wx.StaticText(self, wx.ID_ANY, label="CFG"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        self.spinner_cfg = wx.SpinCtrl(
            self,
            id=wx.ID_ANY,
            value="",
            style=wx.SP_ARROW_KEYS,
            min=1,
            max=30,
            initial=self.preview_options.cfg,
            # size=self.Parent.FromDIP(wx.Size(150, 25)),
        )
        sizerRightAfter2.Add(self.spinner_cfg, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter2.Add(
            wx.StaticText(self, wx.ID_ANY, label="Steps"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        self.spinner_steps = wx.SpinCtrl(
            self,
            id=wx.ID_ANY,
            value="",
            style=wx.SP_ARROW_KEYS,
            min=1,
            max=50,
            initial=self.preview_options.steps,
            # size=self.Parent.FromDIP(wx.Size(150, 25)),
        )
        sizerRightAfter2.Add(self.spinner_steps, proportion=1, flag=wx.ALL, border=5)

        clip_label = sizerRightAfter2.Add(
            wx.StaticText(self, wx.ID_ANY, label="Clip"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        self.spinner_clip = wx.SpinCtrl(
            self,
            id=wx.ID_ANY,
            value="",
            style=wx.SP_ARROW_KEYS,
            min=-12,
            max=-1,
            initial=self.preview_options.clip,
            # size=self.Parent.FromDIP(wx.Size(140, 25)),
        )
        sizerRightAfter2.Add(self.spinner_clip, proportion=1, flag=wx.ALL, border=5)
        firstTagList = self.get_first_tag_list()
        show_clip = False
        if sum(1 for tag in firstTagList if tag != MODEL_SD_TURBO_TAG and tag != MODEL_SD_XL_TAG) > 0:
            show_clip = True
        clip_label.Show(show_clip)
        self.spinner_clip.Show(show_clip)

        sizerRightAfter3 = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfter3.Add(
            wx.StaticText(self, wx.ID_ANY, label="Denoise"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        self.spinner_denoise = floatspin.FloatSpin(
            self,
            id=wx.ID_ANY,
            min_val=0,
            max_val=1,
            increment=0.01,
            value=self.preview_options.denoise,
            agwStyle=floatspin.FS_LEFT,
            # size=self.Parent.FromDIP(wx.Size(140, 25)),
        )
        self.spinner_denoise.SetFormat("%f")
        self.spinner_denoise.SetDigits(2)
        sizerRightAfter3.Add(self.spinner_denoise, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter3.Add(
            wx.StaticText(self, wx.ID_ANY, label="Upscale Denoise"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        self.spinner_upscale_denoise = floatspin.FloatSpin(
            self,
            id=wx.ID_ANY,
            min_val=0,
            max_val=1,
            increment=0.01,
            value=self.preview_options.upscale_denoise,
            agwStyle=floatspin.FS_LEFT,
            # size=self.Parent.FromDIP(wx.Size(140, 25)),
        )
        self.spinner_upscale_denoise.SetFormat("%f")
        self.spinner_upscale_denoise.SetDigits(2)
        sizerRightAfter3.Add(self.spinner_upscale_denoise, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter4 = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfter4.Add(
            wx.StaticText(self, wx.ID_ANY, label="Upscale Factor"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        self.spinner_upscale_factor = floatspin.FloatSpin(
            self,
            id=wx.ID_ANY,
            min_val=0,
            max_val=10,
            increment=0.1,
            value=self.preview_options.upscale_factor,
            agwStyle=floatspin.FS_LEFT,
            # size=self.Parent.FromDIP(wx.Size(140, 25)),
        )
        self.spinner_upscale_factor.SetFormat("%f")
        self.spinner_upscale_factor.SetDigits(2)
        sizerRightAfter4.Add(self.spinner_upscale_factor, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter5 = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfter5.Add(
            wx.StaticText(self, wx.ID_ANY, label="Sampler"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        choices = ["euler_ancestral", "dpmpp_2m"]
        self.sampler = wx.ComboBox(self, id=wx.ID_ANY, value=self.preview_options.sampler, choices=choices)
        sizerRightAfter5.Add(self.sampler, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter6 = wx.BoxSizer(wx.HORIZONTAL)
        scheduler_label = sizerRightAfter6.Add(
            wx.StaticText(self, wx.ID_ANY, label="Scheduler"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        choices = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
        self.scheduler = wx.ComboBox(self, id=wx.ID_ANY, value=self.preview_options.scheduler, choices=choices)
        sizerRightAfter6.Add(self.scheduler, proportion=1, flag=wx.ALL, border=5)
        firstTagList = self.get_first_tag_list()
        show_scheduler = False
        if sum(1 for tag in firstTagList if tag != MODEL_SD_TURBO_TAG) > 0:
            show_scheduler = True
        scheduler_label.Show(show_scheduler)
        self.scheduler.Show(show_scheduler)

        sizerRightAfter7 = wx.BoxSizer(wx.HORIZONTAL)
        lora_base_label = sizerRightAfter7.Add(
            wx.StaticText(self, wx.ID_ANY, label="Lora Base"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        choices = ["PerfectWorld.safetensors", "astranime_V6.safetensors", "AOM3.safetensors", "Anything-V3.0-pruned-fp16.ckpt", "v1-5-pruned.ckpt"]
        self.lora_base = wx.ComboBox(
            self, 
            id=wx.ID_ANY, 
            value=self.preview_options.lora_base, 
            choices=choices,
            size = self.Parent.FromDIP(wx.Size(150, 25)),
        )
        sizerRightAfter7.Add(self.lora_base, proportion=1, flag=wx.ALL, border=5)
        firstTagList = self.get_first_tag_list()
        show_lora_base = False
        if MODEL_SD_LORA_TAG in firstTagList:
            show_lora_base = True
        lora_base_label.Show(show_lora_base)
        self.lora_base.Show(show_lora_base)


        sizerRight = wx.StaticBoxSizer(wx.VERTICAL, self, label="Parameters")
        sizerRight.Add(wx.StaticText(self, wx.ID_ANY, label="Positive"))
        sizerRight.Add(self.text_prompt_before, proportion=3, flag=wx.ALL | wx.EXPAND)
        sizerRight.Add(wx.StaticText(self, wx.ID_ANY, label="Negative"))
        sizerRight.Add(self.text_prompt_after, proportion=3, flag=wx.ALL | wx.EXPAND)
        sizerRight.Add(sizerRightAfter, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfter2, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfter3, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfter4, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfter5, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfter6, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfter7, proportion=1, flag=wx.ALL)
        sizerRight.Add(
            self.models_text,
            proportion=0,
            border=5,
            flag=wx.ALL,
        )

        sizerMain = wx.FlexGridSizer(1, 2, 10, 10)
        sizerMain.AddMany([(sizerLeft, 1), (sizerRight, 1, wx.EXPAND)])

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sizerMain, 1, wx.EXPAND)
        sizer.AddSpacer(8)
        sizer.Add(self.gauge, 0, wx.LEFT | wx.RIGHT)
        sizer.AddSpacer(8)
        sizer.Add(sizerB, 0, wx.LEFT | wx.RIGHT)

        wrapper = wx.BoxSizer(wx.VERTICAL)
        wrapper.Add(sizer, 1, wx.EXPAND | wx.ALL, 10)

        self.SetSizerAndFit(wrapper)

    def set_preview_image(self, image):
        self.image_panel.LoadImageFrompath(image)
        self.last_output = image

    async def save_preview_image(self, item, result):
        self.app.SetStatusText("Saving preview...")
        self.status_text.SetLabel("Saving preview...")

        image_data = self.comfy_api.get_image(result["filename"], result["subfolder"], result["type"])
        filepath = item["filepath"]
        basepath = os.path.splitext(filepath)[0]

        path = basepath + ".png"
        if os.path.exists(path):
            i = 0
            found = None
            while found is None:
                if i == 0:
                    p = f"{basepath}.preview.png"
                else:
                    p = f"{basepath}.preview.{i}.png"
                if not os.path.exists(p):
                    found = p
                i += 1

            if self.duplicate_op == "append":
                path = found
            else:  # replace
                shutil.move(path, found)

        # Write the image
        with open(path, "wb") as f:
            f.write(image_data)

        # Update the metadata
        new_images = find_preview_images(basepath)
        new_images = [
            {"filepath": path, "is_autogenerated": True} for path in new_images  # TODO
        ]

        changes = {"preview_images": new_images}
        print(changes)
        result = await self.app.api.update_lora(item["id"], changes)
        print(result)
        item["preview_images"] = new_images

        try_load_image.cache_clear()
        await self.app.frame.results_panel.refresh_one_item(item)

        self.app.SetStatusText(f"Saved preview to {path}")
    
    def OnDropFiles(self, x, y, filenames):
        if len(filenames) <= 0:
            return
        filepath = filenames[0]
        self.set_preview_image(filepath)
        return True

    async def OnOK(self, evt):
        self.button_regenerate.Disable()
        self.button_upscale.Disable()
        self.button_cancel.Disable()
        self.button_ok.Disable()
        self.text_prompt_before.Disable()
        self.text_prompt_after.Disable()
        self.spinner_seed.Disable()
        self.spinner_denoise.Disable()

        if self.result is not None:
            await self.save_preview_image(self.items[0], self.result)

        if len(self.items) > 1:
            self.status_text.SetLabel(f"Generating previews...")

            upscaled = self.upscaled
            self.autogen = True
            for i, item in enumerate(self.items[1:]):
                filename = os.path.basename(item["filepath"])
                self.status_text.SetLabel(f"Generating preview for {filename}")
                self.models_text.SetLabel(f"Progress: {i}/{len(self.items)-1}")
                self.spinner_seed.SetValue(str(self.last_seed))
                e = Event()
                self.start_prompt(item, e=e)
                await e.wait()
                if upscaled:
                    self.status_text.SetLabel("Starting upscale...")
                    self.spinner_seed.SetValue(str(self.last_upscale_seed))
                    e = Event()
                    thread = self.upscale_prompt(item, e=e)
                    await e.wait()
                if self.result is not None:
                    await self.save_preview_image(item, self.result)
        else:
            self.status_text.SetLabel("Done!")

        self.AsyncEndModal(wx.ID_OK)

    def OnCancel(self, evt):
        self.AsyncEndModal(wx.ID_CANCEL)

    async def OnClose(self, evt):
        if self.autogen:
            return
        await on_close(self, evt)

    async def OnSave(self, evt):
        main_item = self.items[0]
        notes = main_item.get("notes") or ""
        insert_str = ""

        positive = self.text_prompt_before.GetValue()
        if re.search(POSITIVE_REGEX, notes, re.I):
            notes = re.sub(POSITIVE_REGEX, f"positive: {positive}\n", notes, flags = re.I)
        else:
            insert_str += f"positive: {positive}\n"

        negative = self.text_prompt_after.GetValue()
        if re.search(NEGATIVE_REGEX, notes, re.I):
            notes = re.sub(NEGATIVE_REGEX, f"negative: {negative}\n", notes, flags = re.I)
        else:
            insert_str += f"negative: {negative}\n"

        seed = self.spinner_seed.GetValue()
        if re.search(SEED_REGEX, notes, re.I):
            notes = re.sub(SEED_REGEX, f"seed: {seed}\n", notes, flags = re.I)
        else:
            insert_str += f"seed: {seed}\n"
        
        denoise = self.spinner_denoise.GetValue()
        re_result = re.finditer(DENOISE_REGEX, notes, re.I)
        count = sum(1 for match in re_result if not match.group(1))
        if count:
            notes = re.sub(DENOISE_REGEX, lambda match: match.group(0) if match.group(1) else f"denoise: {denoise}\n", notes, flags = re.I)
        else:
            insert_str += f"denoise: {denoise}\n"

        upscale_denoise = self.spinner_upscale_denoise.GetValue()
        if re.search(UPSCALE_DENOISE_REGEX, notes, re.I):
            notes = re.sub(UPSCALE_DENOISE_REGEX, f"upscale denoise: {upscale_denoise}\n", notes, flags = re.I)
        else:
            insert_str += f"upscale denoise: {upscale_denoise}\n"

        cfg = self.spinner_cfg.GetValue()
        if re.search(CFG_REGEX, notes, re.I):
            notes = re.sub(CFG_REGEX, f"cfg: {cfg}\n", notes, flags = re.I)
        else:
            insert_str += f"cfg: {cfg}\n"

        steps = self.spinner_steps.GetValue()
        if re.search(STEPS_REGEX, notes, re.I):
            notes = re.sub(STEPS_REGEX, f"steps: {steps}\n", notes, flags = re.I)
        else:
            insert_str += f"steps: {steps}\n"

        # turbo and xl no clip
        firstTag = self.get_main_first_tag()
        if firstTag != MODEL_SD_TURBO_TAG and firstTag != MODEL_SD_XL_TAG:
            clip = self.spinner_clip.GetValue()
            if re.search(CLIP_REGEX, notes, re.I):
                notes = re.sub(CLIP_REGEX, f"clip: {clip}\n", notes, flags = re.I)
            else:
                insert_str += f"clip: {clip}\n"

        upscale_factor = self.spinner_upscale_factor.GetValue()
        if re.search(UPSCALE_FACTOR_REGEX, notes, re.I):
            notes = re.sub(UPSCALE_FACTOR_REGEX, f"upscale factor: {upscale_factor}\n", notes, flags = re.I)
        else:
            insert_str += f"upscale factor: {upscale_factor}\n"

        sampler = self.sampler.GetValue()
        if re.search(SAMPLER_REGEX, notes, re.I):
            notes = re.sub(SAMPLER_REGEX, f"sampler: {sampler}\n", notes, flags = re.I)
        else:
            insert_str += f"sampler: {sampler}\n"
        
        # turbo no scheduler
        firstTag = self.get_main_first_tag()
        if firstTag != MODEL_SD_TURBO_TAG:
            scheduler = self.scheduler.GetValue()
            if re.search(SCHEDULER_REGEX, notes, re.I):
                notes = re.sub(SCHEDULER_REGEX, f"scheduler: {scheduler}\n", notes, flags = re.I)
            else:
                insert_str += f"scheduler: {scheduler}\n"

        # only lora need base model
        firstTag = self.get_main_first_tag()
        if firstTag == MODEL_SD_LORA_TAG:
            lora_base = self.lora_base.GetValue()
            if re.search(LORA_BASE_REGEX, notes, re.I):
                notes = re.sub(LORA_BASE_REGEX, f"lora base: {lora_base}\n", notes, flags = re.I)
            else:
                insert_str += f"lora base: {lora_base}\n"

        results_panel = self.app.frame.results_panel
        selection = results_panel.get_selection()
        notes = insert_str + notes
        main_item["notes"] = notes
        result = await self.app.api.update_lora(main_item["id"], {"notes": notes})
        print(result)
        await results_panel.search(results_panel.searchBox.GetValue(), True)
        results_panel.restore_selection(selection)
        

    def OnRegenerate(self, evt):
        self.status_text.SetLabel("Starting...")
        self.start_prompt(self.items[0])

    def OnUpscale(self, evt):
        self.status_text.SetLabel("Starting upscale...")
        self.upscale_prompt(self.items[0])

    def upscale_prompt(self, item, e=None):
        return utils.start_async_thread(self.run_upscale_prompt, item, e)

    def start_prompt(self, item, e=None):
        return utils.start_async_thread(self.run_prompt, item, e)

    async def run_prompt(self, item, e):
        try:
            self.do_execute(item)
        except CancelException:
            pass
        except Exception as ex:
            print(traceback.format_exc())
            if e is None:
                await self.on_fail(ex)
        if e is not None:
            e.set()

    async def run_upscale_prompt(self, item, e):
        try:
            self.do_upscale(item)
        except CancelException:
            pass
        except Exception as ex:
            print(traceback.format_exc())
            if e is None:
                await self.on_fail(ex)
        if e is not None:
            e.set()

    def get_tags(self, item, count=None):
        tags = []
        tag_freq = item.get("tag_frequency")
        if tag_freq is not None:
            totals = utils.combine_tag_freq(tag_freq)
            sort = list(sorted(totals.items(), key=lambda p: p[1], reverse=True))
            tags = [p[0] for p in sort]
            if count is not None:
                tags = tags[:count]
        return tags

    def get_prompt_options(self):
        return GeneratePreviewsOptions(
            self.text_prompt_before.GetValue(),
            self.text_prompt_after.GetValue(),
            int(self.spinner_seed.GetValue()),
            float(self.spinner_denoise.GetValue()),
            float(self.spinner_upscale_denoise.GetValue()),
            int(self.spinner_cfg.GetValue()),
            int(self.spinner_steps.GetValue()),
            int(self.spinner_clip.GetValue()),
            float(self.spinner_upscale_factor.GetValue()),
            self.sampler.GetValue(),
            self.scheduler.GetValue(),
            self.lora_base.GetValue(),
        )

    def assemble_prompt_data(self, item):
        checkpoint = item["filename"]
        inputOptions = self.get_prompt_options()
        itemOptions = self.item_to_preview_options(item)
        positive = inputOptions.prompt_before if self.preview_options.prompt_before != inputOptions.prompt_before else itemOptions.prompt_before
        negative = inputOptions.prompt_after if self.preview_options.prompt_after != inputOptions.prompt_after else itemOptions.prompt_after
        seed = inputOptions.seed if inputOptions.seed != self.preview_options.seed else itemOptions.seed
        if seed == -1:
            seed = random.randint(0, SEED_RANDOM_MAX)
        denoise = inputOptions.denoise if inputOptions.denoise != self.preview_options.denoise else itemOptions.denoise
        upscale_denoise = inputOptions.upscale_denoise if inputOptions.upscale_denoise != self.preview_options.upscale_denoise else itemOptions.upscale_denoise
        cfg = inputOptions.cfg if inputOptions.cfg != self.preview_options.cfg else itemOptions.cfg
        steps = inputOptions.steps if inputOptions.steps != self.preview_options.steps else itemOptions.steps
        clip = inputOptions.clip if inputOptions.clip != self.preview_options.clip else itemOptions.clip
        upscale_factor = inputOptions.upscale_factor if inputOptions.upscale_factor != self.preview_options.upscale_factor else itemOptions.upscale_factor
        sampler = inputOptions.sampler if inputOptions.sampler != self.preview_options.sampler else itemOptions.sampler
        scheduler = inputOptions.scheduler if inputOptions.scheduler != self.preview_options.scheduler else itemOptions.scheduler
        loraBase = inputOptions.lora_base if inputOptions.lora_base != self.preview_options.lora_base else itemOptions.lora_base
        tags = item["tags"]
        data = PreviewPromptData(
            seed, 
            denoise, 
            upscale_denoise,
            checkpoint, 
            positive, 
            negative, 
            cfg,
            steps,
            clip,
            upscale_factor,
            sampler,
            scheduler,
            loraBase,
            tags
        )
        print(f"Seed: {seed}")
        return data

    def enqueue_prompt_and_wait(self, executor, prompt):
        queue_result = executor.enqueue(prompt)
        prompt_id = queue_result["prompt_id"]

        while True:
            wx.Yield()
            if not self:
                # self was destroyed
                return

            msg = executor.get_status()
            if msg:
                if msg["type"] == "json":
                    status = msg["data"]
                    print(status)
                    ty = status["type"]
                    if ty == "executing":
                        data = status["data"]
                        if data["node"] is not None:
                            self.on_msg_executing(prompt, data)
                        else:
                            if data["prompt_id"] == prompt_id:
                                # Execution is done
                                break
                    elif ty == "progress":
                        self.on_msg_progress(status)
                else:
                    self.image_panel.LoadImageFromBytes(msg["data"])

        return prompt_id

    def before_execute(self):
        self.result = None
        self.last_data = None
        self.upscaled = False
        # self.image_panel.Clear()
        self.button_regenerate.Disable()
        self.button_upscale.Disable()
        self.button_ok.Disable()

    def after_execute(self):
        if not self.autogen:
            self.button_regenerate.Enable()
            self.button_ok.Enable()
            self.button_upscale.Enable()
            seed = self.last_upscale_seed if self.upscaled else self.last_seed
            self.status_text.SetLabel(f"Finished. (seed: {seed})")

    def get_output_image(self, prompt_id):
        images, files = self.comfy_api.get_images(prompt_id)
        if not images:
            return None, None
        image_datas = []
        image_files = None
        for node_id in images:
            image_datas += images[node_id]
            image_files = files[node_id]
        if not image_datas:
            return None, None
        return image_datas[0], image_files[0]

    def do_execute(self, item):
        self.before_execute()
        self.last_output = None

        with ComfyExecutor() as executor:
            data = self.assemble_prompt_data(item)
            prompt = data.to_prompt()
            prompt_id = self.enqueue_prompt_and_wait(executor, prompt)

        image_data, image_location = self.get_output_image(prompt_id)
        if image_data:
            self.image_panel.LoadImageFromBytes(image_data)

        self.last_data = data
        self.last_output = image_location
        self.result = image_location
        self.last_seed = data.seed

        self.after_execute()

    def do_upscale(self, item):
        self.before_execute()

        with ComfyExecutor() as executor:
            data = self.assemble_prompt_data(item)
            prompt = data.to_hr_prompt(self.last_output)
            prompt_id = self.enqueue_prompt_and_wait(executor, prompt)

        image_data, image_location = self.get_output_image(prompt_id)
        if image_data:
            self.image_panel.LoadImageFromBytes(image_data)

        self.last_data = data
        self.result = image_location
        self.last_upscale_seed = data.seed
        self.upscaled = True

        self.after_execute()

    def on_msg_executing(self, prompt, data):
        node_id = data["node"]
        self.executing_node_id = node_id
        class_type = prompt[node_id]["class_type"]
        self.node_text = f"Node: {class_type}"
        self.status_text.SetLabel(self.node_text)
        self.gauge.SetRange(100)
        self.gauge.SetValue(0)

    def on_msg_progress(self, status):
        value = status["data"]["value"]
        max = status["data"]["max"]
        self.status_text.SetLabel(f"{self.node_text} ({value}/{max})")
        self.gauge.SetRange(max)
        self.gauge.SetValue(value)

    async def on_fail(self, ex):
        dialog = wx.MessageDialog(
            self,
            f"Failed to generate previews:\n{ex}",
            "Generation Failed",
            wx.OK | wx.ICON_ERROR,
        )
        await wxasync.AsyncShowDialogModal(dialog)
        # dialog.Destroy()
        self.AsyncEndModal(wx.ID_CANCEL)

    def item_to_preview_options(self, itemsOrItems):
        item = itemsOrItems
        if isinstance(itemsOrItems, list):
            item = itemsOrItems[0]
        notes = item.get("notes") or ""
        # build positive prompt
        re_notes_positive = re.search(POSITIVE_REGEX, notes, re.I)
        notes_positive = re_notes_positive.group(1).strip() if re_notes_positive else DEFAULT_POSITIVE
        positive = notes_positive.strip()
        posKey = item["keywords"]
        if posKey:
            if posKey.startswith(OVERRIDE_KEYWORD):
                positive = ""
                posKey = posKey.replace(OVERRIDE_KEYWORD, "")
            else:
                keyToInsert = []
                posKeyList = posKey.split(",")
                for key in posKeyList:
                    if key not in positive:
                        keyToInsert.append(key.strip())
                if len(keyToInsert) > 0:
                    keyStr = ",".join(keyToInsert)
                    posKey = f", {keyStr}"
                else:
                    posKey = ""
            positive += posKey
        # build negative prompt
        re_notes_negative = re.search(NEGATIVE_REGEX, notes, re.I)
        notes_negative = re_notes_negative.group(1).strip() if re_notes_negative else DEFAULT_NEGATIVE
        negative = notes_negative.strip()
        negKey = item["negative_keywords"]
        if negKey:
            if negKey.startswith(OVERRIDE_KEYWORD):
                negative = ""
                negKey = negKey.replace(OVERRIDE_KEYWORD, "")
            else:
                keyToInsert = []
                negKeyList = negKey.split(",")
                for key in negKeyList:
                    if key not in negative:
                        keyToInsert.append(key.strip())
                if len(keyToInsert) > 0:
                    keyStr = ",".join(keyToInsert)
                    negKey = f", {keyStr}"
                else:
                    negKey = ""
            negative += negKey
        # build seed
        re_notes_seed = re.search(SEED_REGEX, notes, re.I)
        seed = int(re_notes_seed.group(1).strip()) if re_notes_seed else DEFAULT_SEED
        # build denoise
        re_result = re.finditer(DENOISE_REGEX, notes, re.I)
        re_notes_denoise = next((match for match in re_result if not match.group(1)), None)
        denoise = float(re_notes_denoise.group(2).strip()) if re_notes_denoise else DEFAULT_DENOISE
        # build upscale denoise
        re_notes_upscale_denoise = re.search(UPSCALE_DENOISE_REGEX, notes, re.I)
        upscale_denoise = float(re_notes_upscale_denoise.group(1).strip()) if re_notes_upscale_denoise else DEFAULT_UPSCALE_DENOISE
        # build cfg
        re_notes_cfg = re.search(CFG_REGEX, notes, re.I)
        cfg = int(re_notes_cfg.group(1).strip()) if re_notes_cfg else DEFAULT_CFG
        # build steps
        re_notes_steps = re.search(STEPS_REGEX, notes, re.I)
        steps = int(re_notes_steps.group(1).strip()) if re_notes_steps else DEFAULT_STEPS
        # build clip
        re_notes_clip = re.search(CLIP_REGEX, notes, re.I)
        clip = int(re_notes_clip.group(1).strip()) if re_notes_clip else DEFAULT_CLIP
        # build upscale factor
        re_notes_upscale_factor = re.search(UPSCALE_FACTOR_REGEX, notes, re.I)
        upscale_factor = float(re_notes_upscale_factor.group(1).strip()) if re_notes_upscale_factor else DEFAULT_UPSCALE_FACTOR
        # build sampler
        re_notes_sampler = re.search(SAMPLER_REGEX, notes, re.I)
        sampler = re_notes_sampler.group(1).strip() if re_notes_sampler else DEFAULT_SAMPLER
        # build scheduler
        re_notes_scheduler = re.search(SCHEDULER_REGEX, notes, re.I)
        scheduler = re_notes_scheduler.group(1).strip() if re_notes_scheduler else DEFAULT_SCHEDULER
        # build lora base
        re_notes_lora_base = re.search(LORA_BASE_REGEX, notes, re.I)
        lora_base = re_notes_lora_base.group(1).strip() if re_notes_lora_base else DEFAULT_LORA_BASE
        previewPrompOptions = GeneratePreviewsOptions(
            positive,
            negative,
            seed,
            denoise,
            upscale_denoise,
            cfg,
            steps,
            clip,
            upscale_factor,
            sampler,
            scheduler,
            lora_base
        )
        return previewPrompOptions
    
    def get_first_tag_list(self):
        return [item["tags"].split(",")[0].strip() for item in self.items]
    
    def get_main_first_tag(self):
        return self.items[0]["tags"].split(",")[0].strip()


def any_have_previews(items):
    count = 0

    for item in items:
        filepath = item["filepath"]
        basepath = os.path.splitext(filepath)[0]
        path = basepath + ".png"
        if os.path.exists(path):
            count += 1

    return count

preview_dialog = None
async def run(app, items, op = None):
    if not items:
        return

    if op is None:
        count = any_have_previews(items)
        if count > 0:
            dlg = wx.MessageDialog(
                app.frame,
                f"{count} models already have preview files. Do you want to replace them or add them to the preview images list?",
                "Existing Previews",
                wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION,
            )
            dlg.SetYesNoLabels("Replace", "Append")
            result = await wxasync.AsyncShowDialogModal(dlg)
            if result == wx.ID_CANCEL:
                return
            op = "replace" if result == wx.ID_YES else "append"

    global preview_dialog
    if preview_dialog is not None:
        preview_dialog.Destroy()
        preview_dialog = None

    preview_dialog = PreviewGeneratorDialog(app.frame, app, items, op)
    preview_dialog.Center()
    await AsyncShowDialog(preview_dialog)
