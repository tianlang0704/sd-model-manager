import asyncio
import os
import re
import wx
import time
import shutil
import random
import wxasync
import traceback
import simplejson
from wx.lib.agw import floatspin
from dataclasses import dataclass
from asyncio.locks import Event
from aiopubsub import Key

from sd_model_manager.utils.common import try_load_image
from gui import ids, utils
from gui.api import ComfyAPI, ModelManagerAPI
from gui.utils import PROGRAM_ROOT, COMFY_ROOT, combine_tag_freq
from gui.comfy_executor import ComfyExecutor
from gui.image_panel import ImagePanel
from gui.async_utils import AsyncShowDialog, AsyncShowDialogModal, on_close
from gui.panels.properties import MODEL_SD_15_TAG\
    , MODEL_SD_XL_TAG\
    , MODEL_SD_TURBO_TAG\
    , MODEL_SD_MERGE_TURBO_TAG\
    , MODEL_SD_LORA_TAG

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
DEFAULT_ADD_NOISE = 0.0
DEFAULT_UPSCALE_ADD_NOISE = 0.0
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_SCHEDULER = "normal"
DEFAULT_LORA_BASE = "AOM3.safetensors"
REGEX_POSITIVE = r"positive:(.+)\n*"
REGEX_NEGATIVE = r"negative:(.+)\n*"
REGEX_SEED = r"seed:\s*(\-?\s*\d+)\n*"
REGEX_DENOISE = r"(upscale)?[\t\f \-_]*denoise:\s*?(\-?\s*\d+\s*?\.?\s*?\d*)\n*" #workaround for detecting upscale denoise
REGEX_UPSCALE_DENOISE = r"upscale[\t\f \-_]*denoise:\s*(\-?\s*\d+\s*\.?\s*\d*)\n*"
REGEX_CFG = r"cfg:\s*(\d+)\n*"
REGEX_STEPS = r"steps:\s*(\d+)\n*"
REGEX_CLIP = r"clip:\s*(\-?\s*\d+)\n*"
REGEX_UPSCALE_FACTOR = r"upscale[\t\f \-_]*factor:\s*(\d+\s*\.?\s*\d*)\n*"
REGEX_ADD_NOISE = r"(upscale)?[\t\f \-_]*add[\t\f \-_]*noise:\s*(\-?\s*\d+\s*\.?\s*\d*)\n*"
REGEX_UPSCALE_ADD_NOISE = r"upscale[\t\f \-_]*add[\t\f \-_]*noise:\s*(\-?\s*\d+\s*\.?\s*\d*)\n*"
REGEX_SAMPLER = r"sampler:\s*([\w\.\-_]+)\n*"
REGEX_SCHEDULER = r"scheduler:\s*([\w\.\-_]+)\n*"
REGEX_LORA_BASE = r"lora[\t\f \-_]*base:\s*([\w\.\-_ ]+)\n*"
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
    add_noise: float
    check_add_noise: bool
    upscale_add_noise: float
    check_upscale_add_noise: bool
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
    add_noise: float
    check_add_noise: bool
    upscale_add_noise: float
    check_upscale_add_noise: bool
    sampler: str
    scheduler: str
    lora_base: str
    tags: str

    def get_node_with_type(self, prompt, type_list):
        if isinstance(type_list, str):
            type_list = [type_list]
        for key, value in prompt.items():
            class_type = value.get("class_type")
            if class_type is None:
                continue
            if class_type in type_list:
                return key
        return None

    def add_noise_node(
        self, 
        prompt,
        node_prefix, 
        in_model_node = None,
        in_model_output = None, 
        in_latent_node = None, 
        in_latent_output = None, 
        out_latent_node = None, 
        out_latent_input_name = None,
        add_noise = None,
        check_add_noise = None,
    ):
        if add_noise == 0 or not check_add_noise:
            return
        add_noise_subnodes = load_prompt("add_noise.json")
        for key, value in add_noise_subnodes.items():
            inputs = value.get("inputs")
            if not inputs:
                continue
            for key, value in inputs.items():
                if isinstance(value, list):
                    value[0] = f"{node_prefix}{value[0]}"
        if in_model_node is None:
            in_model_node = self.get_node_with_type(prompt, ["CheckpointLoaderSimple"])
            in_model_output = 0
        if in_latent_node is None:
            in_latent_node = self.get_node_with_type(prompt, ["EmptyLatentImage", "LatentUpscaleBy"])
            in_latent_output = 0
        if out_latent_node is None:
            out_latent_node = self.get_node_with_type(prompt, ["KSampler", "SamplerCustom", "KSamplerAdvanced"])
            out_latent_input_name = "latent_image"

        add_noise_subnodes["62"]["inputs"]["latent"] = [in_latent_node, in_latent_output]
        add_noise_subnodes["43"]["inputs"]["latents"] = [in_latent_node, in_latent_output]
        add_noise_subnodes["54"]["inputs"]["model"] = [in_model_node, in_model_output]
        add_noise_subnodes["42"]["inputs"]["model"] = [in_model_node, in_model_output]
        prompt[out_latent_node]["inputs"][out_latent_input_name] = [node_prefix + "43", 0]

        add_noise_subnodes["54"]["inputs"]["scheduler"] = self.scheduler
        add_noise_subnodes["54"]["inputs"]["steps"] = self.steps
        add_noise_subnodes["54"]["inputs"]["denoise"] = self.denoise
        add_noise_subnodes["42"]["inputs"]["seed"] = self.seed if add_noise > 0 else random.randint(0, SEED_RANDOM_MAX)
        add_noise_subnodes["43"]["inputs"]["strength"] = abs(add_noise)
            
        keys = list(add_noise_subnodes.keys())
        for key in keys:
            add_noise_subnodes[f"{node_prefix}{key}"] = add_noise_subnodes.pop(key)
        prompt.update(add_noise_subnodes)

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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.add_noise, check_add_noise=self.check_add_noise)
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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.add_noise, check_add_noise=self.check_add_noise)
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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.add_noise, check_add_noise=self.check_add_noise)
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
        prompt["15"]["inputs"]["text"] = self.positive
        prompt["16"]["inputs"]["text"] = self.negative
        prompt["53"]["inputs"]["value"] = self.steps
        prompt["54"]["inputs"]["value"] = int(self.steps * self.denoise)
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.add_noise, check_add_noise=self.check_add_noise)
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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.add_noise, check_add_noise=self.check_add_noise)
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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.upscale_add_noise, check_add_noise=self.check_upscale_add_noise)
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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.upscale_add_noise, check_add_noise=self.check_upscale_add_noise)
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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.upscale_add_noise, check_add_noise=self.check_upscale_add_noise)
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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.upscale_add_noise, check_add_noise=self.check_upscale_add_noise)
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
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.upscale_add_noise, check_add_noise=self.check_upscale_add_noise)
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
        self.comfy_api = ComfyAPI(self.app)
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

        def OnEnter(e):
            if wx.GetKeyState(wx.WXK_SHIFT):
                self.OnRegenerate(e)
            if wx.GetKeyState(wx.WXK_ALT):
                self.OnUpscale(e)
        self.Bind(wx.EVT_TEXT_ENTER, OnEnter)
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

        sizerRightAfterCFGStepClip = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfterCFGStepClip.Add(
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
        sizerRightAfterCFGStepClip.Add(self.spinner_cfg, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfterCFGStepClip.Add(
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
        sizerRightAfterCFGStepClip.Add(self.spinner_steps, proportion=1, flag=wx.ALL, border=5)

        clip_label = sizerRightAfterCFGStepClip.Add(
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
        sizerRightAfterCFGStepClip.Add(self.spinner_clip, proportion=1, flag=wx.ALL, border=5)
        firstTagList = self.get_first_tag_list()
        show_clip = False
        if sum(1 for tag in firstTagList if tag != MODEL_SD_TURBO_TAG and tag != MODEL_SD_XL_TAG) > 0:
            show_clip = True
        clip_label.Show(show_clip)
        self.spinner_clip.Show(show_clip)

        sizerRightAfterDenoise = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfterDenoise.Add(
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
        sizerRightAfterDenoise.Add(self.spinner_denoise, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfterDenoise.Add(
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
        sizerRightAfterDenoise.Add(self.spinner_upscale_denoise, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfterAddNoise = wx.BoxSizer(wx.HORIZONTAL)
        self.check_add_noise = wx.CheckBox(self, wx.ID_ANY, label="Noise")
        self.check_add_noise.SetValue(self.preview_options.check_add_noise)
        self.temp_add_noise = self.preview_options.add_noise
        def OnAddNoiseCheck(e):
            if not self.check_add_noise.GetValue():
                self.spinner_add_noise.SetValue(0)
            else:
                self.spinner_add_noise.SetValue(self.temp_add_noise)
        self.check_add_noise.Bind(wx.EVT_CHECKBOX, OnAddNoiseCheck)
        sizerRightAfterAddNoise.Add(self.check_add_noise, proportion=0, border=5, flag=wx.TOP|wx.BOTTOM)
        self.spinner_add_noise = floatspin.FloatSpin(
            self,
            id=wx.ID_ANY,
            min_val=-10,
            max_val=10,
            increment=0.01,
            value=self.preview_options.add_noise,
            agwStyle=floatspin.FS_LEFT,
            size=self.Parent.FromDIP(wx.Size(60, 25)),
        )
        self.spinner_add_noise.SetFormat("%f")
        self.spinner_add_noise.SetDigits(3)
        sizerRightAfterAddNoise.Add(self.spinner_add_noise, proportion=1, flag=wx.ALL, border=5)
        def OnAddNoiseChange(e):
            value = float(e.GetString())
            if value != 0:
                self.temp_add_noise = value
            self.check_add_noise.SetValue(value != 0)
        self.spinner_add_noise.Bind(wx.EVT_TEXT, OnAddNoiseChange)
        
        self.check_upscale_add_noise = wx.CheckBox(self, wx.ID_ANY, label="Upscale N.")
        self.check_upscale_add_noise.SetValue(self.preview_options.check_upscale_add_noise)
        self.temp_upscale_add_noise = self.preview_options.upscale_add_noise
        def OnUpscaleAddNoiseCheck(e):
            if not self.check_upscale_add_noise.GetValue():
                self.spinner_upscale_add_noise.SetValue(0)
            else:
                self.spinner_upscale_add_noise.SetValue(self.temp_upscale_add_noise)
        self.check_upscale_add_noise.Bind(wx.EVT_CHECKBOX, OnUpscaleAddNoiseCheck)
        sizerRightAfterAddNoise.Add(self.check_upscale_add_noise, proportion=0, border=5, flag=wx.TOP|wx.BOTTOM)
        self.spinner_upscale_add_noise = floatspin.FloatSpin(
            self,
            id=wx.ID_ANY,
            min_val=-10,
            max_val=10,
            increment=0.01,
            value=self.preview_options.upscale_add_noise,
            agwStyle=floatspin.FS_LEFT,
            size=self.Parent.FromDIP(wx.Size(60, 25)),
        )
        self.spinner_upscale_add_noise.SetFormat("%f")
        self.spinner_upscale_add_noise.SetDigits(3)
        sizerRightAfterAddNoise.Add(self.spinner_upscale_add_noise, proportion=1, flag=wx.ALL, border=5)
        def OnUpscaleAddNoiseChange(e):
            value = float(e.GetString())
            if value != 0:
                self.temp_upscale_add_noise = value
            self.check_upscale_add_noise.SetValue(value != 0)
        self.spinner_upscale_add_noise.Bind(wx.EVT_TEXT, OnUpscaleAddNoiseChange)
        
        sizerRightAfterUpscale = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfterUpscale.Add(
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
        sizerRightAfterUpscale.Add(self.spinner_upscale_factor, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfterSampler = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfterSampler.Add(
            wx.StaticText(self, wx.ID_ANY, label="Sampler"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        def read_list_from_file(filename, list_name):
            if not os.path.isfile(filename):
                return
            with open(filename, "r") as f:
                content = f.read()
                choices = []
                match = re.search(rf"{list_name} = \[([^\]]+)\]", content)
                list_content = match.group(1) if match else ""
                if not list_content:
                    return
                names = list_content.split(",")
                for name in names:
                    name = name.strip().replace("'", "").replace('"', "")
                    if name:
                        choices.append(name)
                if len(choices) <= 0:
                    return
                return choices
        choices = ["euler_ancestral", "dpmpp_2m", "dpmpp_3m_sde"]
        if self.app.config.mode == "comfyui":
            try:
                file = os.path.join(COMFY_ROOT, "comfy/samplers.py")
                choices = read_list_from_file(file, "KSAMPLER_NAMES") or choices
            except:
                pass
        self.sampler = wx.ComboBox(
            self, 
            id=wx.ID_ANY, 
            value=self.preview_options.sampler, 
            choices=choices, 
            style=wx.TE_PROCESS_ENTER
        )
        sizerRightAfterSampler.Add(self.sampler, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfterScheduler = wx.BoxSizer(wx.HORIZONTAL)
        scheduler_label = sizerRightAfterScheduler.Add(
            wx.StaticText(self, wx.ID_ANY, label="Scheduler"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        choices = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
        if self.app.config.mode == "comfyui":
            try:
                file = os.path.join(COMFY_ROOT, "comfy/samplers.py")
                choices = read_list_from_file(file, "SCHEDULER_NAMES") or choices
            except:
                pass
        self.scheduler = wx.ComboBox(
            self, 
            id=wx.ID_ANY, 
            value=self.preview_options.scheduler, 
            choices=choices, style=wx.TE_PROCESS_ENTER
        )
        sizerRightAfterScheduler.Add(self.scheduler, proportion=1, flag=wx.ALL, border=5)
        firstTagList = self.get_first_tag_list()
        show_scheduler = False
        if sum(1 for tag in firstTagList if tag != MODEL_SD_TURBO_TAG) > 0:
            show_scheduler = True
        scheduler_label.Show(show_scheduler)
        self.scheduler.Show(show_scheduler)

        sizerRightAfterBaseLora = wx.BoxSizer(wx.HORIZONTAL)
        lora_base_label = sizerRightAfterBaseLora.Add(
            wx.StaticText(self, wx.ID_ANY, label="Lora Base"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        choices = ["PerfectWorld.safetensors", "astranime_V6.safetensors", "AOM3.safetensors", "Anything-V3.0-pruned-fp16.ckpt", "v1-5-pruned.ckpt"]
        dynamic_choices = self.app.frame.results_panel.results_panel.list.get_all_names()
        if len(dynamic_choices) > 0:
            for choice in choices:
                if choice in dynamic_choices:
                    dynamic_choices.remove(choice)
                    dynamic_choices.insert(0, choice)
            choices = dynamic_choices
        self.lora_base = wx.ComboBox(
            self, 
            id=wx.ID_ANY, 
            value=self.preview_options.lora_base, 
            choices=choices,
            size = self.Parent.FromDIP(wx.Size(200, 25)),
            style=wx.TE_PROCESS_ENTER,
        )
        # workaround for wxpython backward compatibility: SetItmes and SetValue must trigger EVT_TEXT
        async def OnLoraBaseChangeDummy(e):
            pass
        latest_time = 0
        async def OnLoraBaseChange(e):
            text = self.lora_base.GetValue()
            select = self.lora_base.GetCurrentSelection()
            selected_text = None
            if select != wx.NOT_FOUND:
                items = self.lora_base.GetItems()
                selected_text = items[select]
            if text == selected_text:
                return
            nonlocal latest_time
            latest_time = time.time()
            my_time = latest_time
            if text != "":
                await asyncio.sleep(0.5)
            if my_time != latest_time:
                return
            wxasync.AsyncBind(wx.EVT_TEXT, OnLoraBaseChangeDummy, self.lora_base)
            caret = self.lora_base.GetInsertionPoint()
            filter_choice = [choice for choice in choices if text.lower() in choice.lower()]
            self.lora_base.SetItems(filter_choice)
            self.lora_base.SetSelection(-1, -1)
            self.lora_base.Popup()
            self.lora_base.SetValue(text)
            self.lora_base.SetInsertionPoint(caret)
            wxasync.AsyncBind(wx.EVT_TEXT, OnLoraBaseChange, self.lora_base)
        wxasync.AsyncBind(wx.EVT_TEXT, OnLoraBaseChange, self.lora_base)

        sizerRightAfterBaseLora.Add(self.lora_base, proportion=1, flag=wx.ALL, border=5)
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
        sizerRight.Add(sizerRightAfterCFGStepClip, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterDenoise, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterAddNoise, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterUpscale, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterSampler, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterScheduler, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterBaseLora, proportion=1, flag=wx.ALL)
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
        if not filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return False
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

        await self.app.frame.results_panel.re_search()
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
        if re.search(REGEX_POSITIVE, notes, re.I):
            notes = re.sub(REGEX_POSITIVE, f"positive: {positive}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"positive: {positive}\n"

        negative = self.text_prompt_after.GetValue()
        if re.search(REGEX_NEGATIVE, notes, re.I):
            notes = re.sub(REGEX_NEGATIVE, f"negative: {negative}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"negative: {negative}\n"

        seed = self.spinner_seed.GetValue()
        if re.search(REGEX_SEED, notes, re.I):
            notes = re.sub(REGEX_SEED, f"seed: {seed}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"seed: {seed}\n"
        
        denoise = self.spinner_denoise.GetValue()
        re_result = re.finditer(REGEX_DENOISE, notes, re.I)
        count = sum(1 for match in re_result if not match.group(1))
        if count:
            notes = re.sub(REGEX_DENOISE, lambda match: match.group(0) if match.group(1) else f"denoise: {denoise}\n", notes, flags = re.I)
        else:
            insert_str += f"denoise: {denoise}\n"

        upscale_denoise = self.spinner_upscale_denoise.GetValue()
        if re.search(REGEX_UPSCALE_DENOISE, notes, re.I):
            notes = re.sub(REGEX_UPSCALE_DENOISE, f"upscale denoise: {upscale_denoise}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"upscale denoise: {upscale_denoise}\n"

        cfg = self.spinner_cfg.GetValue()
        if re.search(REGEX_CFG, notes, re.I):
            notes = re.sub(REGEX_CFG, f"cfg: {cfg}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"cfg: {cfg}\n"

        steps = self.spinner_steps.GetValue()
        if re.search(REGEX_STEPS, notes, re.I):
            notes = re.sub(REGEX_STEPS, f"steps: {steps}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"steps: {steps}\n"

        # turbo and xl no clip
        firstTag = self.get_main_first_tag()
        if firstTag != MODEL_SD_TURBO_TAG and firstTag != MODEL_SD_XL_TAG:
            clip = self.spinner_clip.GetValue()
            if re.search(REGEX_CLIP, notes, re.I):
                notes = re.sub(REGEX_CLIP, f"clip: {clip}\n", notes, flags = re.I, count = 1)
            else:
                insert_str += f"clip: {clip}\n"

        upscale_factor = self.spinner_upscale_factor.GetValue()
        if re.search(REGEX_UPSCALE_FACTOR, notes, re.I):
            notes = re.sub(REGEX_UPSCALE_FACTOR, f"upscale factor: {upscale_factor}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"upscale factor: {upscale_factor}\n"

        add_noise = self.spinner_add_noise.GetValue()
        if not self.check_add_noise.GetValue():
            add_noise = 0
        re_result = re.finditer(REGEX_ADD_NOISE, notes, re.I)
        count = sum(1 for match in re_result if not match.group(1))
        if count:
            notes = re.sub(REGEX_ADD_NOISE, lambda match: match.group(0) if match.group(1) else f"add noise: {add_noise}\n", notes, flags = re.I)
        else:
            insert_str += f"add noise: {add_noise}\n"

        upscale_add_noise = self.spinner_upscale_add_noise.GetValue()
        if not self.check_upscale_add_noise.GetValue():
            upscale_add_noise = 0
        if re.search(REGEX_UPSCALE_ADD_NOISE, notes, re.I):
            notes = re.sub(REGEX_UPSCALE_ADD_NOISE, f"upscale add noise: {upscale_add_noise}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"upscale add noise: {upscale_add_noise}\n"

        sampler = self.sampler.GetValue()
        if re.search(REGEX_SAMPLER, notes, re.I):
            notes = re.sub(REGEX_SAMPLER, f"sampler: {sampler}\n", notes, flags = re.I, count = 1)
        else:
            insert_str += f"sampler: {sampler}\n"
        
        # turbo no scheduler
        firstTag = self.get_main_first_tag()
        if firstTag != MODEL_SD_TURBO_TAG:
            scheduler = self.scheduler.GetValue()
            if re.search(REGEX_SCHEDULER, notes, re.I):
                notes = re.sub(REGEX_SCHEDULER, f"scheduler: {scheduler}\n", notes, flags = re.I, count = 1)
            else:
                insert_str += f"scheduler: {scheduler}\n"

        # only lora need base model
        firstTag = self.get_main_first_tag()
        if firstTag == MODEL_SD_LORA_TAG:
            lora_base = self.lora_base.GetValue()
            if re.search(REGEX_LORA_BASE, notes, re.I):
                notes = re.sub(REGEX_LORA_BASE, f"lora base: {lora_base}\n", notes, flags = re.I, count = 1)
            else:
                insert_str += f"lora base: {lora_base}\n"

        results_panel = self.app.frame.results_panel
        notes = insert_str + notes
        await self.app.api.update_lora(main_item["id"], {"notes": notes})
        await results_panel.re_search()
        

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
        prompt_before = self.text_prompt_before.GetValue().strip().replace("\n", " ,")
        return GeneratePreviewsOptions(
            prompt_before=prompt_before,
            prompt_after=self.text_prompt_after.GetValue(),
            seed=int(self.spinner_seed.GetValue()),
            denoise=float(self.spinner_denoise.GetValue()),
            upscale_denoise=float(self.spinner_upscale_denoise.GetValue()),
            cfg=int(self.spinner_cfg.GetValue()),
            steps=int(self.spinner_steps.GetValue()),
            clip=int(self.spinner_clip.GetValue()),
            upscale_factor=float(self.spinner_upscale_factor.GetValue()),
            add_noise=float(self.spinner_add_noise.GetValue()),
            check_add_noise=bool(self.check_add_noise.GetValue()),
            upscale_add_noise=float(self.spinner_upscale_add_noise.GetValue()),
            check_upscale_add_noise=bool(self.check_upscale_add_noise.GetValue()),
            sampler=self.sampler.GetValue(),
            scheduler=self.scheduler.GetValue(),
            lora_base=self.lora_base.GetValue(),
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
        add_noise = inputOptions.add_noise if inputOptions.add_noise != self.preview_options.add_noise else itemOptions.add_noise
        upscale_add_noise = inputOptions.upscale_add_noise if inputOptions.upscale_add_noise != self.preview_options.upscale_add_noise else itemOptions.upscale_add_noise
        sampler = inputOptions.sampler if inputOptions.sampler != self.preview_options.sampler else itemOptions.sampler
        scheduler = inputOptions.scheduler if inputOptions.scheduler != self.preview_options.scheduler else itemOptions.scheduler
        lora_base = inputOptions.lora_base if inputOptions.lora_base != self.preview_options.lora_base else itemOptions.lora_base
        tags = item["tags"]
        data = PreviewPromptData(
            seed = seed, 
            denoise = denoise,
            upscale_denoise = upscale_denoise,
            checkpoint = checkpoint,
            positive = positive,
            negative = negative,
            cfg = cfg,
            steps = steps,
            clip = clip,
            upscale_factor = upscale_factor,
            add_noise = add_noise,
            check_add_noise = inputOptions.check_add_noise,
            upscale_add_noise = upscale_add_noise,
            check_upscale_add_noise = inputOptions.check_upscale_add_noise,
            sampler = sampler,
            scheduler = scheduler,
            lora_base = lora_base,
            tags = tags
        )
        print(f"Seed: {seed}")
        return data

    def enqueue_prompt_and_wait(self, executor, prompt):
        queue_result = executor.enqueue(prompt)
        if queue_result.get("error") is not None:
            raise(Exception(queue_result))
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

        prompt_id = None
        image_data = None
        image_location = None
        with ComfyExecutor(self.app) as executor:
            data = self.assemble_prompt_data(item)
            prompt = data.to_prompt()
            prompt_id = self.enqueue_prompt_and_wait(executor, prompt)
        if prompt_id is not None:
            image_data, image_location = self.get_output_image(prompt_id)
        if image_data is not None:
            self.image_panel.LoadImageFromBytes(image_data)

        self.last_data = data
        self.last_output = image_location
        self.result = image_location
        self.last_seed = data.seed

        self.after_execute()

    def do_upscale(self, item):
        self.before_execute()
        prompt_id = None
        image_data = None
        image_location = None
        with ComfyExecutor(self.app) as executor:
            data = self.assemble_prompt_data(item)
            prompt = data.to_hr_prompt(self.last_output)
            prompt_id = self.enqueue_prompt_and_wait(executor, prompt)
        if prompt_id is not None:
            image_data, image_location = self.get_output_image(prompt_id)
        if image_data is not None:
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
        msg = ex
        error = len(ex.args) > 0 and ex.args[0]
        if isinstance(error, dict):
            msg = error.get("error") or msg
            details = error.get("details")
            if details:
                msg += f"\n{details}"
            
        dialog = wx.MessageDialog(
            self,
            f"Failed to generate previews:\n{msg}\nMost likely the sampler or scheduler or lora base is not available, please check on ComfyUI",
            "Generation Failed",
            wx.OK | wx.ICON_ERROR,
        )
        await wxasync.AsyncShowDialogModal(dialog)
        self.AsyncEndModal(wx.ID_CANCEL)

    def item_to_preview_options(self, itemsOrItems):
        item = itemsOrItems
        if isinstance(itemsOrItems, list):
            item = itemsOrItems[0]
        notes = item.get("notes") or ""
        def get_default_prompt(prompt_key, regex, default):
            re_notes_prompt = re.search(regex, notes, re.I)
            notes_prompt = re_notes_prompt.group(1).strip() if re_notes_prompt else default
            prompt = notes_prompt.strip()
            keywords = item[prompt_key]
            if not keywords:
                return prompt
            if keywords.startswith(OVERRIDE_KEYWORD):
                prompt = ""
                keywords = keywords.replace(OVERRIDE_KEYWORD, "")
            else:
                keyToInsert = []
                keywords = keywords.replace("\n", ", ")
                keywordList = keywords.split(",")
                for key in keywordList:
                    key = key.strip()
                    if key not in prompt:
                        keyToInsert.append(key.strip())
                if len(keyToInsert) > 0:
                    keyStr = ", ".join(keyToInsert)
                    keywords = f", {keyStr}"
                else:
                    keywords = ""
            prompt += keywords
            return prompt
        # build positive prompt
        positive = get_default_prompt("keywords", REGEX_POSITIVE, DEFAULT_POSITIVE)
        # build negative prompt
        negative = get_default_prompt("negative_keywords", REGEX_NEGATIVE, DEFAULT_NEGATIVE)
        # build seed
        re_notes_seed = re.search(REGEX_SEED, notes, re.I)
        seed = int(re_notes_seed.group(1).strip()) if re_notes_seed else DEFAULT_SEED
        # build denoise
        re_result = re.finditer(REGEX_DENOISE, notes, re.I)
        re_notes_denoise = next((match for match in re_result if not match.group(1)), None)
        denoise = float(re_notes_denoise.group(2).strip()) if re_notes_denoise else DEFAULT_DENOISE
        # build upscale denoise
        re_notes_upscale_denoise = re.search(REGEX_UPSCALE_DENOISE, notes, re.I)
        upscale_denoise = float(re_notes_upscale_denoise.group(1).strip()) if re_notes_upscale_denoise else DEFAULT_UPSCALE_DENOISE
        # build cfg
        re_notes_cfg = re.search(REGEX_CFG, notes, re.I)
        cfg = int(re_notes_cfg.group(1).strip()) if re_notes_cfg else DEFAULT_CFG
        # build steps
        re_notes_steps = re.search(REGEX_STEPS, notes, re.I)
        steps = int(re_notes_steps.group(1).strip()) if re_notes_steps else DEFAULT_STEPS
        # build clip
        re_notes_clip = re.search(REGEX_CLIP, notes, re.I)
        clip = int(re_notes_clip.group(1).strip()) if re_notes_clip else DEFAULT_CLIP
        # build upscale factor
        re_notes_upscale_factor = re.search(REGEX_UPSCALE_FACTOR, notes, re.I)
        upscale_factor = float(re_notes_upscale_factor.group(1).strip()) if re_notes_upscale_factor else DEFAULT_UPSCALE_FACTOR
        # build add noise
        re_result = re.finditer(REGEX_ADD_NOISE, notes, re.I)
        re_notes_add_noise = next((match for match in re_result if not match.group(1)), None)
        add_noise = float(re_notes_add_noise.group(2).strip()) if re_notes_add_noise else DEFAULT_ADD_NOISE
        # build upscale add noise
        re_notes_upscale_add_noise = re.search(REGEX_UPSCALE_ADD_NOISE, notes, re.I)
        upscale_add_noise = float(re_notes_upscale_add_noise.group(1).strip()) if re_notes_upscale_add_noise else DEFAULT_UPSCALE_ADD_NOISE
        # build sampler
        re_notes_sampler = re.search(REGEX_SAMPLER, notes, re.I)
        sampler = re_notes_sampler.group(1).strip() if re_notes_sampler else DEFAULT_SAMPLER
        # build scheduler
        re_notes_scheduler = re.search(REGEX_SCHEDULER, notes, re.I)
        scheduler = re_notes_scheduler.group(1).strip() if re_notes_scheduler else DEFAULT_SCHEDULER
        # build lora base
        re_notes_lora_base = re.search(REGEX_LORA_BASE, notes, re.I)
        lora_base = re_notes_lora_base.group(1).strip() if re_notes_lora_base else DEFAULT_LORA_BASE
        previewPrompOptions = GeneratePreviewsOptions(
            prompt_before = positive,
            prompt_after = negative,
            seed = seed,
            denoise = denoise,
            upscale_denoise = upscale_denoise,
            cfg = cfg,
            steps = steps,
            clip = clip,
            upscale_factor = upscale_factor,
            add_noise = add_noise,
            check_add_noise = add_noise != 0,
            upscale_add_noise = upscale_add_noise,
            check_upscale_add_noise = upscale_add_noise != 0,
            sampler = sampler,
            scheduler = scheduler,
            lora_base = lora_base
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
