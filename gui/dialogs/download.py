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
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_SCHEDULER = "normal"
DEFAULT_LORA_BASE = "AOM3.safetensors"
REGEX_POSITIVE = r"positive:(.+)\n*"
REGEX_NEGATIVE = r"negative:(.+)\n*"
REGEX_SEED = r"seed:\s*(\-?\s*\d+)\n*"
REGEX_DENOISE = r"denoise:\s*?(\-?\s*\d+\s*?\.?\s*?\d*)\n*"
REGEX_CFG = r"cfg:\s*(\d+)\n*"
REGEX_STEPS = r"steps:\s*(\d+)\n*"
REGEX_CLIP = r"clip:\s*(\-?\s*\d+)\n*"
REGEX_UPSCALE_FACTOR = r"upscale[\t\f \-_]*factor:\s*(\d+\s*\.?\s*\d*)\n*"
REGEX_ADD_NOISE = r"add[\t\f \-_]*noise:\s*(\-?\s*\d+\s*\.?\s*\d*)\n*"
REGEX_SAMPLER = r"sampler:\s*([\w\.\-_]+)\n*"
REGEX_SCHEDULER = r"scheduler:\s*([\w\.\-_]+)\n*"
REGEX_LORA_BASE = r"lora[\t\f \-_]*base:\s*([\w\.\-_ ]+)\n*"
REGEX_PREFIX = r"({0})?[\t\f \-_]*{1}"
# (generate)?[\t\f \-_]*positive:(.+)\n*
REGEX_GEN_PREFIX = "generate"
REGEX_UPS_PREFIX = "upscale"
SEED_RANDOM_MAX = 2**32


def load_prompt(name):
    with open(
        os.path.join(PROGRAM_ROOT, "gui/prompts", name), "r", encoding="utf-8"
    ) as f:
        return simplejson.load(f)


@dataclass
class GeneratePreviewsOptions:
    positive: str
    negative: str
    seed: int
    denoise: float
    cfg: int
    steps: int
    clip: int
    upscale_factor: float
    add_noise: float
    check_add_noise: bool
    sampler: str
    scheduler: str
    lora_base: str


@dataclass
class PreviewPromptData:
    checkpoint: str
    tags: str
    gen_options: GeneratePreviewsOptions
    ups_options: GeneratePreviewsOptions
    current_options: GeneratePreviewsOptions = None

    def get_node_with_type(self, prompt, type_list):
        if isinstance(type_list, str):
            type_list = [type_list]
        res_list = []
        for key, value in prompt.items():
            class_type = value.get("class_type")
            if class_type is None:
                continue
            if class_type in type_list:
                res_list.append(key)
        first = res_list[0] if len(res_list) > 0 else None
        return first, res_list

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

        add_noise_subnodes["54"]["inputs"]["scheduler"] = self.current_options.scheduler
        add_noise_subnodes["54"]["inputs"]["steps"] = self.current_options.steps
        add_noise_subnodes["54"]["inputs"]["denoise"] = self.current_options.denoise
        add_noise_subnodes["42"]["inputs"]["seed"] = self.current_options.seed if add_noise > 0 else random.randint(0, SEED_RANDOM_MAX)
        add_noise_subnodes["43"]["inputs"]["strength"] = abs(add_noise)
            
        keys = list(add_noise_subnodes.keys())
        for key in keys:
            add_noise_subnodes[f"{node_prefix}{key}"] = add_noise_subnodes.pop(key)
        prompt.update(add_noise_subnodes)

    def to_prompt(self):
        self.current_options = self.gen_options
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
    
    def guess_fit_current_options_to_prompt(self, prompt):
        node, node_list = self.get_node_with_type(prompt, ["KSampler"])
        for key in node_list:
            prompt[key]["inputs"]["seed"] = self.current_options.seed
            prompt[key]["inputs"]["denoise"] = self.current_options.denoise
            prompt[key]["inputs"]["cfg"] = self.current_options.cfg
            prompt[key]["inputs"]["steps"] = self.current_options.steps
            prompt[key]["inputs"]["sampler_name"] = self.current_options.sampler
            prompt[key]["inputs"]["scheduler"] = self.current_options.scheduler
        node, node_list = self.get_node_with_type(prompt, ["KSamplerAdvanced"])
        for key in node_list:
            prompt[key]["inputs"]["noise_seed"] = self.current_options.seed
            prompt[key]["inputs"]["cfg"] = self.current_options.cfg
            prompt[key]["inputs"]["steps"] = self.current_options.steps
            prompt[key]["inputs"]["sampler_name"] = self.current_options.sampler
            prompt[key]["inputs"]["scheduler"] = self.current_options.scheduler
        node, node_list = self.get_node_with_type(prompt, ["SamplerCustom"])
        for key in node_list:
            prompt[key]["inputs"]["noise_seed"] = self.current_options.seed
            prompt[key]["inputs"]["cfg"] = self.current_options.cfg
            prompt[key]["inputs"]["steps"] = self.current_options.steps
        node, node_list = self.get_node_with_type(prompt, ["SDTurboScheduler"])
        for key in node_list:
            prompt[key]["inputs"]["steps"] = self.current_options.steps
            prompt[key]["inputs"]["denoise"] = self.current_options.denoise
        node, node_list = self.get_node_with_type(prompt, ["KSamplerSelect"])
        for key in node_list:
            prompt[key]["inputs"]["sampler_name"] = self.current_options.sampler
        node, node_list = self.get_node_with_type(prompt, ["CheckpointLoaderSimple"])
        for key in node_list:
            prompt[key]["inputs"]["ckpt_name"] = self.checkpoint
        node, node_list = self.get_node_with_type(prompt, ["CLIPTextEncode"])
        for index, key in enumerate(node_list):
            if index % 2 == 0:
                prompt[key]["inputs"]["text"] = self.current_options.positive
            else:
                prompt[key]["inputs"]["text"] = self.current_options.negative
        node, node_list = self.get_node_with_type(prompt, ["CLIPSetLastLayer"])
        for key in node_list:
            prompt[key]["inputs"]["stop_at_clip_layer"] = self.current_options.clip
        node, node_list = self.get_node_with_type(prompt, ["LatentUpscaleBy"])
        for key in node_list:
            prompt[key]["inputs"]["scale_by"] = self.current_options.upscale_factor
        return prompt
            
    def to_prompt_lora(self):
        prompt = load_prompt("lora.json")
        self.guess_fit_current_options_to_prompt(prompt)
        prompt["230"]["inputs"]["lora_name"] = self.checkpoint
        prompt["4"]["inputs"]["ckpt_name"] = self.current_options.lora_base
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt
    
    def to_prompt_merge_turbo(self):
        prompt = load_prompt("merge_turbo.json")
        self.guess_fit_current_options_to_prompt(prompt)
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt

    def to_prompt_turbo(self):
        prompt = load_prompt("turbo.json")
        self.guess_fit_current_options_to_prompt(prompt)
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt

    def to_prompt_xl(self):
        prompt = load_prompt("xl.json")
        self.guess_fit_current_options_to_prompt(prompt)
        prompt["53"]["inputs"]["value"] = self.current_options.steps
        prompt["54"]["inputs"]["value"] = int(self.current_options.steps * self.current_options.denoise)
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt

    def to_prompt_default(self):
        prompt = load_prompt("default.json")
        self.guess_fit_current_options_to_prompt(prompt)
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt

    def to_hr_prompt(self, image):
        self.current_options = self.ups_options
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
        self.guess_fit_current_options_to_prompt(prompt)
        prompt["18"]["inputs"]["image"] = f"{filename} [output]"
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt
    
    def to_prompt_lora_hr(self, filename):
        prompt = load_prompt("lora-hr.json")
        self.guess_fit_current_options_to_prompt(prompt)
        prompt["22"]["inputs"]["lora_name"] = self.checkpoint
        prompt["16"]["inputs"]["ckpt_name"] = self.current_options.lora_base
        prompt["18"]["inputs"]["image"] = f"{filename} [output]"
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt

    def to_prompt_xl_hr(self, filename):
        prompt = load_prompt("xl-hr.json")
        self.guess_fit_current_options_to_prompt(prompt)
        prompt["27"]["inputs"]["start_at_step"] = int(self.current_options.steps * (1 - self.current_options.denoise))
        prompt["24"]["inputs"]["image"] = f"{filename} [output]"
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt
    
    def to_prompt_turbo_hr(self, filename):
        prompt = load_prompt("turbo-hr.json")
        self.guess_fit_current_options_to_prompt(prompt)
        prompt["38"]["inputs"]["image"] = f"{filename} [output]"
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
        return prompt
    
    def to_prompt_merge_turbo_hr(self, filename):
        prompt = load_prompt("merge_turbo-hr.json")
        self.guess_fit_current_options_to_prompt(prompt)
        prompt["10"]["inputs"]["image"] = f"{filename} [output]"
        self.add_noise_node(prompt=prompt, node_prefix="999", add_noise=self.current_options.add_noise, check_add_noise=self.current_options.check_add_noise)
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

class GenerationOptionsPanel(wx.Panel):
    def __init__(self, parent, app, preview_options, is_upscale=False, **kwargs):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY, **kwargs)
        self.app = app
        self.dialog = parent.GetParent()
        self.preview_options = preview_options

        sizerRight = wx.StaticBoxSizer(wx.VERTICAL, self)
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
        firstTagList = self.dialog.get_first_tag_list()
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

        self.check_add_noise = wx.CheckBox(self, wx.ID_ANY, label="Add Noise")
        self.check_add_noise.SetValue(self.preview_options.check_add_noise)
        self.temp_add_noise = self.preview_options.add_noise
        def OnAddNoiseCheck(e):
            if not self.check_add_noise.GetValue():
                self.spinner_add_noise.SetValue(0)
            else:
                self.spinner_add_noise.SetValue(self.temp_add_noise)
        self.check_add_noise.Bind(wx.EVT_CHECKBOX, OnAddNoiseCheck)
        sizerRightAfterDenoise.Add(self.check_add_noise, proportion=0, border=5, flag=wx.TOP|wx.BOTTOM)
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
        sizerRightAfterDenoise.Add(self.spinner_add_noise, proportion=1, flag=wx.ALL, border=5)
        def OnAddNoiseChange(e):
            value = float(e.GetString())
            if value != 0:
                self.temp_add_noise = value
            self.check_add_noise.SetValue(value != 0)
        self.spinner_add_noise.Bind(wx.EVT_TEXT, OnAddNoiseChange)
        
        sizerRightAfterUpscale = wx.BoxSizer(wx.HORIZONTAL)
        self.label_upscale_factor = wx.StaticText(self, wx.ID_ANY, label="Upscale Factor")
        sizerRightAfterUpscale.Add(
            self.label_upscale_factor,
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
        self.label_upscale_factor.Show(is_upscale)
        self.spinner_upscale_factor.Show(is_upscale)

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
        firstTagList = self.dialog.get_first_tag_list()
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
        firstTagList = self.dialog.get_first_tag_list()
        show_lora_base = False
        if MODEL_SD_LORA_TAG in firstTagList:
            show_lora_base = True
        lora_base_label.Show(show_lora_base)
        self.lora_base.Show(show_lora_base)

        self.text_positive = wx.TextCtrl(
            self,
            id=wx.ID_ANY,
            value=self.preview_options.positive,
            size=self.Parent.FromDIP(wx.Size(250, 100)),
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
        )
        
        self.text_negative = wx.TextCtrl(
            self,
            id=wx.ID_ANY,
            value=self.preview_options.negative,
            size=self.Parent.FromDIP(wx.Size(250, 100)),
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
        )

        sizerRight.Add(wx.StaticText(self, wx.ID_ANY, label="Positive"))
        sizerRight.Add(self.text_positive, proportion=3, flag=wx.ALL | wx.EXPAND)
        sizerRight.Add(wx.StaticText(self, wx.ID_ANY, label="Negative"))
        sizerRight.Add(self.text_negative, proportion=3, flag=wx.ALL | wx.EXPAND)
        sizerRight.Add(sizerRightAfter, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterCFGStepClip, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterDenoise, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterUpscale, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterSampler, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterScheduler, proportion=1, flag=wx.ALL)
        sizerRight.Add(sizerRightAfterBaseLora, proportion=1, flag=wx.ALL)

        self.SetSizerAndFit(sizerRight)

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
        self.gen_preview_options = self.item_to_preview_options(items, REGEX_GEN_PREFIX)
        self.ups_preview_options = self.item_to_preview_options(items, REGEX_UPS_PREFIX, is_upscale=True)
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

        self.notebook = wx.Notebook(self)
        self.gen_panel = GenerationOptionsPanel(self.notebook, self.app, self.gen_preview_options)
        self.ups_panel = GenerationOptionsPanel(self.notebook, self.app, self.ups_preview_options, is_upscale=True)
        self.notebook.AddPage(self.gen_panel, "Generation")
        self.notebook.AddPage(self.ups_panel, "Upscale")

        sizerRight = wx.StaticBoxSizer(wx.VERTICAL, self, label="Parameters")
        sizerRight.Add(self.notebook, proportion=1, flag=wx.ALL)
        sizerRight.Add(self.models_text, proportion=0, border=5, flag=wx.ALL)

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
        self.gen_panel.text_positive.Disable()
        self.gen_panel.text_negative.Disable()
        self.gen_panel.spinner_seed.Disable()
        self.gen_panel.spinner_denoise.Disable()
        self.ups_panel.text_positive.Disable()
        self.ups_panel.text_negative.Disable()
        self.ups_panel.spinner_seed.Disable()
        self.ups_panel.spinner_denoise.Disable()

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
                self.gen_panel.spinner_seed.SetValue(str(self.last_seed))
                e = Event()
                self.start_prompt(item, e=e)
                await e.wait()
                if upscaled:
                    self.status_text.SetLabel("Starting upscale...")
                    self.ups_panel.spinner_seed.SetValue(str(self.last_upscale_seed))
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

    def panel_to_notes(self, panel, notes, prefix, is_upscale=False):
        def replace_or_insert(notes, regex, value):
            prefix_regex = REGEX_PREFIX.format(prefix, regex)
            previx_value = f"{prefix} {value}"
            re_result = re.finditer(prefix_regex, notes, re.I)
            count = sum(1 for match in re_result if match.group(1))
            if count:
                notes = re.sub(prefix_regex, lambda match: match.group(0) if not match.group(1) else previx_value, notes, flags = re.I)
            else:
                notes = previx_value + notes
            return notes
        notes = replace_or_insert(notes, REGEX_POSITIVE, f"positive: {panel.text_positive.GetValue()}\n")
        notes = replace_or_insert(notes, REGEX_NEGATIVE, f"negative: {panel.text_negative.GetValue()}\n")
        notes = replace_or_insert(notes, REGEX_SEED, f"seed: {panel.spinner_seed.GetValue()}\n")
        notes = replace_or_insert(notes, REGEX_DENOISE, f"denoise: {panel.spinner_denoise.GetValue()}\n")
        notes = replace_or_insert(notes, REGEX_CFG, f"cfg: {panel.spinner_cfg.GetValue()}\n")
        notes = replace_or_insert(notes, REGEX_STEPS, f"steps: {panel.spinner_steps.GetValue()}\n")
        notes = replace_or_insert(notes, REGEX_ADD_NOISE, f"add noise: {panel.spinner_add_noise.GetValue()}\n")
        notes = replace_or_insert(notes, REGEX_SAMPLER, f"sampler: {panel.sampler.GetValue()}\n")
        # turbo and xl no clip
        firstTag = self.get_main_first_tag()
        if firstTag != MODEL_SD_TURBO_TAG and firstTag != MODEL_SD_XL_TAG:
            notes = replace_or_insert(notes, REGEX_CLIP, f"clip: {panel.spinner_clip.GetValue()}\n")
        # turbo no scheduler
        if firstTag != MODEL_SD_TURBO_TAG:
            notes = replace_or_insert(notes, REGEX_SCHEDULER, f"scheduler: {panel.scheduler.GetValue()}\n")
        # only lora need base model
        if firstTag == MODEL_SD_LORA_TAG:
            notes = replace_or_insert(notes, REGEX_LORA_BASE, f"lora base: {panel.lora_base.GetValue()}\n")
        if is_upscale:
            notes = replace_or_insert(notes, REGEX_UPSCALE_FACTOR, f"upscale factor: {panel.spinner_upscale_factor.GetValue()}\n")
        return notes

    async def OnSave(self, evt):
        main_item = self.items[0]
        notes = main_item.get("notes") or ""
        notes = self.panel_to_notes(self.ups_panel, notes, REGEX_UPS_PREFIX, True)
        notes = self.panel_to_notes(self.gen_panel, notes, REGEX_GEN_PREFIX)
        
        await self.app.api.update_lora(main_item["id"], {"notes": notes})
        await self.app.frame.results_panel.re_search()
        

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

    def get_prompt_options_for_panel(self, panel):
        positive = panel.text_positive.GetValue().strip().replace("\n", " ,")
        return GeneratePreviewsOptions(
            positive=positive,
            negative=panel.text_negative.GetValue(),
            seed=int(panel.spinner_seed.GetValue()),
            denoise=float(panel.spinner_denoise.GetValue()),
            cfg=int(panel.spinner_cfg.GetValue()),
            steps=int(panel.spinner_steps.GetValue()),
            clip=int(panel.spinner_clip.GetValue()),
            upscale_factor=float(panel.spinner_upscale_factor.GetValue()),
            add_noise=float(panel.spinner_add_noise.GetValue()),
            check_add_noise=bool(panel.check_add_noise.GetValue()),
            sampler=panel.sampler.GetValue(),
            scheduler=panel.scheduler.GetValue(),
            lora_base=panel.lora_base.GetValue(),
        )

    def assemble_prompt_data(self, item):
        checkpoint = item["filename"]
        tags = item["tags"]
        def process_options(options, inputs, initial_inputs):
            options.positive = inputs.positive if inputs.positive != initial_inputs.positive else options.positive
            options.negative = inputs.negative if inputs.negative != initial_inputs.negative else options.negative
            options.seed = inputs.seed if inputs.seed != initial_inputs.seed else options.seed
            if options.seed == -1:
                options.seed = random.randint(0, SEED_RANDOM_MAX)
            options.denoise = inputs.denoise if inputs.denoise != initial_inputs.denoise else options.denoise
            options.cfg = inputs.cfg if inputs.cfg != initial_inputs.cfg else options.cfg
            options.steps = inputs.steps if inputs.steps != initial_inputs.steps else options.steps
            options.clip = inputs.clip if inputs.clip != initial_inputs.clip else options.clip
            options.upscale_factor = inputs.upscale_factor if inputs.upscale_factor != initial_inputs.upscale_factor else options.upscale_factor
            options.add_noise = inputs.add_noise if inputs.add_noise != initial_inputs.add_noise else options.add_noise
            options.check_add_noise = inputs.check_add_noise
            options.sampler = inputs.sampler if inputs.sampler != initial_inputs.sampler else options.sampler
            options.scheduler = inputs.scheduler if inputs.scheduler != initial_inputs.scheduler else options.scheduler
            options.lora_base = inputs.lora_base if inputs.lora_base != initial_inputs.lora_base else options.lora_base
            return options
        gen_input = self.get_prompt_options_for_panel(self.gen_panel)
        gen_options = self.item_to_preview_options(item, REGEX_GEN_PREFIX)
        gen_options = process_options(gen_options, gen_input, self.gen_preview_options)
        ups_input = self.get_prompt_options_for_panel(self.ups_panel)
        ups_options = self.item_to_preview_options(item, REGEX_UPS_PREFIX, True)
        ups_options = process_options(ups_options, ups_input, self.ups_preview_options)
        data = PreviewPromptData(
            checkpoint = checkpoint,
            tags = tags,
            gen_options = gen_options,
            ups_options = ups_options,
        )
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
        self.last_seed = data.current_options.seed

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
        self.last_upscale_seed = data.current_options.seed
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

    def item_to_preview_options(self, itemsOrItems, prefix, is_upscale=False):
        item = itemsOrItems
        if isinstance(itemsOrItems, list):
            item = itemsOrItems[0]
        notes = item.get("notes") or ""
        def regex_get(regex, prefix, notes, default):
            prefix_regex = regex
            if prefix:
                prefix_regex = REGEX_PREFIX.format(prefix, regex)
            re_prefix_result = re.finditer(prefix_regex, notes, re.I)
            re_prefix = next((match for match in re_prefix_result if match.group(1)), None)
            if re_prefix:
                return re_prefix.group(2).strip()
            re_noprefix_result = re.finditer(prefix_regex, notes, re.I)
            re_noprefix = next((match for match in re_noprefix_result if not match.group(1)), None)
            if re_noprefix:
                return re_noprefix.group(2).strip()
            return default
        def get_default_prompt(keywords_key, regex, default):
            prompt = regex_get(regex, prefix, notes, default)
            keywords = item[keywords_key]
            if not keywords:
                return prompt
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
        seed = int(regex_get(REGEX_SEED, prefix, notes, DEFAULT_SEED))
        # build denoise
        denoise = float(regex_get(REGEX_DENOISE, prefix, notes, DEFAULT_DENOISE if not is_upscale else DEFAULT_UPSCALE_DENOISE))
        # build cfg
        cfg = int(regex_get(REGEX_CFG, prefix, notes, DEFAULT_CFG))
        # build steps
        steps = int(regex_get(REGEX_STEPS, prefix, notes, DEFAULT_STEPS))
        # build clip
        clip = int(regex_get(REGEX_CLIP, prefix, notes, DEFAULT_CLIP))
        # build upscale factor
        upscale_factor = float(regex_get(REGEX_UPSCALE_FACTOR, prefix, notes, DEFAULT_UPSCALE_FACTOR))
        # build add noise
        add_noise = float(regex_get(REGEX_ADD_NOISE, prefix, notes, DEFAULT_ADD_NOISE))
        # build sampler
        sampler = regex_get(REGEX_SAMPLER, prefix, notes, DEFAULT_SAMPLER)
        # build scheduler
        scheduler = regex_get(REGEX_SCHEDULER, prefix, notes, DEFAULT_SCHEDULER)
        # build lora base
        lora_base = regex_get(REGEX_LORA_BASE, prefix, notes, DEFAULT_LORA_BASE)
        previewPrompOptions = GeneratePreviewsOptions(
            positive = positive,
            negative = negative,
            seed = seed,
            denoise = denoise,
            cfg = cfg,
            steps = steps,
            clip = clip,
            upscale_factor = upscale_factor,
            add_noise = add_noise,
            check_add_noise = add_noise != 0,
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
