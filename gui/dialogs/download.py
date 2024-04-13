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
from PIL import Image

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
DEFAULT_CFG = 8
DEFAULT_STEPS = 20
DEFAULT_CLIP = -1
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_SCHEDULER = "normal"
DEFAULT_LORA_BASE = "AOM3.safetensors"

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
    cfg: int
    steps: int
    clip: int
    sampler: str
    scheduler: str
    lora_base: str


@dataclass
class PreviewPromptData:
    seed: int
    denoise: float
    checkpoint: str
    positive: str
    negative: str
    cfg: int
    steps: int
    clip: int
    sampler: str
    scheduler: str
    lora_base: str
    tags: str

    def to_prompt(self):
        firstTag = ""
        if self.tags:
            firstTag = self.tags.split(",")[0].strip()
        keywordToFunc = {
            "sd-1.5": self.to_prompt_default,
            "sd-xl": self.to_prompt_xl,
            "sd-turbo": self.to_prompt_turbo,
            "sd-merge-turbo": self.to_prompt_merge_turbo,
            "sd-lora": self.to_prompt_lora,
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
        prompt = load_prompt("hr.json")
        filename = image["filename"]
        prompt["11"]["inputs"]["seed"] = self.seed
        prompt["11"]["inputs"]["denoise"] = self.denoise
        prompt["16"]["inputs"]["ckpt_name"] = self.checkpoint
        prompt["17"]["inputs"]["vae_name"] = self.vae
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["7"]["inputs"]["text"] = self.negative
        prompt["18"]["inputs"]["image"] = f"{filename} [output]"
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

class PreviewGeneratorDialog(wx.Dialog):
    def __init__(self, parent, app, items, duplicate_op):
        super(PreviewGeneratorDialog, self).__init__(
            parent, -1, "Preview Generator", size=app.FromDIP(700, 500)
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
        self.last_seed = 0
        self.last_upscale_seed = 0
        self.node_text = ""

        utils.set_icons(self)
        self.autogen = False

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
        self.models_text = wx.StaticText(
            self, wx.ID_ANY, label=f"Selected models: {len(self.items)}"
        )
        self.gauge = wx.Gauge(self, -1, 100, size=app.FromDIP(800, 32))
        self.image_panel = ImagePanel(
            self, style=wx.SUNKEN_BORDER, size=app.FromDIP(512, 512)
        )
        self.button_regenerate = wx.Button(self, wx.ID_HELP, "Generate")
        self.button_upscale = wx.Button(self, wx.ID_APPLY, "Upscale")
        self.button_upscale.Disable()
        self.button_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        self.button_ok = wx.Button(self, wx.ID_OK, "OK")
        self.button_ok.Disable()

        self.Bind(wx.EVT_BUTTON, self.OnRegenerate, id=wx.ID_HELP)
        self.Bind(wx.EVT_BUTTON, self.OnUpscale, id=wx.ID_APPLY)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, id=wx.ID_CANCEL)
        wxasync.AsyncBind(wx.EVT_BUTTON, self.OnOK, self.button_ok, id=wx.ID_OK)
        wxasync.AsyncBind(wx.EVT_CLOSE, self.OnClose, self)

        sizerB = wx.StdDialogButtonSizer()
        sizerB.AddButton(self.button_regenerate)
        sizerB.AddButton(self.button_upscale)
        sizerB.AddButton(self.button_cancel)
        sizerB.AddButton(self.button_ok)
        sizerB.Realize()

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
        self.spinner_seed = wx.TextCtrl(self, wx.ID_ANY, value=str(self.preview_options.seed), size=self.Parent.FromDIP(wx.Size(140, 25)))
        sizerRightAfter.Add(self.spinner_seed, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter.Add(
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
            size=self.Parent.FromDIP(wx.Size(140, 25)),
        )
        self.spinner_denoise.SetFormat("%f")
        self.spinner_denoise.SetDigits(2)
        sizerRightAfter.Add(self.spinner_denoise, proportion=1, flag=wx.ALL, border=5)

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
            size=self.Parent.FromDIP(wx.Size(150, 25)),
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
            size=self.Parent.FromDIP(wx.Size(150, 25)),
        )
        sizerRightAfter2.Add(self.spinner_steps, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter3 = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfter3.Add(
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
            size=self.Parent.FromDIP(wx.Size(140, 25)),
        )
        sizerRightAfter3.Add(self.spinner_clip, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter4 = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfter4.Add(
            wx.StaticText(self, wx.ID_ANY, label="Sampler"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        choices = ["euler", "euler_ancestral", "dpmpp_3m_sde"]
        self.sampler = wx.ComboBox(self, id=wx.ID_ANY, value=self.preview_options.sampler, choices=choices)
        sizerRightAfter4.Add(self.sampler, proportion=1, flag=wx.ALL, border=5)
        sizerRightAfter4.Add(
            wx.StaticText(self, wx.ID_ANY, label="Scheduler"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        choices = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
        self.scheduler = wx.ComboBox(self, id=wx.ID_ANY, value=self.preview_options.scheduler, choices=choices)
        sizerRightAfter4.Add(self.scheduler, proportion=1, flag=wx.ALL, border=5)

        sizerRightAfter5 = wx.BoxSizer(wx.HORIZONTAL)
        sizerRightAfter5.Add(
            wx.StaticText(self, wx.ID_ANY, label="Lora Base"),
            proportion=0,
            border=5,
            flag=wx.ALL,
        )
        choices = ["AOM3.safetensors", "Anything-V3.0-pruned-fp16.ckpt", "v1-5-pruned.ckpt"]
        self.lora_base = wx.ComboBox(self, id=wx.ID_ANY, value=self.preview_options.lora_base, choices=choices)
        sizerRightAfter5.Add(self.lora_base, proportion=1, flag=wx.ALL, border=5)


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

    async def save_preview_image(self, item, result):
        self.app.SetStatusText("Saving preview...")
        self.status_text.SetLabel("Saving preview...")

        image_data = self.comfy_api.get_image(
            result["filename"], result["subfolder"], result["type"]
        )
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
                thread = self.start_prompt(item, e=e)
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
            int(self.spinner_cfg.GetValue()),
            int(self.spinner_steps.GetValue()),
            int(self.spinner_clip.GetValue()),
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
            seed = random.randint(0, 2**32)
        denoise = inputOptions.denoise if inputOptions.denoise != self.preview_options.denoise else itemOptions.denoise
        cfg = inputOptions.cfg if inputOptions.cfg != self.preview_options.cfg else itemOptions.cfg
        steps = inputOptions.steps if inputOptions.steps != self.preview_options.steps else itemOptions.steps
        clip = inputOptions.clip if inputOptions.clip != self.preview_options.clip else itemOptions.clip
        sampler = inputOptions.sampler if inputOptions.sampler != self.preview_options.sampler else itemOptions.sampler
        scheduler = inputOptions.scheduler if inputOptions.scheduler != self.preview_options.scheduler else itemOptions.scheduler
        loraBase = inputOptions.lora_base if inputOptions.lora_base != self.preview_options.lora_base else itemOptions.lora_base
        tags = item["tags"]
        data = PreviewPromptData(
            seed, 
            denoise, 
            checkpoint, 
            positive, 
            negative, 
            cfg,
            steps,
            clip,
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
        re_notes_positive = re.search(r"positive:(.+)\n", notes)
        notes_positive = re_notes_positive.group(1).strip() if re_notes_positive else DEFAULT_POSITIVE
        positive = notes_positive.strip()
        posKey = item["keywords"]
        if posKey:
            if posKey.startswith(OVERRIDE_KEYWORD):
                positive = ""
                posKey = posKey.replace(OVERRIDE_KEYWORD, "")
            else:
                posKey = f", {posKey}"
            positive += posKey
        # build negative prompt
        re_notes_negative = re.search(r"negative:(.+)\n", notes)
        notes_negative = re_notes_negative.group(1).strip() if re_notes_negative else DEFAULT_NEGATIVE
        negative = notes_negative.strip()
        negKey = item["negative_keywords"]
        if negKey:
            if negKey.startswith(OVERRIDE_KEYWORD):
                negative = ""
                negKey = negKey.replace(OVERRIDE_KEYWORD, "")
            else:
                negKey = f", {negKey}"
            negative += negKey
        # build seed
        re_notes_seed = re.search(r"seed:\s*(\d+)", notes, re.I)
        seed = int(re_notes_seed.group(1).strip()) if re_notes_seed else DEFAULT_SEED
        # build denoise
        re_notes_denoise = re.search(r"denoise:\s*(\d+\.?\d*)", notes, re.I)
        denoise = float(re_notes_denoise.group(1).strip()) if re_notes_denoise else DEFAULT_DENOISE
        # build cfg
        re_notes_cfg = re.search(r"cfg:\s*(\d+)", notes, re.I)
        cfg = int(re_notes_cfg.group(1).strip()) if re_notes_cfg else DEFAULT_CFG
        # build steps
        re_notes_steps = re.search(r"steps:\s*(\d+)", notes, re.I)
        steps = int(re_notes_steps.group(1).strip()) if re_notes_steps else DEFAULT_STEPS
        # build clip
        re_notes_clip = re.search(r"clip:\s*(-?\s*\d+)", notes, re.I)
        clip = int(re_notes_clip.group(1).strip()) if re_notes_clip else DEFAULT_CLIP
        # build sampler
        re_notes_sampler = re.search(r"sampler:\s*([\w\.\-_]+)", notes, re.I)
        sampler = re_notes_sampler.group(1).strip() if re_notes_sampler else DEFAULT_SAMPLER
        # build scheduler
        re_notes_scheduler = re.search(r"scheduler:\s*([\w\.\-_]+)", notes, re.I)
        scheduler = re_notes_scheduler.group(1).strip() if re_notes_scheduler else DEFAULT_SCHEDULER
        # build lora base
        re_notes_lora_base = re.search(r"lora[_\-\s]*base:\s*([\w\.\-_]+)", notes, re.I)
        lora_base = re_notes_lora_base.group(1).strip() if re_notes_lora_base else DEFAULT_LORA_BASE
        previewPrompOptions = GeneratePreviewsOptions(
            positive,
            negative,
            seed,
            denoise,
            cfg,
            steps,
            clip,
            sampler,
            scheduler,
            lora_base
        )
        return previewPrompOptions


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
