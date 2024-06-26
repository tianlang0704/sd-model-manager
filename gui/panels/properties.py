import os
import aiopubsub
from aiopubsub import Key

import wx
import wxasync
import wx.aui
import wx.lib.newevent
import wx.lib.mixins.listctrl as listmix

from gui.rating_ctrl import RatingCtrl, EVT_RATING_CHANGED
from gui.utils import PUBSUB_HUB

MODEL_SD_MERGE_TURBO_TAG = "sd-merge-turbo"
MODEL_SD_TURBO_TAG = "sd-turbo"
MODEL_SD_XL_TAG = "sd-xl"
MODEL_SD_15_TAG = "sd-1.5"
MODEL_SD_14_TAG = "sd-1.4"
MODEL_SD_LORA_TAG = "sd-lora"
MODEL_SD_VAE_TAG = "sd-vae"
MODEL_SD_EMBEDDING_TAG = "sd-embedding"
MODEL_SD_CONTROLNET_TAG = "sd-controlnet"
MODEL_SD_INPAINT_TAG = "sd-inpaint"

taglist = [
    MODEL_SD_MERGE_TURBO_TAG,
    MODEL_SD_TURBO_TAG,
    MODEL_SD_XL_TAG,
    MODEL_SD_15_TAG,
    MODEL_SD_14_TAG,
    MODEL_SD_LORA_TAG,
    MODEL_SD_VAE_TAG,
    MODEL_SD_EMBEDDING_TAG,
    MODEL_SD_CONTROLNET_TAG,
    MODEL_SD_INPAINT_TAG,
]

class PropertiesPanel(wx.lib.scrolledpanel.ScrolledPanel):
    def __init__(self, parent, app=None):
        self.app = app
        self.roots = []

        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, id=wx.ID_ANY, style=wx.VSCROLL)

        self.sub = aiopubsub.Subscriber(PUBSUB_HUB, Key("events"))
        self.sub.add_async_listener(Key("events", "item_selected"), self.SubItemSelected)
        self.sub.add_async_listener(Key("events", "tree_filter_changed"), self.SubTreeFilterChanged)

        self.ctrls = {}
        self.selected_items = []
        self.changes = {}
        self.original_values = {}
        self.values = {}
        self.is_committing = False
        self.change_handlers = {}

        # bind escape
        def OnCharHook(evt):
            if evt.GetKeyCode() == wx.WXK_ESCAPE:
                self.restore_changes()
            evt.Skip()
        self.Bind(wx.EVT_CHAR_HOOK, OnCharHook)

        ctrls = [
            ("display_name", "Name", None, None),
            ("version", "Version", None, None),
            ("author", "Author", None, None),
            ("source", "Source", None, None),
            ("tags", "Tags", None, None),
            ("keywords","Keywords", wx.TE_MULTILINE, self.Parent.FromDIP(wx.Size(250, 40)),),
            ("negative_keywords","Negative Keywords", wx.TE_MULTILINE|wx.TE_BESTWRAP, self.Parent.FromDIP(wx.Size(250, 40)),),
            ("description","Description", wx.TE_MULTILINE, self.Parent.FromDIP(wx.Size(250, 150)),),
            ("notes", "Notes", wx.TE_MULTILINE, self.Parent.FromDIP(wx.Size(250, 180))),
        ]

        for key, label, style, size in ctrls:
            if style is not None:
                ctrl = wx.TextCtrl(self, id=wx.ID_ANY, size=size, value="", style=style)
            elif "tags" == key:
                choices = ["< blank >", "< keep >"]
                choices.extend(taglist)
                ctrl = wx.ComboBox(self, id=wx.ID_ANY, value="", choices=choices)
            else:
                choices = ["< blank >", "< keep >"]
                ctrl = wx.ComboBox(self, id=wx.ID_ANY, value="", choices=choices)
            self.ctrls[key] = (ctrl, wx.StaticText(self, label=label))

        def handler(key, label, evt):
            value = evt.GetString()
            self.modify_value(key, label, value)
        for key, (ctrl, label) in self.ctrls.items():
            change_handler = lambda evt, key=key, label=label: handler(key, label, evt)
            self.change_handlers[key] = change_handler
            ctrl.Bind(wx.EVT_TEXT, change_handler)

        self.label_filename = wx.StaticText(self, label="Filename")
        self.label_rating = wx.StaticText(self, label="Rating")
        self.text_filename = wx.TextCtrl(self)
        self.text_filename.SetEditable(False)
        self.ctrl_rating = RatingCtrl(self)
        self.text_id = wx.TextCtrl(self)
        self.text_id.SetEditable(False)
        self.text_filepath = wx.TextCtrl(self)
        self.text_filepath.SetEditable(False)
        self.other_ctrls = {}
        self.other_ctrls["filename"] = self.text_filename
        self.other_ctrls["id"] = self.text_id
        self.other_ctrls["filepath"] = self.text_filepath
        self.other_ctrls["rating"] = self.ctrl_rating

        self.ctrl_rating.Bind(EVT_RATING_CHANGED, lambda evt: self.modify_value("rating", self.label_rating, evt.rating))
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.label_filename, flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.text_filename, flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["display_name"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["display_name"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["author"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["author"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["version"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["version"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["source"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["source"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["tags"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["tags"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["keywords"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["keywords"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["negative_keywords"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["negative_keywords"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.label_rating, flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrl_rating, flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["description"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["description"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(self.ctrls["notes"][1], flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.ctrls["notes"][0], flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(wx.StaticText(self, label="ID"), flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.text_id, flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.sizer.Add(wx.StaticText(self, label="Filepath"), flag=wx.ALL | wx.ALIGN_TOP, border=2)
        self.sizer.Add(self.text_filepath, flag=wx.ALL | wx.EXPAND | wx.ALIGN_TOP, border=5)
        self.SetSizerAndFit(self.sizer)
        self.SetupScrolling(scroll_x=False)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def modify_value(self, key, label, value):
        if key in self.values and value != self.values[key]:
            self.app.frame.toolbar.EnableTool(wx.ID_SAVE, True)
            self.app.frame.toolbar.EnableTool(wx.ID_CLEAR, True)
            self.original_values[key] = self.values[key] if key not in self.original_values else self.original_values[key]
            self.changes[key] = value
            text = label.GetLabel()
            if not text.endswith("*"):
                label.SetLabel(text + "*")
        self.values[key] = value

    def OnSize(self, evt):
        size = self.GetSize()
        vsize = self.GetVirtualSize()
        self.SetVirtualSize((size[0], vsize[1]))

        evt.Skip()

    def restore_changes(self):
        self.clear_changes()
        for key, value in self.original_values.items():
            if key in self.ctrls:
                self.ctrls[key][0].ChangeValue(value)
            elif key in self.other_ctrls:
                self.other_ctrls[key].ChangeValue(value)
        self.original_values = {}
        
    def clear_changes(self):
        self.values = {}
        for key, (ctrl, label) in self.ctrls.items():
            label.SetLabel(label.GetLabel().rstrip("*"))
            self.values[key] = ctrl.GetValue()

        self.label_rating.SetLabel(self.label_rating.GetLabel().rstrip("*"))
        self.values["rating"] = self.ctrl_rating.GetValue()

        self.changes = {}

        self.app.frame.toolbar.EnableTool(wx.ID_SAVE, False)
        self.app.frame.toolbar.EnableTool(wx.ID_CLEAR, False)

    async def commit_changes(self, isForce = False):
        if isForce:
            self.is_committing = False
        if not self.changes or not self.selected_items or self.is_committing:
            return

        self.is_committing = True

        changes = {}

        for ctrl_name, new_value in self.changes.items():
            if new_value == "< blank >":
                new_value = ""
            elif new_value == "< keep >":
                continue

            if ctrl_name in self.ctrls:
                changes[ctrl_name] = new_value
            elif ctrl_name in self.other_ctrls:
                changes[ctrl_name] = new_value

            for item in self.selected_items:
                item[ctrl_name] = new_value

        if not changes:
            self.clear_changes()
            return

        count = 0
        updated = 0
        progress = wx.ProgressDialog(
            "Saving",
            f"Saving changes... ({0}/{len(self.selected_items)})",
            parent=self.app.frame,
            maximum=len(self.selected_items),
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE,
        )

        for item in self.selected_items:
            result = await self.app.api.update_lora(item["id"], changes)
            count += 1
            updated += result["fields_updated"]
            progress.Update(
                count, f"Saving changes... ({count}/{len(self.selected_items)})"
            )

            await self.app.frame.results_panel.refresh_one_item(item)

        progress.Destroy()
        self.app.SetStatusText(f"Updated {updated} fields")

        self.is_committing = False

        self.clear_changes()

    async def prompt_commit_changes(self):
        self.original_values = {}
        if not self.selected_items:
            self.changes = {}
            return

        if not self.changes or self.is_committing:
            return

        PROMPT = False  # TODO
        if self.changes:
            if PROMPT:
                dlg = wx.MessageDialog(
                    None,
                    "You have unsaved changes, would you like to commit them?",
                    "Updater",
                    wx.YES_NO | wx.ICON_QUESTION,
                )
                result = await wxasync.AsyncShowDialogModal(dlg)
                if result == wx.ID_YES:
                    await self.commit_changes()
                    self.original_values = {}
            else:
                await self.commit_changes()

    async def SubItemSelected(self, key, items):
        if items == self.selected_items:
            return
        await self.prompt_commit_changes()
        self.selected_items = items
        if len(self.selected_items) == 0:
            for (ctrl, label) in self.ctrls.values():
                ctrl.SetEditable(False)
                ctrl.Clear()
            for ctrl in self.other_ctrls.values():
                ctrl.Clear()
        else:
            for name, (ctrl, label) in self.ctrls.items():
                ctrl.SetEditable(True)
                choices = ["< blank >", "< keep >"]
                if (name == "tags"):
                    choices.extend(taglist)

                def get_value(item):
                    value = item.get(name, "< blank >")
                    if value is None or value == "":
                        value = "< blank >"
                    return value
                value = None
                if len(items) == 1:
                    value = get_value(items[0])
                    if value not in choices:
                        choices.append(value)
                else:
                    for item in items:
                        if value != "< keep >":
                            item_value = get_value(item)
                            if value is None:
                                value = item_value
                            elif value != item_value:
                                value = "< keep >"
                        choice = get_value(item)
                        if choice not in choices:
                            choices.append(choice)

                is_combo_box = hasattr(ctrl, "SetItems")
                if is_combo_box:
                    # SetItems triggers EVT_TEXT because of history reasons, so we need to temporarily bind a dummy handler
                    def dummy_handler(evt):
                        pass
                    ctrl.Bind(wx.EVT_TEXT, dummy_handler)
                    ctrl.SetItems(choices)
                    ctrl.Bind(wx.EVT_TEXT, self.change_handlers[name])
                else:
                    if value == "< blank >":
                        value = ""
                ctrl.ChangeValue(value)

            if len(items) == 1:
                filename = os.path.basename(items[0]["filepath"])
                filepath = items[0]["filepath"]
                id = str(items[0]["id"])
                rating = items[0].get("rating") or 0
                self.ctrl_rating.ChangeValue(rating)
            else:
                filename = "< multiple >"
                filepath = "< multiple >"
                id = "< multiple >"
                rating = None
                for item in items:
                    nrating = item.get("rating") or 0
                    if rating is None:
                        rating = nrating
                        self.ctrl_rating.ChangeValue(rating)
                    elif rating != nrating:
                        self.ctrl_rating.SetMultiple()
                        break
            self.text_filename.ChangeValue(filename)
            self.text_id.ChangeValue(id)
            self.text_filepath.ChangeValue(filepath)

        self.clear_changes()

    async def SubTreeFilterChanged(self, key, sel):
        pass