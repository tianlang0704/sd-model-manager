import os
from typing import Callable, Optional

import wx
import wx.aui
import wx.lib.newevent
import wxasync

from sd_model_manager.prompt import infotext
from gui import utils


class PopupMenuSeparator:
    pass


class PopupMenuItem:
    title: str
    callback: Callable
    enabled: bool
    is_async: bool
    checked: Optional[bool]
    bitmap: Optional[wx.Bitmap]

    def __init__(
        self, title, callback, enabled=True, is_async=False, checked=None, icon=None
    ):
        self.title = title
        self.callback = callback
        self.enabled = enabled
        self.is_async = is_async
        self.checked = checked
        self.icon = icon


class PopupMenu(wx.Menu):
    def __init__(self, *args, target=None, event=None, items=None, app=None, object=None, **kwargs):
        wx.Menu.__init__(self, *args, **kwargs)
        self.target = target
        self.event = event
        self.items = {}
        self.order = []
        self.app = app

        for item in items:
            id = wx.NewIdRef(count=1)
            self.order.append(id)
            self.items[id] = item

        for id in self.order:
            item = self.items[id]

            if isinstance(item, PopupMenuSeparator):
                self.AppendSeparator()
            else:
                if item.checked is not None:
                    self.AppendCheckItem(id, item.title)
                    self.Check(id, item.checked)
                else:
                    self.Append(id, item.title)
                    if item.icon is not None:
                        menu_item, _menu = self.FindItem(id)
                        menu_item.SetBitmap(item.icon)
                self.Enable(id, item.enabled)
            if object is None:
                object = self.app.frame
            wxasync.AsyncBind(wx.EVT_MENU, self.OnMenuSelection, object, id=id)

    async def OnMenuSelection(self, event):
        item = self.items[event.GetId()]
        if item.is_async:
            await item.callback(self.target, self.event)
        else:
            item.callback(self.target, self.event)


def open_folder(target, event):
    path = target["filepath"]
    utils.open_on_file(path)


def copy_item_value(target, event, app):
    column = utils.COLUMNS[event.GetColumn()]
    value = column.callback(target)
    copy_to_clipboard(value, app)


def copy_to_clipboard(value, app=None):
    if wx.TheClipboard.Open():
        wx.TheClipboard.SetData(wx.TextDataObject(str(value)))
        wx.TheClipboard.Close()

        if app:
            app.frame.statusbar.SetStatusText(f"Copied: {utils.trim_string(value)}")

def create_popup_menu_for_item(target, evt, app):
    tag_freq = target.get("tag_frequency")

    image_prompt = None
    image_tags = None
    image_neg_tags = None
    image, image_path = utils.find_image_for_model(target)
    if image is not None:
        if "parameters" in image.info:
            image_prompt = image.info["parameters"]
            image_tags = infotext.parse_a1111_prompt(image_prompt)
        elif "prompt" in image.info:
            image_prompt = image.info["prompt"]
            image_tags = infotext.parse_comfyui_prompt(image_prompt)
        if image_tags is not None:
            image_neg_tags = next(
                iter(i for i in image_tags if i.startswith("negative:")), "negative:"
            ).lstrip("negative:")
            image_tags = infotext.remove_metatags(image_tags)
            image_tags = ", ".join([t for t in image_tags])

    icon_copy = utils.load_bitmap("images/icons/16/page_copy.png")
    icon_folder_go = utils.load_bitmap("images/icons/16/folder_go.png")
    icon_picture_add = utils.load_bitmap("images/icons/16/picture_add.png")
    icon_picture_delete = utils.load_bitmap("images/icons/16/picture_delete.png")
    items = [
        PopupMenuItem("Open Folder", open_folder, icon=icon_folder_go),
        PopupMenuItem("Copy Value", lambda t, e: copy_item_value(t, e, app)),
        PopupMenuSeparator(),
        PopupMenuItem(
            "Show Metadata...",
            lambda t, e: app.frame.OnShowMetadata(None),
            is_async=True,
            icon=icon_folder_go,
        ),
        PopupMenuSeparator(),
        PopupMenuItem(
            "Generate Previews(Replace)...",
            lambda t, e: app.frame.OnGeneratePreviews(None, op="replace"),
            is_async=True,
            icon=icon_picture_add,
        ),
        PopupMenuItem(
            "Generate Previews(Append)...",
            lambda t, e: app.frame.OnGeneratePreviews(None, op="append"),
            is_async=True,
            icon=icon_picture_add,
        ),
        PopupMenuSeparator(),
        PopupMenuItem(
            "Copy Preview Image Prompt",
            lambda t, e: copy_to_clipboard(image_prompt, app),
            enabled=image_prompt is not None,
            icon=icon_copy,
        ),
        PopupMenuSeparator(),
        PopupMenuItem(
            "Remove Data",
            lambda t, e: app.frame.OnRemoveData(None),
            is_async=True,
            icon=icon_picture_delete,
        ),
    ]
    items = [i for i in items if i]

    return PopupMenu(target=target, event=evt, items=items, app=app)
