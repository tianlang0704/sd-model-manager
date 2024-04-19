import os
import re
import aiopubsub
from PIL import Image
from aiopubsub import Key

import wx
import wx.aui
import wx.lib.newevent
# from wx.lib.agw import ultimatelistctrl
from gui import ultimatelistctrl
import wxasync

from sd_model_manager.utils.common import try_load_image, PATH
from gui.scrolledthumbnail import (
    ScrolledThumbnail,
    Thumb,
    PILImageHandler,
    file_broken,
    EVT_THUMBNAILS_SEL_CHANGED,
    EVT_THUMBNAILS_DCLICK,
    EVT_THUMBNAILS_RCLICK,
)
from gui.dialogs.metadata import MetadataDialog
from gui.utils import PUBSUB_HUB, COLUMNS, find_image_path_for_model
from gui.popup_menu import PopupMenu, PopupMenuItem, create_popup_menu_for_item
import simplejson

SEARCH_SAVE_FILE = os.path.join(PATH, "search.json")
SAVE_NAME_REGEX = r"^button.*name:\s*([\w\-]+)\s+"
SAVE_NAME_DEFAULT = "Saved"

class ResultsListCtrl(ultimatelistctrl.UltimateListCtrl):
    def __init__(self, parent, app=None):
        ultimatelistctrl.UltimateListCtrl.__init__(
            self,
            parent,
            -1,
            agwStyle=ultimatelistctrl.ULC_VIRTUAL
            | ultimatelistctrl.ULC_REPORT
            | wx.LC_HRULES
            | wx.LC_VRULES
            | ultimatelistctrl.ULC_SHOW_TOOLTIPS,
        )

        self.app = app

        self.results = []
        self.values = {}
        self.filtered = []
        self.filter = None
        self.clicked = False
        self.primary_item = None
        self.all_names = set()

        # EVT_LIST_ITEM_SELECTED and ULC_VIRTUAL don't mix
        # https://github.com/wxWidgets/wxWidgets/issues/4541
        self.Bind(wx.EVT_LIST_CACHE_HINT, self.OnListItemSelected)
        wxasync.AsyncBind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnListItemActivated, self)
        self.Bind(wx.EVT_LIST_DELETE_ALL_ITEMS, self.OnListItemSelected)
        wxasync.AsyncBind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self.OnListItemRightClicked, self)
        self.Bind(wx.EVT_LIST_COL_RIGHT_CLICK, self.OnColumnRightClicked)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_KEY_UP, self.OnKeyUp)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnClick)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnClick)

        self.pub = aiopubsub.Publisher(PUBSUB_HUB, Key("events"))
        self.sub = aiopubsub.Subscriber(PUBSUB_HUB, Key("events"))
        self.sub.add_async_listener(Key("events", "tree_filter_changed"), self.SubTreeFilterChanged)

        for col, column in enumerate(COLUMNS):
            self.InsertColumn(col, column.name)
            self.SetColumnShown(col, column.is_visible)
            if column.width is not None:
                width = self.Parent.FromDIP(column.width)
            else:
                width = wx.LIST_AUTOSIZE_USEHEADER
            self.SetColumnWidth(col, width)

    async def SubTreeFilterChanged(self, key, path):
        self.pub.publish(Key("item_selected"), [])

    def ClearSelection(self):
        i = self.GetFirstSelected()
        while i != -1:
            self.Select(i, 0)
            i = self.GetNextSelected(i)

    def OnKeyUp(self, evt):
        self.clicked = True
        evt.Skip()

    def OnKeyDown(self, evt):
        self.clicked = True
        evt.Skip()

    def get_all_names(self):
        return list(self.all_names)

    def record_all_names(self, results):
        for item in results["data"]:
            self.all_names.add(item["filename"])

    def set_results(self, results):
        self.results = results
        self.record_all_names(results)

    def select_default(self):
        if len(self.filtered) <= 0:
            return
        self.Select(0, 1)
        self.Focus(0)
        self.pub.publish(Key("item_selected"), self.get_selection())

    def restore_selection(self, selection, scrollY=None):
        selected_index = [self.index_from_id(item["id"]) for item in selection]
        selected_index = [i for i in selected_index if i is not None]
        if len(selected_index) <= 0:
            return False
        self.ClearSelection()
        for index in selected_index:
            self.Select(index, 1)
        self.Focus(selected_index[0])
        if scrollY is not None:
            self._mainWin.Scroll(-1, scrollY)
            self._mainWin.ResetVisibleLinesRange()
        selection = self.get_selection()
        self.pub.publish(Key("item_selected"), selection)
        return True

    def refresh_filtered_data(self):
        data = self.results["data"]
        self.filtered = []
        if self.filter is None:
            self.filtered = data
        else:
            for d in data:
                p = d["filepath"]
                if p.startswith(self.filter):
                    self.filtered.append(d)
        
    def refresh_text_view(self):
        self.app.frame.statusbar.SetStatusText("Loading results...")
        self.values = {}
        for i, data in enumerate(self.filtered):
            self.refresh_one_value(data, i)

        self.DeleteAllItems()
        count = len(self.filtered)
        self.SetItemCount(count)
        self.Refresh()
        self.app.frame.statusbar.SetStatusText(f"Done. ({count} records)")

    def index_from_id(self, id):
        return next((i for i, element in enumerate(self.filtered) if element["id"] == id), None)
    def refresh_one_value(self, data, index = None):
        if index is None:
            index = self.index_from_id(data["id"])
        if index is None:
            return
        for col, column in enumerate(COLUMNS):
            value = column.callback(data)
            if col not in self.values:
                self.values[col] = {}
            self.values[col][index] = value

    def get_selection(self):
        item = self.GetFirstSelected()
        num = self.GetSelectedItemCount()
        if num == 0:
            return []
        selection = [item]
        for i in range(1, num):
            item = self.GetNextSelected(item)
            selection.append(item)
        selection = [self.filtered[i] for i in selection]

        if len(selection) == 0:
            self.primary_item = None

        if self.primary_item is not None:
            if self.primary_item in selection:
                selection.insert(0, selection.pop(selection.index(self.primary_item)))
        return selection

    def OnGetItemColumnImage(self, item, col):
        return []

    def OnGetItemImage(self, item):
        return []

    def OnGetItemAttr(self, item):
        return None

    def OnGetItemText(self, item, col):
        entry = self.values[col][item]
        if entry is None:
            column = COLUMNS[col]
            if not column.is_meta:
                return "(None)"

        return str(entry)

    def OnGetItemTextColour(self, item, col):
        entry = self.values[col][item]
        if entry is None:
            return "gray"
        if COLUMNS[col].name == "Filepath" and not os.path.isfile(entry):
            return "red"
        return None

    def OnGetItemToolTip(self, item, col):
        return None

    def OnGetItemKind(self, item):
        return 0

    def OnGetItemColumnKind(self, item, col):
        return 0

    def OnClick(self, evt):
        self.clicked = True
        evt.Skip()

    def OnListItemSelected(self, evt):
        if not self.clicked:
            return
        self.clicked = False
        selection = self.get_selection()
        self.pub.publish(Key("item_selected"), selection)

    async def OnListItemActivated(self, evt):
        # await self.app.frame.OnShowMetadata(None)
        await self.app.frame.OnGeneratePreviews(None, op="replace")

    async def OnListItemRightClicked(self, evt):
        idx = evt.GetIndex()
        self.Select(idx)
        self.Focus(idx)
        self.primary_item = self.filtered[idx]
        selection = self.get_selection()
        await self.app.frame.ForceSelect(selection)

        target = self.filtered[evt.GetIndex()]

        menu = create_popup_menu_for_item(target, evt, self.app)

        pos = evt.GetPoint()
        self.PopupMenu(menu, pos)
        menu.Destroy()

    def OnColumnRightClicked(self, evt):
        items = []
        for i, col in enumerate(COLUMNS):
            def check(target, event, col=col, i=i):
                col.is_visible = not col.is_visible
                self.SetColumnShown(i, col.is_visible)
            items.append(PopupMenuItem(col.name, check, checked=col.is_visible))    
        menu = PopupMenu(target=self, items=items, app=self.app)
        pos = evt.GetPoint()
        self.PopupMenu(menu, pos)
        menu.Destroy()

    def MousePosToCol(self, pos):
        pos_with_scroll = self._mainWin.CalcUnscrolledPosition(pos)
        local_pos = self.ScreenToClient(pos_with_scroll)
        x = local_pos[0]
        for i, col in enumerate(COLUMNS):
            if not col.is_visible:
                continue
            col_width = self.GetColumnWidth(i)
            if x < col_width:
                return i
            x -= col_width
        return None


class GalleryThumbnailHandler(PILImageHandler):
    def Load(self, filename):
        try:
            with Image.open(filename) as pil:
                originalsize = pil.size

                pil.thumbnail((768, 768), Image.Resampling.LANCZOS)

                img = wx.Image(pil.size[0], pil.size[1])

                img.SetData(pil.convert("RGB").tobytes())

                alpha = False
                if "A" in pil.getbands():
                    img.SetAlpha(pil.convert("RGBA").tobytes()[3::4])
                    alpha = True
        except:
            img = file_broken.GetImage()
            originalsize = (img.GetWidth(), img.GetHeight())
            alpha = False

        return img, originalsize, alpha


class ResultsGallery(wx.Panel):
    def __init__(self, parent, app=None, **kwargs):
        self.app = app
        self.thumbs = []
        self.needs_update = True

        wx.Panel.__init__(self, parent, id=wx.ID_ANY, **kwargs)

        self.gallery_font = wx.Font(
            16, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False
        )

        self.gallery = ScrolledThumbnail(self, -1)
        self.gallery.SetThumbSize(256, 240)
        self.gallery.SetCaptionFont(font=self.gallery_font)
        self.gallery.EnableToolTips()
        self.gallery._tTextHeight = 32

        self.gallery.Bind(EVT_THUMBNAILS_SEL_CHANGED, self.OnThumbnailSelected)
        wxasync.AsyncBind(
            EVT_THUMBNAILS_DCLICK, self.OnThumbnailActivated, self.gallery
        )
        self.gallery.Bind(EVT_THUMBNAILS_RCLICK, self.OnThumbnailRightClicked)

        self.pub = aiopubsub.Publisher(PUBSUB_HUB, Key("events"))

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.gallery, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        self.SetSizerAndFit(self.sizer)

    def get_selection(self):
        selected = []
        for ii in self.gallery._selectedarray:
            sel = self.gallery.GetItem(ii)
            if sel is not None:
                selected.append(sel.GetData())
        return selected

    def refresh_one_thumbnail(self, item):
        ii = self.gallery._id_to_idx.get(item["id"])
        if ii is not None and ii in self.gallery._cache:
            del self.gallery._cache[ii]
        else:
            self.gallery._cache = {}

    def OnThumbnailSelected(self, evt):
        selected = self.get_selection()
        list = self.app.frame.results_panel.results_panel.list

        list.ClearSelection()
        for item in selected:
            list_index = list.index_from_id(item["id"])
            list.Select(list_index, 1)
        self.pub.publish(Key("item_selected"), list.get_selection())

    async def OnThumbnailActivated(self, evt):
        await self.app.frame.OnGeneratePreviews(None, op="replace")

    def OnThumbnailRightClicked(self, evt):
        selected = self.get_selection()
        if not selected:
            return

        target = selected[0]
        menu = create_popup_menu_for_item(target, evt, self.app)

        pos = evt.GetPoint()
        self.PopupMenu(menu, pos)
        menu.Destroy()

    def SetThumbs(self, filtered):
        if not self.needs_update:
            return

        self.app.frame.statusbar.SetStatusText("Refreshing thumbnails...")
        self.gallery.Clear()
        self.gallery.Refresh()

        self.needs_update = False
        to_show = []

        for item in filtered:
            image_path = find_image_path_for_model(item)

            if image_path is not None:
                thumb = Thumb(
                    os.path.dirname(image_path),
                    os.path.basename(image_path),
                    caption=os.path.splitext(os.path.basename(item["filepath"]))[0],
                    imagehandler=GalleryThumbnailHandler,
                    lastmod=os.path.getmtime(image_path),
                    data=item,
                )
                thumb.SetId(len(to_show))
                to_show.append(thumb)

        self.gallery.ShowThumbs(to_show)

        self.app.frame.statusbar.SetStatusText(f"Done. ({len(to_show)} entries)")


class ResultsPanel(wx.Panel):
    def __init__(self, parent, app=None, **kwargs):
        self.app = app
        self.results = []

        wx.Panel.__init__(self, parent, id=wx.ID_ANY, **kwargs)

        self.list = ResultsListCtrl(self, app)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.list, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        self.SetSizerAndFit(self.sizer)

    def get_selection(self):
        return self.list.get_selection()

class ResultsNotebook(wx.Panel):
    def __init__(self, parent, app=None):
        self.app = app

        wx.Panel.__init__(self, parent, id=wx.ID_ANY)

        self.results = {}
        self.notebook = wx.Notebook(self)
        self.thumbs_need_update = False

        self.results_panel = ResultsPanel(self.notebook, app=self.app)
        self.results_gallery = ResultsGallery(self.notebook, app=self.app)

        self.notebook.AddPage(self.results_panel, "List")
        self.notebook.AddPage(self.results_gallery, "Gallery")

        self.pub = aiopubsub.Publisher(PUBSUB_HUB, Key("events"))
        self.sub = aiopubsub.Subscriber(PUBSUB_HUB, Key("events"))
        self.sub.add_async_listener(
            Key("events", "tree_filter_changed"), self.SubTreeFilterChanged
        )

        # self.searchBox = wx.TextCtrl(self, wx.ID_ANY, style=wx.TE_PROCESS_ENTER)
        self.searchBox = wx.ComboBox(self, wx.ID_ANY, style=wx.TE_PROCESS_ENTER)
        self.searchButton = wx.Button(self, label="Search")
        self.clearButton = wx.Button(self, label="Clear")
        self.saveButton = wx.Button(self, label="Save")
        self.removeButton = wx.Button(self, label="Remove")

        wxasync.AsyncBind(wx.EVT_BUTTON, self.OnSearch, self.searchButton)
        wxasync.AsyncBind(wx.EVT_BUTTON, self.OnClear, self.clearButton)
        wxasync.AsyncBind(wx.EVT_BUTTON, self.OnSaveSearch, self.saveButton)
        wxasync.AsyncBind(wx.EVT_BUTTON, self.OnRemoveSearch, self.removeButton)
        wxasync.AsyncBind(wx.EVT_TEXT_ENTER, self.OnSearch, self.searchBox)
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnPageChanged)

        self.sizerSearchSave = wx.WrapSizer(wx.HORIZONTAL)

        self.sizerSearch = wx.BoxSizer(wx.HORIZONTAL)
        self.sizerSearch.Add(self.searchBox, proportion=5, flag=wx.LEFT | wx.EXPAND | wx.ALL, border=5)
        self.sizerSearch.Add(self.searchButton, proportion=1, flag=wx.ALL, border=5)
        self.sizerSearch.Add(self.clearButton, proportion=1, flag=wx.ALL, border=5)
        self.sizerSearch.Add(self.saveButton, proportion=1, flag=wx.ALL, border=5)
        self.sizerSearch.Add(self.removeButton, proportion=1, flag=wx.ALL, border=5)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.notebook, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.sizer.Add(self.sizerSearchSave, flag=wx.EXPAND | wx.ALL, border=0)
        self.sizer.Add(self.sizerSearch, flag=wx.EXPAND | wx.ALL, border=5)

        self.SetSizerAndFit(self.sizer)
        self.LoadSearchOptions()
        self.UpdateSearchHistory()
        self.UpdateSearchSave()

    def get_selection(self):
        return self.results_panel.get_selection()


    async def refresh_one_item(self, item):
        try_load_image.cache_clear()
        self.results_panel.list.refresh_one_value(item)
        self.results_gallery.refresh_one_thumbnail(item)
        self.Refresh()
        selection = self.get_selection()
        if item in selection:
            await self.app.frame.ForceSelect(selection)

    def OnPageChanged(self, evt):
        sel = evt.GetSelection()
        if sel == 1:  # gallery page
            self.results_gallery.SetThumbs(self.results_panel.list.filtered)

    async def SubTreeFilterChanged(self, key, path):
        list = self.results_panel.list
        list.filter = path
        list.refresh_filtered_data()
        list.refresh_text_view()
        list.select_default()

        self.results_gallery.needs_update = True
        if self.notebook.GetSelection() == 1:
            self.results_gallery.SetThumbs(list.filtered)

    async def re_search(self):
        query = self.searchBox.GetValue()
        await self.search(query, restore_list=True)

    async def search(self, query, restore_list=False):
        query = re.sub(SAVE_NAME_REGEX, "", query)
        # self.pub.publish(Key("item_selected"), [])
        self.results = await self.app.api.get_loras(query)
        try_load_image.cache_clear()

        list = self.results_panel.list
        selection_before = list.get_selection()
        scroll_before = list.GetScrollPos(wx.VERTICAL)
        list.DeleteAllItems()
        list.Arrange(ultimatelistctrl.ULC_ALIGN_DEFAULT)
        list.set_results(self.results)
        if not restore_list:
            list.filter = None
        list.refresh_filtered_data()
        list.refresh_text_view()
        if restore_list and not list.restore_selection(selection_before, scroll_before):
            list.select_default()
        self.results_gallery.needs_update = True
        if self.notebook.GetSelection() == 1:
            self.results_gallery.SetThumbs(list.filtered)

    async def OnSearch(self, evt):
        self.SaveSearchHistory()
        await self.app.frame.search(self.searchBox.GetValue())
        self.UpdateSearchHistory()

    async def OnClear(self, evt):
        self.searchBox.SetValue("")
        await self.app.frame.search("")

    def LoadSearchOptions(self):
        empty_search_options = {
            "search_history": [],
            "search_save": [],
        }
        if not os.path.exists(SEARCH_SAVE_FILE):
            self.search_options = empty_search_options
            return
        with open(SEARCH_SAVE_FILE, "r") as f:
            self.search_options = simplejson.loads(f.read())
        if not isinstance(self.search_options, dict):
            self.search_options = empty_search_options
        if self.search_options.get("search_history") is None:
            self.search_options["search_history"] = []
        if self.search_options.get("search_save") is None:
            self.search_options["search_save"] = []

    def SaveSearchOptions(self):
        if not self.search_options:
            self.search_options = {}
        with open(SEARCH_SAVE_FILE, "w") as f:
            jsonStr = simplejson.dumps(self.search_options)
            f.write(jsonStr)
    
    def UpdateSearchHistory(self):
        query = self.searchBox.GetValue()
        search_history = self.search_options.get("search_history")
        self.searchBox.SetItems(search_history)
        self.searchBox.SetValue(query)

    def UpdateSearchSave(self):
        search_save = self.search_options.get("search_save")
        self.sizerSearchSave.Clear(True)
        for i, value in enumerate(search_save):
            name = SAVE_NAME_DEFAULT
            result = re.search(SAVE_NAME_REGEX, value)
            if result:
                name = result.group(1)
            button = wx.Button(self, label=name)
            self.sizerSearchSave.Add(button, proportion=1, flag=wx.ALL, border=0)
            async def OnClick(evt, value=value):
                self.searchBox.SetValue(value)
                await self.OnSearch(evt)
            wxasync.AsyncBind(wx.EVT_BUTTON, OnClick, button)
        self.Layout()

    def SaveSearchHistory(self):
        value = self.searchBox.GetValue()
        if not value:
            return
        search_history = self.search_options.get("search_history")
        if value in search_history:
            return
        search_history.insert(0, value)
        if len(search_history) > 20:
            self.search_options["search_history"] = search_history[:20]
        self.SaveSearchOptions()

    async def OnSaveSearch(self, evt):
        options = self.search_options.get("search_save")
        value = self.searchBox.GetValue()
        result = re.search(SAVE_NAME_REGEX, value)
        if not result:
            value = f"button name: {SAVE_NAME_DEFAULT} {value}"
        options.append(value)
        self.SaveSearchOptions()
        self.UpdateSearchSave()

    async def OnRemoveSearch(self, evt):
        value = self.searchBox.GetValue()
        options = self.search_options.get("search_save")
        if value not in options:
            return
        options.remove(value)
        self.SaveSearchOptions()        
        self.UpdateSearchSave()
