#!/usr/bin/env python

"""
    Name:

        ImagePanel.py

    Description:

        A panel containing a wx.StaticBitmap control that can be used to display
        an image. The image is scale to fit inside the panel while maintaining the
        image's original aspect ratio. The image size is recaulated whenever the
        panel is resized.

        You can zoom in/out using CTRL+Scroll Wheel. The image is displayed in a
        panel with scroll bars. If zoomed in you can scroll to see portions of the
        image that are off the display.

    Methods:

        Load(file)  - load and display the image from the given file
        Clear()     - clear the display

        All common image formats are supported.

    Audit:

        2021-07-20  rj  original code

"""

import io
import os
import wx
# import wx.lib.mixins.inspection
import wx.lib.scrolledpanel as scrolled
from PIL import Image
import gui.utils

class ImagePanel(wx.Panel):
    """
    This control implements a basic image viewer. As the control is
    resized the image is resized (aspect preserved) to fill the panel.

    Methods:

        Load(filename)   display the image from the given file
        Clear()          clear the displayed image
    """

    def __init__(
        self,
        parent,
        id=wx.ID_ANY,
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.SUNKEN_BORDER,
    ):

        super().__init__(parent, id, pos, size, style=style)

        self.bmpImage = wx.StaticBitmap(self, wx.ID_ANY)
        self.bmpImage.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.bmpImage, 1, wx.EXPAND, 0)
        self.SetSizer(self.sizer)

        self.image = None  # loaded image in image format
        self.aspect = None  # loaded image aspect ratio
        self.zoom = 1.0  # zoom factor
        self.blank = wx.Bitmap(1, 1)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)

    def OnLeftDown(self, event):
        """Start dragging"""
        if not self.image:
            return
        data = ""
        if "parameters" in self.image.info:
            data = self.image.info["parameters"]
        elif "prompt" in self.image.info:
            data = self.image.info["prompt"]
        write_path = os.path.join(gui.utils.PROGRAM_ROOT, "drag_drop_temp.json")
        try:
            if os.path.exists(write_path):
                os.remove(write_path)
            with open(write_path, "w") as f:
                f.write(data)
        except:
            return
        dataObj = wx.FileDataObject()
        dataObj.AddFile(write_path)
        dropSource = wx.DropSource(self)
        dropSource.SetData(dataObj)
        dropSource.DoDragDrop(True)

    def OnSize(self, event):
        """When panel is resized, scale the image to fit"""
        self.ScaleToFit()
        event.Skip()

    def OnMouseWheel(self, event):
        """zoom in/out on CTRL+scroll"""
        m = wx.GetMouseState()

        if m.ControlDown():
            delta = 0.1 * event.GetWheelRotation() / event.GetWheelDelta()
            self.zoom = max(1, self.zoom + delta)
            self.ScaleToFit()

        event.Skip()

    def SetImage(self, image):
        """Set the image to be displayed"""
        self.image = image
        self.aspect = image.height / image.width
        self.zoom = 1.0
        self.ScaleToFit()

    def LoadImageFromBytes(self, data) -> None:
        image = Image.open(io.BytesIO(data))
        self.SetImage(image)

    def LoadImageFrompath(self, path) -> None:
        image = Image.open(path)
        self.SetImage(image)

    def Clear(self):
        """Set the displayed image to blank"""
        self.bmpImage.SetBitmap(self.blank)
        self.image = None
        self.aspect = None
        self.zoom = 1.0

    def ScaleToFit(self) -> None:
        """
        Scale the image to fit in the container while maintaining
        the original aspect ratio.
        """
        if self.image:

            # get container (c) dimensions
            cw, ch = self.GetSize()

            # calculate new (n) dimensions with same aspect ratio
            nw = cw
            nh = int(nw * self.aspect)

            # if new height is too large then recalculate sizes to fit
            if nh > ch:
                nh = ch
                nw = int(nh / self.aspect)

            # Apply zoom
            nh = int(nh * self.zoom)
            nw = int(nw * self.zoom)

            # scale the image to new dimensions and display
            image = self.image.resize((nw, nh), resample=Image.BICUBIC)
            self.bmpImage.SetBitmap(wx.BitmapFromBuffer(nw, nh, image.tobytes()))
            self.Layout()
