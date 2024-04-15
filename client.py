#!/usr/bin/env python

import os
cwd = os.path.dirname(os.path.realpath(__file__))
if os.getcwd() != cwd:
    os.chdir(cwd)
import sys
sys.path.insert(0, cwd)

import sys
import ctypes
import asyncio
import argparse
import traceback
from aiohttp import web
import wx
from sd_model_manager.server_app import create_server
from sd_model_manager.utils.common import get_config

import gui.patch

from gui.client_app import ClientApp

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except:
    pass


app = None
def exception_handler(exception_type, exception_value, exception_traceback):
    global app
    msg = "An error has occurred!\n\n"
    tb = traceback.format_exception(
        exception_type, exception_value, exception_traceback
    )
    for i in tb:
        msg += i

    parent = None
    if app is not None:
        parent = app.frame
    dlg = wx.MessageDialog(parent, msg, str(exception_type), wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()


async def main():
    global app
    config = get_config(sys.argv[1:])
    is_comfyui = config.mode == "comfyui"
    is_noserver = config.mode == "noserver"

    if is_comfyui:
        config.listen = config.comfy_listen
        config.port = config.comfy_port
    if not config.listen:
        config.listen = "127.0.0.1"
    if config.port is None:
        config.port = 8188 if is_comfyui else 7779

    server = None
    use_internal_server = (not is_noserver) and (not is_comfyui)
    if use_internal_server:
        server = await create_server(start_type="runner")

    app = ClientApp(server, config, redirect=False)
    sys.excepthook = exception_handler

    try:
        await app.MainLoop()
    except asyncio.exceptions.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
