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
from main import create_app
from sd_model_manager.utils.common import get_config

import gui.patch

from gui.app import App

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except:
    pass

async def init_server():
    server = await create_app([])
    host = server["sdmm_config"].listen
    port = server["sdmm_config"].port

    runner = web.AppRunner(server)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    return server


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
    config.listen = config.listen.strip()
    config.mode = config.mode.strip()
    use_internal_server = config.mode != "noserver" and config.mode != "comfyui"
    is_comfyui = config.mode == "comfyui"

    if config.port is None:
        config.port = 8188 if is_comfyui else 7779

    server = None
    if use_internal_server:
        server = await init_server()

    app = App(server, config, redirect=False)
    sys.excepthook = exception_handler

    try:
        await app.MainLoop()
    except asyncio.exceptions.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
