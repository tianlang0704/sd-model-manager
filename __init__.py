# Entrypoint for use as a ComfyUI extension

import os
import sys
import asyncio
import shutil
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, 'custom_nodes')
js_path = os.path.join(comfy_path, "web", "extensions")
sd_model_manager_path = os.path.dirname(__file__)

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

from sd_model_manager.db import DB
from sd_model_manager.api.views import routes as api_routes
from sd_model_manager.utils.common import get_config, is_comfyui

def setup_js():
    import nodes
    js_dest_path = os.path.join(js_path, "sd-model-manager")

    if hasattr(nodes, "EXTENSION_WEB_DIRS"):
        if os.path.exists(js_dest_path):
            shutil.rmtree(js_dest_path)
    else:
        print(f"[WARN] SD-MODEL-Manager: Your ComfyUI version is outdated. Please update to the latest version.")
        # setup js
        if not os.path.exists(js_dest_path):
            os.makedirs(js_dest_path)
        js_src_path = os.path.join(sd_model_manager_path, "js", "sd-model-manager.js")

        print(f"### SD-MODEL-Manager: Copy .js from '{js_src_path}' to '{js_dest_path}'")
        shutil.copy(js_src_path, js_dest_path)


setup_js()

async def initialize_comfyui():
    print("[SD-Model-Manager] Initializing...")

    import server
    from aiohttp import web
    import aiohttp

    prompt_server = server.PromptServer.instance

    for route in api_routes:
        prompt_server.routes._items.append(
            web.RouteDef(
                route.method, "/models" + route.path, route.handler, route.kwargs
            )
        )

    app = prompt_server.app
    app["sdmm_config"] = get_config([])

    db = DB()
    await db.init(app["sdmm_config"].model_paths)
    # await db.scan(app["sdmm_config"].model_paths)
    app["sdmm_db"] = db

    print("[SD-Model-Manager] Initialized via ComfyUI server.")


if not is_comfyui():
    raise RuntimeError(
        "This script was not run from ComfyUI, use client.py for a standalone GUI instead"
    )


asyncio.run(initialize_comfyui())

WEB_DIRECTORY = "js"
NODE_CLASS_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS"]