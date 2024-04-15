import os
import pathlib
import functools
import sys
from PIL import Image
from typing import Any, Optional, List

from aiohttp import web
import configargparse
import argparse
import inspect

PATH = pathlib.Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = PATH / "config.yml"

p = configargparse.ArgParser(
    default_config_files=[DEFAULT_CONFIG_PATH],
    config_file_parser_class=configargparse.YAMLConfigFileParser,
)
p.add_argument("-c", "--config-file", is_config_file=True, help="Config file path")
p.add_argument("-l", "--listen", type=str, default="127.0.0.1", help="Address for model manager in standalone mode")
p.add_argument("-p", "--port", type=int, default=7779, help="Port for model manager in standalone mode")
p.add_argument("-cl", "--comfy-listen", type=str, default="127.0.0.1", help="Address for comfyui server")
p.add_argument("-cp", "--comfy-port", type=int, default=8188, help="Port for comfyui server")
p.add_argument("-m", "--mode", type=str, default="standalone", help="Runtime mode ('standalone', 'noserver', 'comfyui')")
p.add_argument("--model-paths", type=str, nargs="+")

def get_config(argv):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) > 0 and argv[0].endswith("adev"):
        argv = []
    config = p.parse_args(argv)
    config.listen = config.listen.strip()
    config.comfy_listen = config.comfy_listen.strip()
    config.mode = config.mode.strip()
    return config

IMAGE_EXTS = set([".png", ".jpg", ".jpeg", ".gif", ".webp"])

def is_image_path(path):
    ext = os.path.splitext(path)
    return ext in IMAGE_EXTS and os.path.isfile(path)


@functools.lru_cache
def try_load_image(file):
    if not os.path.isfile(file):
        return None
    try:
        image = Image.open(file)
        image.load()
        return image.convert("RGB")
    except Exception as ex:
        return None


def find_image(filepath, load=False, fuzzy=True):
    path = os.path.dirname(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]

    for s in [".preview.png", ".png"]:
        file = os.path.join(path, basename + s)
        if load:
            image = try_load_image(file)
            if image:
                return image, file
        elif os.path.isfile(file):
            return None, file

    if fuzzy:
        for fname in os.listdir(path):
            file = os.path.realpath(os.path.join(path, fname))
            if basename in file:
                if load:
                    image = try_load_image(file)
                    if image:
                        return image, file
                elif is_image_path(file):
                    return None, file
    return None, None

def is_comfyui():
    for i in inspect.stack(0):
        filename = os.path.basename(i[1])
        function = i.function
        if filename == "nodes.py" and function == "load_custom_node":
            return True
    return False