from dataclasses import dataclass
import urllib
import aiohttp
import simplejson

import wx
import wx.aui


class ModelManagerAPI:
    def __init__(self, config):
        self.config = config
        self.client = aiohttp.ClientSession()

    def base_url(self):
        host = self.config.listen
        if host == "0.0.0.0":
            host = "localhost"
        url = f"http://{host}:{self.config.port}"
        if self.config.mode == "comfyui":
            url += "/models"
        return url

    async def get_loras(self, query):
        params = {"limit": 1000}
        if query:
            params["query"] = query

        async with self.client.get(
            self.base_url() + "/api/v1/loras", params=params
        ) as response:
            if response.status != 200:
                print(await response.text())
            return await response.json()

    async def update_lora(self, id, changes):
        async with self.client.patch(
            self.base_url() + f"/api/v1/lora/{id}",
            data=simplejson.dumps({"changes": changes}),
        ) as response:
            if response.status != 200:
                print(await response.text())
            return await response.json()

    def update_lora_sync(self, id, changes):
        """
        not putting up with the tire fire that is asyncio any longer
        """
        req = urllib.request.Request(
            self.base_url() + f"/api/v1/lora/{id}", method="PATCH"
        )
        req.add_header("Content-Type", "application/json")
        data = simplejson.dumps({"changes": changes})
        data = data.encode("utf-8")
        with urllib.request.urlopen(req, data=data) as response:
            return response.read()
    
    async def remove_lora(self, id_or_list, is_remove_model=False):
        if isinstance(id_or_list, list):
            id_or_list = ",".join(map(str, id_or_list))
        async with self.client.delete(self.base_url() + f"/api/v1/lora/{id_or_list}/{is_remove_model}") as response:
            if response.status != 200:
                print(await response.text())
            return await response.json()

# TODO make async
class ComfyAPI:
    def __init__(self, app):
        listen = app.config.comfy_listen
        if listen == "0.0.0.0":
            listen = "localhost"
        self.server_address = f"{listen}:{app.config.comfy_port}"

    def get_history(self, prompt_id):
        with urllib.request.urlopen(
            f"http://{self.server_address}/history/{prompt_id}"
        ) as response:
            return simplejson.loads(response.read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(
            f"http://{self.server_address}/view?{url_values}"
        ) as response:
            return response.read()

    def get_images(self, prompt_id):
        output_images = {}
        output_files = {}

        history = self.get_history(prompt_id)[prompt_id]
        for o in history["outputs"]:
            for node_id in history["outputs"]:
                node_output = history["outputs"][node_id]
                if "images" in node_output:
                    images_output = []
                    for image in node_output["images"]:
                        image_data = self.get_image(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        images_output.append(image_data)
                output_images[node_id] = images_output
                output_files[node_id] = node_output["images"]

        return output_images, output_files
