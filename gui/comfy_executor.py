import uuid
import json
import urllib.request
import urllib.parse
import os
import websocket
import simplejson

from gui.utils import PROGRAM_ROOT


class ComfyExecutor:
    def __init__(self, app):
        listen = app.config.comfy_listen
        if listen == "0.0.0.0":
            listen = "localhost"
        self.server_address = f"{listen}:{app.config.comfy_port}"
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def __enter__(self):
        self.ws = websocket.WebSocket()
        self.ws.connect(
            "ws://{}/ws?clientId={}".format(self.server_address, self.client_id)
        )
        return self

    def __exit__(self, type, value, traceback):
        self.ws.close()

    def enqueue(self, prompt_json):
        p = {"prompt": prompt_json, "client_id": self.client_id, "number": 10000}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        res = None
        try:
            res = json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            error = str(e)
            try:
                error_content = json.loads(e.read())
                error = error_content["error"]["message"]
            except:
                pass
            details = None
            try:
                node_errors = error_content["node_errors"]
                one_error = next(iter(node_errors.values()))["errors"][0]
                details = one_error["message"] + "\n" + one_error["details"]
            except:
                pass
            res = {"error": error, "details": details}
        except Exception as e:
            res = {"error": str(e)}
        return res

    def get_status(self):
        out = self.ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            return {"type": "json", "data": message}
        else:
            return {"type": "binary", "data": out}
        
    def interrupt(self):
        req = urllib.request.Request(f"http://{self.server_address}/interrupt")
        req.method = "POST"
        urllib.request.urlopen(req)
