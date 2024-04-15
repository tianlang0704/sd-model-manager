from pathlib import Path
from typing import (
    Literal,
    Optional,
    List,
    AsyncGenerator,
)

import aiohttp_jinja2
from aiohttp import web
import jinja2

from sd_model_manager.db import DB
from sd_model_manager.routes import init_routes
from sd_model_manager.utils.common import get_config


path = Path(__file__).parent


def init_jinja2(app: web.Application) -> None:
    """
    Initialize jinja2 template for application.
    """
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(path / "templates")))


def init_server(argv: Optional[List[str]] = None) -> web.Application:
    app = web.Application()

    init_jinja2(app)
    init_routes(app)

    return app

async def create_server(
        argv=None, 
        existing_server=None, 
        need_db = True, 
        try_debug = True,
        start_type:Literal["no_start", "loop", "runner"] = "no_start"
) -> web.Application:
    if existing_server is None:
        server_app = init_server()
    else:
        server_app = existing_server
        start_type = "no_start"

    if need_db:
        server_app["sdmm_config"] = get_config(argv)
        db = DB()
        await db.init(server_app["sdmm_config"].model_paths)
        server_app["sdmm_db"] = db

    if try_debug:
        try:
            import aiohttp_debugtoolbar
            aiohttp_debugtoolbar.setup(server_app, check_host=False)
        except ModuleNotFoundError:
            pass

    host = server_app["sdmm_config"].listen
    port = server_app["sdmm_config"].port
    if start_type == "runner":
        runner = web.AppRunner(server_app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
    elif start_type == "loop":
        web.run_app(server_app, host=host, port=port)
    return server_app
