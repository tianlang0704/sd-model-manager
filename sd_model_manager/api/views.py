from aiohttp import web
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectin_polymorphic
from sqlakeyset.asyncio import select_page
import simplejson

from sd_model_manager.models.sd_models import SDModel, LoRAModel, LoRAModelSchema

def paging_to_json(paging, limit):
    return {
        "next": paging.bookmark_next,
        "current": paging.bookmark_current,
        "previous": paging.bookmark_previous,
        "limit": limit
    }

routes = web.RouteTableDef()

@routes.get("/api/v1/loras")
async def index(request):
    page_marker = request.rel_url.query.get("page", None)
    limit = int(request.rel_url.query.get("limit", 20))

    async with request.app["db"].AsyncSession() as s:
        query = select(LoRAModel).order_by(SDModel.id).options(selectin_polymorphic(SDModel, [LoRAModel]))
        page = await select_page(s, query, per_page=limit, page=page_marker)

        schema = LoRAModelSchema()

        resp = {
            "paging": paging_to_json(page.paging, limit),
            "data": [schema.dump(m[0]) for m in page]
        }

        return web.json_response(resp, dumps=simplejson.dumps)

@routes.get("/api/v1/lora/{id}")
async def show(request):
    model_id = request.match_info.get("id", None)
    if model_id is None:
        return web.Response(status=404)

    async with request.app["db"].AsyncSession() as s:
        row = await s.get(LoRAModel, model_id)

        schema = LoRAModelSchema()

        resp = {
            "data": schema.dump(row)
        }

        return web.json_response(resp, dumps=simplejson.dumps)
