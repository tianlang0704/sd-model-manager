import os
import subprocess
import sys
from aiohttp import web
from sqlalchemy import create_engine, select, or_
from sqlalchemy.orm import Session, selectinload, selectin_polymorphic
from sqlakeyset.asyncio import select_page
import simplejson

from sd_model_manager.models.sd_models import (
    PreviewImage,
    PreviewImageSchema,
    SDModel,
    LoRAModel,
    LoRAModelSchema,
)
from sd_model_manager.query import build_search_query


def paging_to_json(paging, limit):
    return {
        "next": paging.bookmark_next,
        "current": paging.bookmark_current,
        "previous": paging.bookmark_previous,
        "limit": limit,
    }


routes = web.RouteTableDef()

@routes.get("/api/v1/open_manager")
async def open_manager(request):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    client_path = os.path.abspath(os.path.join(current_dir, "..", "..", "client.py"))
    msg = "ok"
    cmd = f"{sys.executable} {client_path} -m comfyui"
    try:
        p = subprocess.Popen(cmd)
    except Exception as e:
        msg = str(e)
    return web.json_response({"msg": msg})

@routes.get("/api/v1/preview_image/{id}")
async def show_preview_image(request):
    image_id = request.match_info.get("id", None)
    if image_id is None:
        return web.Response(status=404)

    async with request.app["sdmm_db"].AsyncSession() as s:
        query = select(PreviewImage).filter(PreviewImage.id == image_id)

        row = (await s.execute(query)).one()
        if row is None:
            return web.json_response(
                {"message": f"Preview image not found: {image_id}"}, status=404
            )
        row = row[0]

        schema = PreviewImageSchema()

        resp = {"data": schema.dump(row)}

        return web.json_response(resp, dumps=simplejson.dumps)


@routes.get("/api/v1/preview_image/{id}/view")
async def view_preview_image_file(request):
    image_id = request.match_info.get("id", None)
    if image_id is None:
        return web.Response(status=404)

    async with request.app["sdmm_db"].AsyncSession() as s:
        query = select(PreviewImage).filter(PreviewImage.id == image_id)

        row = (await s.execute(query)).one()
        if row is None:
            return web.Response(status=404)
        row = row[0]

        if not os.path.isfile(row.filepath):
            return web.Response(status=404)

        with open(row.filepath, "rb") as b:
            return web.Response(body=b.read(), content_type="image/jpeg")


@routes.get("/api/v1/loras")
async def index_loras(request):
    page_marker = request.rel_url.query.get("page", None)
    limit = int(request.rel_url.query.get("limit", 100))
    search_query = request.rel_url.query.get("query", None)

    async with request.app["sdmm_db"].AsyncSession() as s:
        query = select(LoRAModel)
        if search_query:
            query = build_search_query(query, search_query)
        query = query.options(selectin_polymorphic(SDModel, [LoRAModel])).options(
            selectinload(SDModel.preview_images)
        )

        page = await select_page(s, query, per_page=limit, page=page_marker)

        schema = LoRAModelSchema()

        resp = {
            "paging": paging_to_json(page.paging, limit),
            "data": [schema.dump(m[0]) for m in page],
        }

        return web.json_response(resp, dumps=simplejson.dumps)


@routes.get("/api/v1/lora/{id}")
async def show_loras(request):
    model_id = request.match_info.get("id", None)
    if model_id is None:
        return web.Response(status=404)

    async with request.app["sdmm_db"].AsyncSession() as s:
        query = select(LoRAModel).filter(LoRAModel.id == model_id)
        query = query.options(selectin_polymorphic(SDModel, [LoRAModel])).options(
            selectinload(SDModel.preview_images)
        )

        row = (await s.execute(query)).one()
        if row is None:
            return web.json_response(
                {"message": f"LoRA not found: {model_id}"}, status=404
            )
        row = row[0]

        schema = LoRAModelSchema()

        resp = {"data": schema.dump(row)}

        return web.json_response(resp, dumps=simplejson.dumps)

@routes.delete("/api/v1/lora/{id_or_list}")
async def delete_lora(request):
    id_or_list = request.match_info.get("id_or_list", None)
    if id_or_list is None:
        return web.json_response({"message": "No LoRA ID provided"}, status=404)

    async with request.app["sdmm_db"].AsyncSession() as s:
        query = select(LoRAModel).filter(LoRAModel.id.in_(id_or_list.split(",")))
        query = query.options(selectin_polymorphic(SDModel, [LoRAModel]))
        rows = (await s.execute(query)).all()
        if rows is None:
            return web.json_response(
                {"message": f"LoRA not found: {id_or_list}"}, status=404
            )
        for row in rows:
            await s.delete(row[0])
        await s.commit()
        return web.json_response({"status": "ok"})
    
@routes.patch("/api/v1/lora/{id}")
async def update_lora(request):
    model_id = request.match_info.get("id", None)
    if model_id is None:
        return web.json_response({"message": "No LoRA ID provided"}, status=404)

    data = await request.json()
    changes = data.get("changes", None)

    if changes is None:
        return web.Response(status=400)

    async with request.app["sdmm_db"].AsyncSession() as s:
        query = select(LoRAModel).filter(LoRAModel.id == model_id)
        query = query.options(selectin_polymorphic(SDModel, [LoRAModel])).options(
            selectinload(SDModel.preview_images)
        )

        row = (await s.execute(query)).one()
        if row is None:
            return web.json_response(
                {"message": f"LoRA not found: {model_id}"}, status=404
            )
        row = row[0]

        updated = 0

        fields = [
            "display_name",
            "version",
            "author",
            "source",
            "tags",
            "keywords",
            "negative_keywords",
            "description",
            "notes",
            "rating",
            "root_path",
            "filepath",
        ]

        for field in fields:
            if field in changes:
                setattr(row, field, changes[field])
                updated += 1

        if "preview_images" in changes:
            row.preview_images = []
            await s.flush()

            new_images = []
            for image in changes["preview_images"]:
                if "id" in image:
                    existing = await s.get(PreviewImage, image["id"])
                    if existing is None:
                        return web.json_response(
                            {"message": f"Preview image not found: {image['id']}"},
                            status=404,
                        )
                    for k, v in image.items():
                        if k == "filepath":
                            v = os.path.normpath(v)
                        setattr(existing, k, v)
                    new_images.append(existing)
                else:
                    new_image = PreviewImage(
                        filepath=os.path.normpath(image["filepath"]),
                        is_autogenerated=image.get("is_autogenerated", False),
                        model_id=row.id,
                    )
                    new_images.append(new_image)
                    s.add(new_image)
            row.preview_images = new_images
            updated += 1

        error = None
        try:
            await s.commit()
        except Exception as e:
            error = str(e)

        if error:
            resp = {"status": "error", "message": error}
        else:
            resp = {"status": "ok", "fields_updated": updated}
        return web.json_response(resp, dumps=simplejson.dumps)
