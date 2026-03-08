"""
OpenAI 兼容路由：/{type}/v1/chat/completions、/{type}/v1/models。
依赖注入 ChatHandler（由 app 组装）。
支持 ReAct：解析 Thought/Action/Action Input 格式并转为 OpenAI tool_calls，由 Cursor 执行。
"""

import json
import logging
import re
import time
import uuid as uuid_mod
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.api.auth import require_api_key
from core.api.chat_handler import ChatHandler
from core.plugin.base import PluginRegistry
from core.api.function_call import build_tool_calls_response
from core.api.conv_parser import extract_session_id_marker, strip_session_id_suffix
from core.api.react import (
    format_react_final_answer_content,
    format_react_prompt,
    parse_react_output,
    react_output_to_tool_calls,
)
from core.api.react_stream_parser import ReactStreamParser
from core.api.schemas import OpenAIChatRequest, extract_user_content

logger = logging.getLogger(__name__)


def get_chat_handler(request: Request) -> ChatHandler:
    """从 app state 取出 ChatHandler。"""
    handler = getattr(request.app.state, "chat_handler", None)
    if handler is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return handler


def create_router() -> APIRouter:
    """创建 v1 兼容路由，路径为 /{type}/v1/..."""
    router = APIRouter(dependencies=[Depends(require_api_key)])

    @router.get("/{type}/v1/models")
    def list_models(type: str) -> dict[str, Any]:
        plugin = PluginRegistry.get(type)
        try:
            mapping = plugin.model_mapping() if plugin is not None else None
        except Exception:
            mapping = None

        # 优先使用插件暴露的 OpenAI 兼容模型名（mapping 的 key）
        model_ids: list[str]
        if isinstance(mapping, dict) and mapping:
            model_ids = list(mapping.keys())
        else:
            raise HTTPException(
                status_code=500, detail="model_mapping is not implemented"
            )

        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "created": now,
                    "owned_by": type,
                }
                for mid in model_ids
            ],
        }

    @router.post("/{type}/v1/chat/completions")
    async def chat_completions(
        type: str,
        req: OpenAIChatRequest,
        handler: ChatHandler = Depends(get_chat_handler),
    ) -> Any:
        if not req.messages:
            raise HTTPException(
                status_code=400,
                detail="messages 不能为空",
            )

        has_tools = bool(req.tools)
        react_prompt_prefix = format_react_prompt(req.tools or []) if has_tools else ""
        content = extract_user_content(
            req.messages,
            has_tools=has_tools,
            react_prompt_prefix=react_prompt_prefix,
        )
        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="messages 中需至少有一条带 content 的 user 消息",
            )

        chat_id = f"chatcmpl-{uuid_mod.uuid4().hex[:24]}"
        created = int(time.time())
        model = req.model

        if req.stream:

            async def sse_stream() -> AsyncIterator[str]:
                try:
                    parser = ReactStreamParser(
                        chat_id=chat_id,
                        model=model,
                        created=created,
                        has_tools=has_tools,
                    )
                    async for chunk in handler.stream_completion(type, req):
                        # 不能 strip_session_id_suffix：会话 ID 用零宽字符附在末尾，必须原样发给客户端，
                        # 客户端把 assistant 消息存进历史后，下一轮请求会带回来，服务端才能解析出 conv_uuid 复用会话
                        for sse in parser.feed(chunk):
                            yield sse
                    for sse in parser.finish():
                        yield sse
                except ValueError as e:
                    logger.warning("chat 请求参数错误: %s", e)
                    err_sse = f"data: {json.dumps({'error': {'message': str(e), 'type': 'invalid_request_error'}}, ensure_ascii=False)}\n\n"
                    yield err_sse
                except Exception as e:
                    logger.exception("流式 chat 失败")
                    err_sse = f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}}, ensure_ascii=False)}\n\n"
                    yield err_sse

            return StreamingResponse(
                sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # 非流式：收集完整内容后解析，若有 tool_call 则返回 tool_calls 格式
        full: list[str] = []
        try:
            async for chunk in handler.stream_completion(type, req):
                full.append(chunk)
        except Exception as e:
            logger.exception("chat/completions 失败")
            raise HTTPException(status_code=500, detail=str(e)) from e

        reply = "".join(full)
        tool_calls_list: list[dict[str, Any]] = []
        if has_tools:
            content_for_parse = strip_session_id_suffix(reply)
            parsed = parse_react_output(content_for_parse)
            tool_calls_list = react_output_to_tool_calls(parsed) if parsed else []
        if tool_calls_list:
            session_id_content = extract_session_id_marker(reply)
            thought_ns = ""
            if "Thought" in content_for_parse:
                m = re.search(
                    r"Thought[:：]\s*(.+?)(?=\s*Action[:：]|$)",
                    content_for_parse,
                    re.DOTALL | re.I,
                )
                thought_ns = (m.group(1) or "").strip() if m else ""
            # Thought 解析为 thinking：用 <think> 包裹，单换行避免与 tool_calls 间距过大
            text_content = (
                f"<think>{thought_ns}</think>\n{session_id_content}".strip()
                if thought_ns
                else session_id_content
            )
            resp = build_tool_calls_response(
                tool_calls_list,
                chat_id,
                model,
                created,
                text_content=text_content,
            )
            return resp

        # 无 tool_calls 时为纯文本或 ReAct Final Answer，Thought 用 <think> 包裹便于客户端展示思考
        content_reply = format_react_final_answer_content(reply) if has_tools else reply
        resp = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_reply},
                    "finish_reason": "stop",
                }
            ],
        }
        return resp

    return router
