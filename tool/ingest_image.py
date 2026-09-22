"""图片入库工具：把一张图片写入感知记忆并做视觉抽取（写工具）。

为什么这是一个独立能力
======================

图片入库是"多模态入口"：图片字节是**感知记忆的原始载荷**（由 VL 嵌入后端向量化），
而图上的边只能来自结构化视觉抽取器，绝不来自向量相似度。这条规则与纯文本入库
（``memory.rag``）完全不同，所以它是一个独立动作，而不是 ``ingest`` 的一个参数。

它的三步是固定的：过嵌入锁闸门 → 写一条 perceptual 记忆（含图片字节）→
视觉抽取并物化成 n-ary 观察（实体/关系/时序）。
抽取失败**不**回滚已入库的图片：源句必须留下，错误写进报告，这是原来的语义。

唯一实现
========

图片入库实现整体搬到这里；曾经保留的 ``RAGPipeline.ingest_media`` 薄委托没有生产调用，
还会造成 ``memory → tool → memory`` 的反向依赖，因此已删除。Web 与工具协议现在都直接
调用本模块的 ``ingest_image``；``accepts_parameter``（抽取器协议自省）仍由
``memory.rag.pipeline`` 提供，不写第二份。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from constants import MAX_UPLOAD_BYTES
from core import BaseTool, ToolSpec
from memory.base import MemoryType
from memory.embedding_lock import apply_embedding_lock
from memory.rag.knowledge import EntityResolver, build_graph_context, materialize_extraction
from memory.rag.pipeline import RAGPipeline, accepts_parameter

from ._memory import build_default_pipeline
from ._shared import resolve_path, workspace_root

TOOL_ENABLED = True

#: 扩展名 → MIME（只用于把文件读成正确媒体类型；内容仍由抽取器解释）。
MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

#: 未配置 VL 嵌入时的诚实提示（前端与工具共用同一句话）。
TEXT_ONLY_WARNING = (
    "当前 embedding 不是 VL 模型，图片已留存且已识图，但向量只使用文字说明。"
)


def guess_mime_type(path: Path) -> str:
    return MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")


def ingest_image(
    pipeline: RAGPipeline,
    *,
    image: bytes,
    text: str = "",
    mime_type: str = "image/jpeg",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index one image and materialize vision-extracted n-ary observations.

    The image bytes are the canonical perceptual payload and are embedded by
    a VL-capable embedding backend. Knowledge graph edges come only from the
    structured vision extractor, never from vector similarity.
    """

    if not isinstance(image, bytes) or not image:
        raise ValueError("image must be non-empty bytes")
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        raise ValueError("mime_type must be an image media type")
    details = dict(metadata or {})
    details.setdefault("source", details.get("filename") or "图片入库")
    details["modality"] = "image"
    content = text.strip() if isinstance(text, str) else ""
    if not content:
        content = str(details.get("filename") or "图片观察")
    apply_embedding_lock(pipeline.manager, pipeline.document_repo())
    item = pipeline.manager.add(
        content,
        memory_type=MemoryType.PERCEPTUAL,
        metadata=details,
        payload=image,
        modality="image",
        timestamp=details.get("captured_at") or None,
    )
    report: dict[str, Any] = {
        "chunks": 1,
        "domains": [],
        "entities": 0,
        "relations": 0,
        "superseded": 0,
        "retracted": 0,
        "skipped_relations": 0,
        "errors": [],
        "modality": "image",
        "multimodal_embedding": bool(
            getattr(pipeline.manager.embedding, "multimodal", False)
        ),
    }
    if pipeline.auto_extract:
        try:
            resolver = EntityResolver(pipeline.manager)
            graph_context = build_graph_context(
                pipeline.manager, content, resolver=resolver
            )
            kwargs: dict[str, Any] = {"metadata": details}
            if accepts_parameter(pipeline.extractor, "graph_context"):
                kwargs["graph_context"] = graph_context
            if accepts_parameter(pipeline.extractor, "image"):
                kwargs.update({"image": image, "mime_type": mime_type})
            extraction = pipeline.extractor.extract(content, **kwargs)
            materialized = materialize_extraction(
                pipeline.manager,
                extraction,
                source_item=item,
                source_metadata=details,
                resolver=resolver,
            )
            for key in (
                "entities",
                "relations",
                "superseded",
                "retracted",
                "skipped_relations",
            ):
                report[key] = materialized[key]
            report["domains"] = [materialized["domain"]]
        except Exception as exc:  # noqa: BLE001 - keep indexed source on extraction failure
            report["errors"].append(f"{type(exc).__name__}: {exc}")
    pipeline.last_ingest_report = report
    return {
        "item_id": item.id,
        "modality": "image",
        "extraction": report,
        "warning": "" if report.get("multimodal_embedding") else TEXT_ONLY_WARNING,
    }


class IngestImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(min_length=1, max_length=1000, description="图片文件路径（相对工作区根，或工作区内绝对路径）。")
    text: str = Field(default="", max_length=20_000, description="图片说明/上下文文字；留空则用文件名当观察内容。")
    mime_type: str = Field(
        default="",
        max_length=80,
        description="图片媒体类型；留空按扩展名推断（jpg/png/webp/gif/bmp）。",
    )
    captured_at: str = Field(default="", max_length=80, description="拍摄时间（ISO 8601）；留空表示未知。")


class IngestExtractionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunks: int = 0
    domains: list[str] = Field(default_factory=list)
    entities: int = 0
    relations: int = 0
    superseded: int = 0
    retracted: int = 0
    skipped_relations: int = 0
    errors: list[str] = Field(default_factory=list, description="抽取失败原因；图片本身仍然已入库。")
    modality: str = "image"
    multimodal_embedding: bool = Field(description="false 表示当前 embedding 不是 VL 模型，向量只用了文字。")


class IngestImageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item_id: str = Field(description="写入的感知记忆项 id。")
    modality: str = "image"
    extraction: IngestExtractionReport
    warning: str = Field(default="", description="非空时表示向量只用了文字说明（诚实降级提示）。")


class IngestImageTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.ingest_image",
        description=(
            "Index one image: store it as a perceptual memory (VL-embedded when "
            "available) and materialize the vision extractor's n-ary observations "
            "into the knowledge graph. Extraction failure keeps the stored image "
            "and reports the error."
        ),
        version="1.0.0",
        input_model=IngestImageInput,
        output_model=IngestImageOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=300.0,
        idempotent=False,
        parallel_safe=False,
        tags=("knowledge", "image", "multimodal", "write"),
        guidance=(
            "用户给出本地图片路径并希望它进入知识库时使用。图片会作为感知记忆留存，图上的边只能来自视觉抽取器。path 必须位于工作区内，越界会被拒绝。"
            "非 VL 嵌入时返回的 warning 必须如实告知用户（向量只用了文字说明）；抽取失败不会回滚已入库的图片，此时要在回答里说明这一点。"
        ),
    )

    def __init__(self, pipeline: RAGPipeline | None = None) -> None:
        self._pipeline = pipeline

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = build_default_pipeline()
        return self._pipeline

    def execute(self, arguments: IngestImageInput) -> IngestImageOutput:
        path = resolve_path(workspace_root(), arguments.path)
        if not path.is_file():
            raise LookupError(f"图片文件不存在：{path}")
        size = path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"图片超过上限：{size} 字节 > {MAX_UPLOAD_BYTES} 字节"
            )
        mime_type = arguments.mime_type.strip() or guess_mime_type(path)
        if not mime_type.startswith("image/"):
            raise ValueError("mime_type must be an image media type")
        metadata = {
            "source": path.name,
            "filename": path.name,
            "captured_at": arguments.captured_at.strip(),
            "reference_time": datetime.now(UTC).isoformat(),
            "modality": "image",
        }
        result = ingest_image(
            self.pipeline,
            image=path.read_bytes(),
            text=arguments.text,
            mime_type=mime_type,
            metadata=metadata,
        )
        return IngestImageOutput(
            item_id=result["item_id"],
            modality=result["modality"],
            extraction=IngestExtractionReport(**result["extraction"]),
            warning=result["warning"],
        )


def create_tool() -> BaseTool:
    return IngestImageTool()


__all__ = [
    "MIME_BY_SUFFIX",
    "TEXT_ONLY_WARNING",
    "IngestExtractionReport",
    "IngestImageInput",
    "IngestImageOutput",
    "IngestImageTool",
    "create_tool",
    "guess_mime_type",
    "ingest_image",
]
