# OCR 服务层重构方案（草案）

## 目标与范围
- 保留现有 Web UI（用户体验不变，必要时调整前端请求），升级服务层为高并发、生产级、可扩展。
- 服务框架升级为 ASGI（FastAPI/Starlette）。
- 不再保持对现有 Flask 路由层的兼容，直接切换到新的 API；但“ONNX 推理层（onnxocr/*.py）”不做改动，仅做封装与复用。
- 部署与监控从简：仅使用 Docker 构建与运行，暴露 5005；不引入数据库、反向代理、观测/遥测；仅提供完善日志与等级控制。

## 技术选型
- Web/API：FastAPI（ASGI，Pydantic 校验）。
- 服务器：Uvicorn（可选启用 uvloop、httptools）；生产使用 Gunicorn+Uvicorn workers。
- 模板与静态：Jinja2 + Starlette StaticFiles（复用 `templates/`、`static/`）。
- 日志：Python 标准 `logging` + Uvicorn/FastAPI 统一日志配置（可设定等级、格式、轮转）。

## 架构概览
- 入口：`app/main.py`（ASGI）。Mount `/` 提供 Web UI 与静态资源，`/api` 提供 API。
- 业务：`onnxocr_runtime/engine.py` 封装对 `onnxocr/` 中现有推理代码的调用（不改动原实现），支持多模型与并发门控。
- 路由：
  - `routes/v1.py`：旧接口（`POST /ocr`，base64 JSON），保证字段与格式100%兼容。
  - `routes/v2.py`：新接口（`/api/v2/ocr` 等），Web UI 通过前端适配使用新接口。
- 任务：BackgroundTask + 内存任务索引（轻量级）；提供任务 ID、查询与取消（不中断长任务）。
- 配置：`settings.py` 统一读取环境变量（12-factor）。

### 目标目录结构（新）
```
app/
  main.py            # FastAPI 实例与挂载
  settings.py        # 环境配置
  logging.py         # 日志配置
  middleware.py      # 日志/请求ID/异常中间件
  engine.py          # 推理引擎封装与池化
  routers/
    v2.py            # /api/v2/ocr, /tasks, /healthz, /readyz, /download
  ui.py              # Web UI 路由（渲染 templates）
docs/
  SERVICE_REDESIGN.md
templates/           # 复用现有
static/              # 复用现有
results/             # 运行时生成
```

## 现状评估（基于当前仓库代码）
- Web 服务：
  - `app-service.py` 提供 Flask JSON API `POST /ocr`，接收 base64，返回 `processing_time` 与 `results`（含文本、置信度、四点框）。
  - `webui.py` 提供批量上传 UI（Flask），基于 `onnxocr/ocr_images_pdfs.py` 的 `OCRLogic` 批量处理，输出至每次会话目录 `results/<timestamp>/Output_OCR`，并打包 txt 为 zip 提供下载。
- OCR 推理：
  - `onnxocr/onnx_paddleocr.py` 提供 `ONNXPaddleOcr`（继承 `TextSystem`），`ocr(img, det=True, rec=True, cls=True)` 返回兼容 PaddleOCR 的结构：`[[points, [text, score]], ...]`。
  - `onnxocr/predict_system.py` 管线：det -> (crop -> cls?) -> rec，阈值过滤由 `args.drop_score` 控制；`args.det_box_type in {quad, poly}` 影响裁剪方式。
  - `onnxocr/utils.py.infer_args()` 定义了大量默认参数与模型路径（默认 ppocrv5），关键项：`use_angle_cls`、`rec_image_shape`、`rec_char_dict_path`、`det/rec/cls_*` 路径、`drop_score`、`cpu_threads`；不应更改默认值来源。
- 依赖：`requirements.txt` 含 Flask、opencv、onnxruntime-gpu、pymupdf、pdf2image 等；Dockerfile 基于 `python:3.7-slim`，安装 `libgl1` 等运行时库；容器启动 `python app-service.py`，暴露 5005。

结论：推理层已稳定，I/O 与 API 层为重构重点；UI 需改为调用新接口但前端结构可保持。

## 依赖与版本建议（pyproject）
- 运行核心：
  - `fastapi>=0.111`、`uvicorn[standard]>=0.30`、`gunicorn>=21`、`python-multipart>=0.0.9`、`jinja2>=3.1`
  - 继续沿用：`onnxruntime-gpu==1.14.1`（或 `onnxruntime` 依据硬件）、`opencv-*`、`numpy<2.0.0`、`pymupdf`、`pdf2image`、`shapely`、`pyclipper` 等。
- 开发：`httpx`, `pytest`（如添加测试）。
- 迁移：逐步移除 `flask` 与 `gunicorn` 冲突项（gunicorn 保留）。


## API 设计（保留旧接口 + 新接口）
- 旧接口（v1，兼容保留）：
  - `POST /ocr` Content-Type: `application/json`
    - 请求：`{"image": "<base64>"}`
    - 响应：保持现有字段与结构完全一致：`{"processing_time": float, "results": [{"text": str, "confidence": float, "bounding_box": [[x,y],...]}]}`。
  - 备注：错误响应亦需与现有实现保持字段一致（400/500 时的 `{"error": "..."}`）。
- 新接口（v2，推荐）：
  - `POST /api/v2/ocr`（推荐）Content-Type: `multipart/form-data`
  - `file`: 必填，图片或单页 PDF；或 `files[]` 多文件（可返回打包下载链接）。
  - `model_name`: `PP-OCRv5 | PP-OCRv4 | ch_ppocr_server_v2.0`（默认 PP-OCRv5）。
  - `conf_threshold`: 浮点，默认 0.5。
  - `output_format`: `json|text|tsv|hocr`（默认 json）。
  - `return_image`: `true|false`（默认 false，是否返回渲染可视化）。
  - 可选：`language`、`angle_cls`、`det_only`、`rec_only`、`bbox=true|false`。
- `GET /api/v2/tasks/{id}`：查询异步任务状态与结果。
- `GET /api/v2/stream/{id}`（SSE，可选）：流式返回进度，不依赖外部基础设施。
- `GET /healthz` 与 `GET /readyz`：存活与就绪检查。
- Web UI：保持页面与交互不变，但前端请求切换为调用 `/api/v2/ocr`。
  - 同时保留旧接口 `/ocr` 以确保外部系统平滑过渡。

请求/响应详解：
- `POST /api/v2/ocr`（multipart/form-data）
  - 入参：
    - `file` 或 `files[]`: 单文件或多文件；支持图片（jpg/png/bmp）、单页 PDF；中文路径通过二进制上传避免编码问题。
    - `model_name`: 同 `OCRLogic.set_model` 的 key（`PP-OCRv5|PP-OCRv4|ch_ppocr_server_v2.0`）。
    - `conf_threshold`: 覆盖 `args.drop_score`，范围 [0,1]；默认 0.5。
    - `output_format`: `json|text|tsv|hocr`（默认 json）。
    - `bbox`: `true|false`，是否包含四点框。
    - `return_image`: `true|false`，是否返回绘制结果图（base64 或下载链接）。
  - 成功响应（json）：
    - 单文件：`{ processing_time, results: [{text, confidence, bounding_box?}], preview_image? }`
    - 多文件：`{ processing_time, items: [ { filename, results | text }, ... ], zip_url? }`
  - 失败响应：`{ error: string, code: string }`，HTTP 400/415/500；详见错误码规范。
- `GET /api/v2/tasks/{id}`：`{ status: queued|running|done|error, progress, result?, error? }`
- `GET /download/{timestamp}`：返回 zip；路径规则与现有 `webui.py` 一致（`results/<timestamp>/ocr_txt_<timestamp>.zip`）。

示例请求/响应：
```bash
curl -X POST http://localhost:5005/api/v2/ocr \
  -F "file=@onnxocr/test_images/1.jpg" \
  -F "model_name=PP-OCRv5" -F "conf_threshold=0.6" -F "output_format=json" -F "bbox=true"
```
```json
{
  "processing_time": 0.456,
  "results": [
    {"text": "Name", "confidence": 0.9999, "bounding_box": [[4,8],[31,8],[31,24],[4,24]]},
    {"text": "Header", "confidence": 0.9998, "bounding_box": [[233,7],[258,7],[258,23],[233,23]]}
  ],
  "preview_image": null
}
```
多文件示例：
```bash
curl -X POST http://localhost:5005/api/v2/ocr \
  -F "files[]=@a.jpg" -F "files[]=@b.jpg" -F "output_format=text"
```
```json
{
  "processing_time": 1.234,
  "items": [
    {"filename": "a.jpg", "text": "...\n..."},
    {"filename": "b.jpg", "text": "...\n..."}
  ],
  "zip_url": "/download/20250101_120000"
}
```

## 并发与性能策略
- 进程/线程：Gunicorn `--workers n`（CPU 核数或 2*核数），`--threads m`（I/O/图像读写）。
- Uvicorn：启用 `--loop uvloop --http httptools`，减少事件循环与 HTTP 解析开销。
- ONNXRuntime：按模型类型配置 `intra_op`/`inter_op` 线程；进程级预热（Linux 可用 `preload_app` 共享 COW）。
- 模型池：每 worker 维护 1~k 个会话实例（受内存限制）；基于 `semaphore` 做推理并发门控，避免抖动。
- 批处理：PDF/多图分页并行（受限于 CPU/内存）；对外提供队列长度与排队时间。
- 资源限制：最大上传（如 50MB）、并发阈值、请求超时、CPU/内存水位告警。

实现要点（结合本仓库）：
- ONNXRuntime 会话：当前在 `PredictBase.get_onnx_session()` 中构造 `InferenceSession`；保持不变。为避免过度并发竞争，ASGI 层使用 `asyncio.Semaphore(K)` 控制每进程最大并发推理数（K 可由 `MODEL_CONCURRENCY` 配置）。
- 线程模型：Gunicorn worker 数量由 CPU 决定（建议 `workers = min(4, CPU*2)`），`threads` 用于 I/O 处理；推理由 onnxruntime 自身线程池管理。
- 预热：启动时对默认模型执行一次空跑（读取一张小图或构造假输入）以加载权重，降低首包时延；可配置开关 `WARMUP=true|false`。

## 稳定性与可观测（约简方案）
- 健康检查：`/healthz`（存活）与 `/readyz`（模型加载完成），便于本地与 Docker HEALTHCHECK 使用。
- 日志系统：仅使用标准日志（不接入外部监控）。按等级输出：DEBUG/INFO/WARNING/ERROR。
  - 统一格式：`%(asctime)s %(levelname)s %(name)s %(message)s`。
  - 请求范围日志：包含 `request_id`（若头中提供 `X-Request-ID` 则透传）。
  - 可选 JSON 格式（仍使用标准库，无第三方依赖）。
  - 中间件：记录方法、路径、状态码、耗时、体积；限长打印用户输入摘要（避免日志泄露 base64 与大文件）。

## 安全与治理
- 输入校验与限制（文件类型、分辨率、页数、大小）。
- 可选认证（API Key/Bearer Token）与 CORS（默认关闭或最小放行）。
- 不引入数据库与外部依赖；崩溃自动恢复由容器编排层承担（如 Docker restart 策略）。

## 日志设计
- 输出目标：默认标准输出（stdout），便于 `docker logs` 直接查看。
- 等级控制：通过环境变量 `LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` 控制应用与 Uvicorn 日志等级。
- 格式：`%(asctime)s %(levelname)s %(name)s %(message)s`；包含请求方法、路径、状态码与耗时（由中间件记录）。
- 请求关联：从请求头读取 `X-Request-ID`，若不存在则生成，并注入到日志上下文。
- 文件落地（可选）：如需持久化，挂载卷并启用 `RotatingFileHandler`（仍使用标准库）。

## 迁移与落地步骤（不保留 Flask 路由兼容）
1) 新增 `pyproject.toml` 与 `uv.lock`；以 `uv` 管理依赖，逐步弃用 `requirements.txt`（过渡期保留）。
2) 创建 ASGI 应用：挂载 Web UI（`templates/`、`static/`），实现 `/ocr`（v1 兼容）与 `/api/v2/*` 路由；前端改为调用新接口。
3) 引擎封装：基于 `onnxocr` 现有模块实现 `engine.py`，统一模型管理、阈值与并发门控（不改动推理核心）。
4) 完成健康检查与统一日志中间件；准备 Docker 镜像与启动脚本。
5) 回归与压测：功能正确性（单图/多图/PDF）、并发稳定性、长任务不中断；按测试结果调优 worker/thread 与模型池参数。
6) 文档更新与切换：更新 README 与 Web UI 调用说明；切换生产镜像。

### 详细改造清单（分任务/文件级）
1. 新建 ASGI 应用骨架（不改动 `onnxocr/*.py`）
   - `app/main.py`: 创建 FastAPI 实例；挂载静态与模板；注册路由；安装中间件（日志、CORS、异常处理、request_id、限流可选）。
   - `app/settings.py`: 从环境读取配置（端口、workers、model 默认、阈值、最大上传 MB、日志等级/格式、池大小、是否预热）。
   - `app/logging.py`: 统一 `logging.config.dictConfig`，覆盖 Uvicorn/FastAPI loggers；支持 `LOG_LEVEL` 与可选 JSON 格式；提供 `get_logger(name)`。
   - `app/middleware.py`: 请求 ID 注入、访问日志、异常归一化（捕获未处理异常，返回 `{code,message}`）。
   - `app/engine.py`: 轻量封装现有 `ONNXPaddleOcr`：
     - `EngineManager`：按 `model_name` 管理模型实例，维护 `asyncio.Semaphore`；支持 `get(model_name, drop_score, use_gpu)`；惰性加载与预热；线程安全（进程内）。
     - `run_ocr(image: np.ndarray, params) -> OCRResult`：调用 `model.ocr(img)` 并按 `conf_threshold` 过滤（如需覆盖 `drop_score`）。
   - `app/routers/v2.py`: `POST /api/v2/ocr`、`GET /api/v2/tasks/{id}`、`GET /healthz`、`GET /readyz`、`GET /download/{ts}`。
   - `app/ui.py`: 复用 `templates/` 与 `static/`，重写原 `webui.py` 的 Flask 视图为 FastAPI 模式；上传逻辑改为调用内部 `engine`。

2. 表单解析与输出格式化
   - 解析 `multipart/form-data`（依赖 `python-multipart`）；限制单文件与总大小（由 `MAX_UPLOAD_MB` 控制）。
   - 支持多文件并行处理（`ThreadPoolExecutor`/`asyncio.to_thread`），顺序写回。
   - 输出格式：
     - json：与现有结构兼容，保留 `text/confidence/bounding_box`；
     - text：使用 `OCRLogic._result_to_text` 的等价逻辑（复制为工具函数，避免导入 UI 逻辑类）；
     - tsv/hocr：新实现（简化版，可选）。
   - 可选返回绘制图：调用 `sav2Img`，存储至会话目录并返回链接或 base64（默认链接）。

3. 任务与会话目录
   - 目录结构：`results/<timestamp>/`；与现有 `webui.py` 一致；子目录 `Output_OCR` 用于 txt/可视化。
   - 多文件时打包 zip：`ocr_txt_<timestamp>.zip`，路由 `/download/{timestamp}` 返回。
   - 轻量任务表：`TaskStore`（内存字典）记录状态/进度/结果；进程内有效（容器重启后不保留）。

4. 健康检查
   - `/healthz`: 返回 200 + `{"status":"ok"}`；
   - `/readyz`: 在默认模型加载完成后返回 200，否则 503；依赖 `EngineManager.ready` 状态。

5. 日志与错误码
   - 错误码规范：
     - `VALIDATION_ERROR`(400): 文件/参数不合法；
     - `UNSUPPORTED_MEDIA_TYPE`(415): 文件类型不支持；
     - `INFERENCE_ERROR`(500): 模型推理失败；
     - `INTERNAL_ERROR`(500): 未知异常。
   - 统一异常处理：将异常转换为上述结构，日志按 ERROR 级别记录 traceback（可裁剪）。

6. Web UI 适配
   - `templates/webui.html` 的提交地址改为 `/api/v2/ocr`，并适配新增参数（模型、阈值、输出格式、是否返回可视化）。
   - 前端展示保持不变；批量结果展示使用后端返回的 `results` 或合并文本。

7. Docker 与运行
   - 更新 `Dockerfile`（示例）：
     - 基于 `python:3.11-slim`；安装 `uv`（`pip install uv`）；安装运行库（`libgl1` 等）。
     - 拷贝 `pyproject.toml` 与 `uv.lock` 先于源码以命中依赖缓存；`uv pip install -r requirements.txt` 过渡期兼容。
     - CMD：`gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5005 --workers ${WORKERS:-4} --threads ${THREADS:-2} --preload`。

8. 配置项（环境变量，全部可选）
   - `WORKERS`、`THREADS`、`MAX_UPLOAD_MB`、`MODEL_POOL_SIZE`、`MODEL_CONCURRENCY`、`DEFAULT_MODEL`、`LOG_LEVEL`、`WARMUP`。
   - 若未设置，程序根据硬件与启动校准结果自动推导合理值（见“自适应与自动调优”）。

9. 兼容性与回退
   - 推理层完全复用；若 v2 变更影响 UI，可在 UI 中提供“旧版 API 说明”链接作为过渡文档。
   - 回退策略：保留旧镜像 tag 与启动命令，一键回滚容器版本。

### 代码约束（不该做什么）
- 不修改 `onnxocr/*.py` 的任何行为（仅封装与调用）。
- 不引入数据库、消息队列、外部监控/遥测；不新增反向代理依赖。
- 不在日志中打印完整输入图像/大段 base64；不记录敏感数据。
- 不在单进程内创建过多 ONNX 会话实例（避免内存膨胀）。



## Docker 与部署建议
- 多阶段构建，基础镜像 `python:3.7-slim`
- 容器内直接运行应用并暴露 `5005`；不依赖外部反向代理或数据库。
- 环境变量（可选）：`WORKERS`、`THREADS`、`MAX_UPLOAD_MB`、`MODEL_POOL_SIZE`、`LOG_LEVEL`。未设置时由程序自适应推导。
  - 启动命令

## 验收与目标（简化）
- 兼容性：`/ocr` 完全兼容现有输入/输出；Web UI 正常工作。
- 部署：Docker 单容器，暴露 `5005`，无外部依赖；在无任何环境变量时也能自动选择合理并发与池化参数。
- 日志：可按环境变量设置等级；包含时间、级别、模块、信息与 request_id。
- 性能：在目标硬件上完成预期并发测试（自测脚本即可），维持稳定。

## 测试计划（落地可执行）
- 单元：
  - `engine.run_ocr` 对单张图片的输出字段一致性（text/confidence/bbox）；阈值过滤正确性（drop_score 覆盖）。
  - 参数解析：`model_name/conf_threshold/output_format/bbox/return_image` 的默认与边界值。
- 集成：
  - `POST /api/v2/ocr` 单文件/多文件、图片/PDF、错误类型（空文件、超限、格式不支持）。
  - `POST /ocr`（v1）字段级兼容性：对同一张图片，重构前/后响应在字段名、层级、类型、值域上100%一致（容许浮点微小差异）。
  - 并发：使用 `ab`/`wrk` 或自制脚本并发请求，观察 `P95` 与错误率。
  - 会话产物：生成 `Output_OCR`、zip，下载链接可用。
  - 健康检查：`/healthz` 200，`/readyz` 在未加载前 503，加载后 200。
- 回归：
  - 与 `test_ocr.py` 结果对齐（取一张样例图对比文本与框数目）。
  - 零配置启动：环境变量全部缺省时，应用仍可用，并发/池化参数由程序自动推导，性能稳定。

## 性能与资源预算（建议）
- 典型配置（4C8G）：`workers=4, threads=2, MODEL_POOL_SIZE=1-2`；
- 内存：每个模型会话约数百 MB（视模型），模型池需谨慎；
- 首包时延：预热后显著降低；
- CPU：开启 `uvloop` 与 `httptools` 降低开销（如平台支持）。

## 风险与缓解
- 多进程模型内存占用高：使用 `preload_app`、合理池化与模型按需加载。
- 大文件与超时：切换后台任务与 SSE，设置合理超时与重试策略。
- Windows 与 Linux 差异：部署以 Linux 为主，开发兼容 Windows 运行方式。

## 参考实现片段（便于快速落地）
- 日志配置（dictConfig 片段）：
```python
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.config.dictConfig({
  'version': 1,
  'formatters': {'default': {'format': LOG_FORMAT}},
  'handlers': {'console': {'class': 'logging.StreamHandler','formatter': 'default','stream': 'ext://sys.stdout'}},
  'root': {'level': LOG_LEVEL, 'handlers': ['console']},
  'loggers': {'uvicorn.access': {'propagate': True}, 'uvicorn': {'propagate': True}}
})
```
- 推理并发门控（简化示例）：
 - 推理并发门控（简化示例）：
```python
class EngineManager:
  def __init__(self, pool_size=1, concurrency=1, default_model='PP-OCRv5'):
    self._lock = asyncio.Lock(); self._models = {}; self._sem = asyncio.Semaphore(concurrency)
    self._default = default_model
  async def get(self, model_name=None, **kwargs):
    name = model_name or self._default
    async with self._lock:
      if name not in self._models:
        self._models[name] = ONNXPaddleOcr(**self._resolve_kwargs(name, **kwargs))
      return self._models[name]
  async def run(self, img, model_name=None, conf_threshold=None):
    async with self._sem:
      model = await self.get(model_name)
      t0=time.time(); res=model.ocr(img); dt=time.time()-t0
      # 按需过滤/重排结果...
      return dt, res
```

- Pydantic Schema（路由契约）：
```python
class OCRParams(BaseModel):
  model_name: Literal['PP-OCRv5','PP-OCRv4','ch_ppocr_server_v2.0'] = 'PP-OCRv5'
  conf_threshold: conint(ge=0, le=1) | confloat(ge=0, le=1) = 0.5
  output_format: Literal['json','text','tsv','hocr'] = 'json'
  bbox: bool = True
  return_image: bool = False

class OCRBox(BaseModel):
  text: str
  confidence: float
  bounding_box: list[list[float]] | None = None

class OCRResponse(BaseModel):
  processing_time: float
  results: list[OCRBox] | None = None
  items: list[dict] | None = None
  zip_url: str | None = None
```

- 异常与响应包装：
```python
class APIError(Exception):
  def __init__(self, code: str, status: int, message: str):
    self.code, self.status, self.message = code, status, message

def error_handler(request, exc: APIError):
  logger.error("api_error", extra={"code":exc.code,"path":request.url.path})
  return JSONResponse({"error": exc.message, "code": exc.code}, status_code=exc.status)
```

## 示例（v2 调用）
```bash
curl -X POST http://localhost:5005/api/v2/ocr \
  -F "file=@./onnxocr/test_images/1.jpg" \
  -F "model_name=PP-OCRv5" \
  -F "conf_threshold=0.6" \
  -F "output_format=json" \
  -F "bbox=true"
```

---
备注：本文档为实施蓝图，落地时将产出对应的代码结构（app/main.py、routes、engine、settings）、配置模板与部署脚本，并附带兼容性与性能回归用例。

## 编码规范与变更纪律（补充）
- 基础规范：
  - 遵循 PEP8；4 空格缩进；`snake_case`（变量/函数）、`PascalCase`（类）；模块文件名小写。
  - 严禁在代码、返回 JSON、UI 文本中加入 emoji 或特殊装饰字符。
  - 变量命名清晰，不使用无语义的单字母（循环索引可例外）。
- 注释与文档：
  - 禁止堆砌无用注释；注释聚焦“为什么这样做”和“边界/权衡”。
  - 公共函数、路由、数据模型需有简明 docstring；变更需同步更新文档。
- 日志：
  - 统一使用 `app/logging.py` 配置；等级仅限 DEBUG/INFO/WARNING/ERROR。
  - 日志不得记录敏感信息、完整 base64 或超大 payload；Debug 字段仅在控制台/日志出现，禁止出现在 API 响应。
- 变更联动检查（每次提交前执行）：
  - 文档：README、docs/SERVICE_REDESIGN.md 是否需同步更新？
  - 部署：Dockerfile、启动命令、环境变量是否需同步更新？
  - 协议：Web UI 与 API 参数/字段是否一致？是否影响外部依赖方？
  - 兼容：旧接口 `/ocr` 是否仍100%兼容？
- 清理旧代码：
  - 先做引用检索（`rg`/IDE 全局搜索）；分阶段删除并在 PR 说明影响面；必要时保留适配层有限期。
- 接口与 UI 设计：
  - 坚持“简洁、稳定、高可用”；字段语义直观、默认合理、可覆盖；避免过度参数化与复杂交互。
  - 错误信息面向用户友好，不暴露栈信息或内部细节。
- API 兼容要求：
  - 新旧接口并行：`POST /ocr`（base64 JSON）完整保留并实现 100% 字段级兼容；`/api/v2/ocr` 为新增推荐接口。

## 简洁性与可复用原则
- 小而清晰：函数保持单一职责，长度适中；复杂流程拆分为可单测的私有辅助函数。
- 复用优先：当逻辑在两处以上出现或可预见通用时，抽取为工具函数/类（如 `app/utils.py` 或 `engine.py` 内部私有方法）。避免复制粘贴代码。
- 类型与契约：为公共函数、路由、数据模型添加类型标注（typing/Pydantic）；必要时使用 `dataclasses`/`TypedDict` 提升可读性。
- 避免过度抽象：仅在出现明确复用边界时抽象；不要为一次性逻辑设计笨重层次。
- 公共接口最少化：模块对外仅暴露必要入口；内部实现以 `_` 前缀标记私有，便于重构。
 - 避免写死数值：并发、线程、池大小、上传限制等默认值均通过 `settings` 的自适应辅助函数计算，不在业务代码中硬编码常量。

## 库选择与实现策略
- 充分利用标准库：并发（`concurrent.futures`/`asyncio`）、路径（`pathlib`）、临时文件（`tempfile`）、压缩（`zipfile`）、ID（`uuid`）、时间（`time`/`datetime`）、日志（`logging`）、base64（`base64`）。
- 采用成熟维护的三方库：路由/校验（FastAPI/Pydantic）、ASGI（Uvicorn/Gunicorn）、表单解析（python-multipart）、模板（Jinja2），不要自行实现解析器或路由。
- 图像/PDF：继续使用 OpenCV、PyMuPDF/pdf2image，不自行实现底层编解码与转换算法。
- 不自造轮子：若标准/成熟库可满足需求，优先使用；仅在确无合适库且实现极简的情况下自实现。
- 版本与锁定：关键依赖版本显式声明；避免引入边缘或无人维护的库。

## 自适应与自动调优
- 目标：在不设置任何环境变量的情况下自动获得“合理且稳定”的性能参数；允许通过环境变量覆盖。
- 机制（建议实现）：
  - CPU 并发：`os.cpu_count()` 结合短暂冷/热启动基准（1～N 次小样本推理）估测最佳 `workers/threads`，目标 CPU 利用率 ~70%，避免过拟合特定硬件。
  - 模型池大小：加载一次模型，测量内存占用（`psutil`/`resource`）；根据可用内存估算池大小上限，保证预留系统空间（例如 20% 安全余量）。
  - 上传大小：根据可用内存与图像解码开销计算合理上限；如未设置，给出保守但可用的默认（通过算法推导而非硬编码常量）。
  - 运行时自检：启动日志输出最终采用的 workers/threads/池大小/上传限制等，供运维确认。

## 一次落地与兼容性验收（DoD）
- 可运行/可部署：
  - Docker 构建成功；容器启动暴露 5005；环境变量生效；无外部依赖。
  - Gunicorn+Uvicorn 启动成功，日志按等级输出；健康检查可用（/healthz, /readyz）。
- 兼容性：
  - 旧接口 `/ocr`（base64）字段级对齐旧实现；用同一输入对比重构前后响应（误差仅限浮点小数位）。
  - 推理层 `onnxocr/*.py` 未改动；`test_ocr.py` 可继续工作（或提供等效脚本）。
- 新接口可用：
  - `/api/v2/ocr` 支持单/多文件、图片/PDF、阈值/模型/输出格式参数；返回约定的 JSON/文本/zip 链接。
  - Web UI 指向 v2 正常使用；多文件合并与下载可用。
- 并发稳定：
  - 具备并发门控（Semaphore）与模型池配置；在目标硬件上完成并发自测（脚本即可），无崩溃与资源泄漏。
- 文档同步：
  - README 与本方案同步更新；Docker 启动命令、环境变量、接口文档与 UI 使用指引齐备。
