# amie-gcp

AI 专利发明评估平台. LangGraph 5-phase pipeline, 部署在 GCP Cloud Run.

## Tech Stack
Backend: Python / FastAPI+Uvicorn / LangGraph / Gemini (Vertex AI) / GCS / SerpAPI + OpenAlex + Semantic Scholar
Frontend: TypeScript / Vite / Express proxy (Node 20)
Deploy: Cloud Run (us-west1), GCP project: aime-hello-world

### Python & 依赖 pin (2026-09-18, M2)
- **本地开发和跑测试一律用 `python3` = python3.14 (Homebrew)**。这是 `pytest tests -q` 308 passed 的参考环境。
- **不要用 python3.13** 跑这个 repo: 那个环境装的是 langgraph 0.6.4，而本仓库 pin 的是 **0.2.76**。
  两者行为差异: 0.6.x 的 `graph.ainvoke()` 在 gate interrupt 时把 `Interrupt` 列表放进结果 dict 的
  `__interrupt__` 键，0.2.x 不放（只能从 `graph.get_state(config).tasks[*].interrupts` 读）。
  `tests/test_graph_structure.py` 的 `_interrupt_of` 已同时兼容两者 (75d90ee)。
  `graph.astream(stream_mode="updates")`（`app/main.py` 实际走的路径）**两个版本都会**发
  `{"__interrupt__": (...)}` update — 已实测验证，所以服务端 HITL 不受影响。
- 精确版本在 `backend/requirements.txt`；`pyproject.toml` 只留下界。Cloud Run 镜像 (python 3.12)
  装的就是同一个 requirements.txt。

## Architecture
LangGraph StateGraph with 5 nodes + conditional routing:
- **IDCA** (Phase 1): PDF 读取 + 发明检测 + 分类 + 摘要 + persona 生成
- **SSR** (Phase 2): 创新轴分析 → 技术选择 → checklist 生成 → 搜索查询规划
- **Search** (Phase 3): 5 通道并行搜索 (SerpAPI×2 + SS + OA + arXiv) → 语义 rerank → PDF 下载
- **Evaluate** (Phase 4): 逐文档深度评估 vs checklist → 评分排序 → §103 组合分析
- **Report** (Phase 5): 生成 HTML/Markdown 报告 → 上传 GCS → 邮件通知
- 条件路由: Absent/无发明 → 跳到 Report; 搜索结果为空 → 跳到 Report

## Key Files
- `backend/state.py` — GraphState TypedDict + Annotated reducers
- `backend/graph/main_graph.py` — LangGraph 连线 + 条件路由
- `backend/nodes/{idca,ssr,search,evaluate,report}.py` — 5 个 pipeline node
- `backend/app/main.py` — FastAPI 入口 + A2A + job 管理 (原 patent-analyzer, 逐步迁移到 LangGraph)
- `backend/app/llm.py` — Gemini 调用层 (call_llm, call_llm_with_pdfs, self_check, evaluate_batch)
- `backend/patent_analyzer/` — 搜索源 (recall/)、query builder、report generator、scorer
- `frontend/amie/server.js` — Express proxy (IAM 认证)

## Commands
```bash
# Backend 本地
cd backend
PYTHONPATH=$(pwd) uvicorn app.main:app --reload --port 8000

# Frontend 本地
cd frontend/amie
BACKEND_URL=http://localhost:8000 BACKEND_ENV=dev npm run serve

# Cloud Run 部署 (backend)
cd backend
gcloud run deploy patent-analyzer --source . --region us-west1 \
  --service-account amie-backend-sa@aime-hello-world.iam.gserviceaccount.com \
  --env-vars-file .env.yaml --no-allow-unauthenticated
```

## Gotchas
1. **State 不可变**: node 必须 return patch dict, 不能 in-place mutate GraphState
2. **前端不直连后端**: 所有请求经 Express proxy, 生产环境 proxy 从 metadata server 拿 IAM token
3. **本地开发**: BACKEND_ENV=dev 跳过 IAM 认证
4. **GCS Signed URL**: 客户端直传 GCS, 不经后端, 避免大文件瓶颈
5. **SerpAPI throttle**: 共享 Semaphore(1) + 1.5s cooldown, 防止 429
6. **LLM 调用数量受控**: 确定性步骤 (detect_invention, classify_document) 不用 LLM
7. **events 不进 checkpoint**: LLM prompt/response 通过 events side-channel, 不序列化到 checkpoint

## Roadmap
- [x] P0: 代码搬迁 + LangGraph 线性骨架
- [ ] P1: Subgraph + Map-Reduce (Phase 4 Send API) + Self-loop + Checkpoint
- [ ] P2: HITL 后端 (interrupt_after + Command resume)
- [ ] P3: HITL 前端 (toggle + checklist 审核 UI + ABCD 选择 + 对话历史)
- [ ] P4: Streaming tokens (SSE)
