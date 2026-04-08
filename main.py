import logging
import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from utils.mcp_manager import mcp_manager
from utils.api.rag import router as rag_router
from utils.ollama_rag import initialize_all_vectorstores

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 [통합 서비스] 앱 시작: 벡터 스토어 및 리소스 초기화...")

    # 벡터 스토어 초기화 with error handling
    try:
        await asyncio.to_thread(initialize_all_vectorstores)
        logger.info("✅ 벡터 스토어 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 벡터 스토어 초기화 실패: {e}")
        # 실패 시에도 앱은 실행되지만 로그 기록

    # MCP 서버 연결
    try:
        await mcp_manager.connect_as_module("utils.mcp_db_server")
        logger.info("✅ MCP DB 서버 연결 성공")
        app.state.mcp_connected = True
    except Exception as e:
        logger.error(f"❌ MCP 연결 실패: {e}")
        app.state.mcp_connected = False

    yield
    
    # 종료 처리
    try:
        await mcp_manager.disconnect()
        logger.info("🛑 MCP 연결 종료")
    except Exception as e:
        logger.error(f"❌ MCP 종료 중 오류: {e}")
    
    logger.info("🛑 앱 종료 완료")

app = FastAPI(lifespan=lifespan, title="Unified Smart RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)