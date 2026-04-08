# utils/mcp_manager.py
import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

class MCPClientManager:
    def __init__(self):
        self.session = None
        self._exit_stack = AsyncExitStack()

    async def connect(self, server_script_path: str):
        """로컬 MCP DB 서버에 stdio 방식으로 연결합니다."""
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env={"PYTHONPATH": "."} # 프로젝트 루트를 경로에 추가
        )
        
        # stdio 통신 채널 설정
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        
        # 세션 초기화
        await self.session.initialize()
        return self.session

    async def disconnect(self):
        """연결된 모든 리소스를 안전하게 해제합니다."""
        await self._exit_stack.aclose()
        self.session = None

    async def connect_as_module(self, module_name: str):
        """파이썬 모듈 모드(-m)로 MCP 서버를 실행하고 연결합니다."""
        server_params = StdioServerParameters(
            command=sys.executable, # 현재 사용 중인 python.exe 경로
            args=["-m", module_name], # 예: -m utils.mcp_db_server
            env=os.environ.copy()    # 현재 환경 변수 상속
        )
        
        # stdio 통신 설정
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        
        await self.session.initialize()    

mcp_manager = MCPClientManager()