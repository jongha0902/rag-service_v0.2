# mcp_db_server.py
from mcp.server.fastmcp import FastMCP
from utils.db import get_oracle_conn
from utils.db_full_schema import get_all_table_names, search_db_metadata
import logging

# MCP 서버 초기화
mcp = FastMCP("Oracle-SQLite-DB-Manager")
logger = logging.getLogger(__name__)

@mcp.tool()
async def list_all_tables() -> str:
    """Oracle 데이터베이스의 모든 테이블 목록과 설명을 가져옵니다."""
    try:
        return get_all_table_names()
    except Exception as e:
        return f"테이블 목록 조회 중 오류 발생: {str(e)}"

@mcp.tool()
async def search_metadata(keyword: str) -> str:
    """입력된 키워드와 관련된 테이블, 컬럼, 소스코드를 검색합니다."""
    # utils/db_full_schema.py의 기존 함수 활용
    return search_db_metadata(keyword)

@mcp.tool()
async def execute_oracle_query(sql_query: str) -> str:
    """Oracle DB에서 읽기 전용(SELECT) SQL 쿼리를 직접 실행합니다."""
    # 보안을 위해 SELECT 쿼리인지 기본 체크
    if not sql_query.strip().upper().startswith("SELECT"):
        return "보안상 SELECT 쿼리만 실행 가능합니다."
        
    try:
        with get_oracle_conn() as conn: # utils/db.py의 연결 함수 활용
            cursor = conn.cursor()
            cursor.execute(sql_query)
            # 최대 50개 행만 반환하도록 제한
            rows = cursor.fetchmany(50)
            if not rows:
                return "결과가 없습니다."
            
            # 컬럼 헤더 추출
            columns = [col[0] for col in cursor.description]
            result = [dict(zip(columns, row)) for row in rows]
            return str(result)
    except Exception as e:
        return f"SQL 실행 중 오류 발생: {str(e)}"

if __name__ == "__main__":
    mcp.run()