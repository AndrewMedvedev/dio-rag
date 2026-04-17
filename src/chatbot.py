import os
from uuid import UUID

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .depends import llm, summarization_middleware
from .rag import retrieve
from .settings import SQLITE_PATH

PROMPT = """
Данные из RAG:
{rag_data}

Запрос пользователя:
{user_prompt}
"""


async def call_chatbot(user_id: UUID, user_prompt: str, chat_id: UUID) -> str:
    """Вызов чат-бот агента для диалога со студентом в рамках его учебного прогресса"""

    async with AsyncSqliteSaver.from_conn_string(os.fspath(SQLITE_PATH)) as checkpointer:
        await checkpointer.setup()
        agent = create_agent(
            model=llm,
            middleware=[summarization_middleware],
            checkpointer=checkpointer,
        )
        rag = await retrieve(
            query=user_prompt,
            metadata_filter={"tenant_id": str(user_id), "chat_id": str(chat_id)},
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=PROMPT.format(rag_data=rag, user_prompt=user_prompt))
                ],
            },
            config={"configurable": {"thread_id": f"{user_id}"}},
        )
    return result["messages"][-1].content
