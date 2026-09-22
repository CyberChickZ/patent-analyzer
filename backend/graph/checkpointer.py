"""Checkpoint configuration.

Development: MemorySaver (in-process, no persistence across restarts)
Production: Switch to PostgresSaver when Cloud SQL is provisioned.

Usage:
    from graph.checkpointer import get_checkpointer
    graph = build_graph(checkpointer=get_checkpointer())
"""

import os


def get_checkpointer():
    backend = os.environ.get("CHECKPOINT_BACKEND", "memory")

    if backend == "postgres":
        db_uri = os.environ.get("CHECKPOINT_DB_URI")
        if not db_uri:
            raise ValueError("CHECKPOINT_BACKEND=postgres requires CHECKPOINT_DB_URI")
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        return AsyncPostgresSaver.from_conn_string(db_uri)

    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
