import json
import logging
from typing import Any, Dict, List, Optional
import os
import time
import aiofiles
import asyncio

class LettaVirtualContextSyncV3:
    """
    Advanced synchronization hook to push procedural memory from virtual context
    streams to disk in real-time, inspired by MemGPT/Letta patterns.
    """

    def __init__(self, sync_path: str = "./procedural_sync.jsonl") -> None:
        """
        Initialize the sync hook.

        Args:
            sync_path (str): The file path where procedural memory will be appended.
        """
        self.sync_path = sync_path

    async def bootstrap(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        if "sync_path" in config:
            self.sync_path = config["sync_path"]
        logging.info(f"LettaVirtualContextSyncV3 bootstrapped with sync_path={self.sync_path}")

    async def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        """Process incoming content before it is stored or used."""
        return content

    async def assemble(self, context_items: List[Any], metadata: Dict[str, Any]) -> str:
        """Assemble the context string from retrieved items for the LLM."""
        return "\n".join([str(item) for item in context_items])

    async def compact(self, context_items: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Compact or summarize the context when limits are reached."""
        return context_items

    def before_retrieval(self, query: str, user_id: int) -> str:
        """Called before context is retrieved."""
        return query

    def after_retrieval(self, context: List[Any], query: str, user_id: int) -> List[Any]:
        """Called after context is retrieved."""
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        """Called before context is written."""
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        """Called after context is written."""
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """
        Called when the overall context is updated.
        Extracts procedural context from the update and writes it to disk.
        Since we might be in a sync context, we use asyncio.create_task if a loop is running.
        """
        if not isinstance(new_context, dict):
            return

        procedural_data = new_context.get("procedural_memory")
        if procedural_data:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._sync_to_disk(procedural_data, user_id))
            except RuntimeError:
                # No running event loop, execute synchronously (e.g. tests)
                asyncio.run(self._sync_to_disk(procedural_data, user_id))

    async def _sync_to_disk(self, procedural_data: Any, user_id: int) -> None:
        """
        Internal async method to perform the disk sync.
        """
        entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "data": procedural_data
        }
        try:
            async with aiofiles.open(self.sync_path, mode="a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logging.debug(f"Synced procedural memory to {self.sync_path}")
        except Exception as e:
            logging.error(f"Failed to sync procedural memory: {e}")

    async def pre_process(self, content: str, metadata: Dict[str, Any]) -> str:
        """Called to pre-process content before ingestion."""
        return content

    async def post_process(self, response: str, metadata: Dict[str, Any]) -> str:
        """Called to post-process a response before returning to user."""
        return response
