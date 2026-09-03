"""
Tests for MemGPT Procedural Memory Manager.
"""

import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.memory.procedural import (
        ProceduralMemory,
        InMemoryProceduralCollection,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "memory" / "procedural.py"
    spec = importlib.util.spec_from_file_location("procedural", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ProceduralMemory = module.ProceduralMemory
    InMemoryProceduralCollection = module.InMemoryProceduralCollection


class TestProceduralMemory(unittest.TestCase):
    """
    Test suite verifying procedural memory storage, semantic recall,
    versioning, user isolation, and mock DB integration.
    """

    def setUp(self):
        # Use in-memory collection explicitly for deterministic, isolated unit tests
        self.memory = ProceduralMemory(persist_directory=":memory:")

    # -------------------------------------------------------------------------
    # 1. Store and Semantic Recall
    # -------------------------------------------------------------------------
    def test_store_and_recall_procedure(self):
        """Storing a procedure should allow semantic retrieval by related query."""
        code_snippet = (
            "def binary_search(arr, target):\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: low = mid + 1\n"
            "        else: high = mid - 1\n"
            "    return -1\n"
        )

        mem_id = self.memory.store_procedure(
            name="binary_search_algorithm",
            procedure=code_snippet,
            language="python",
            tags=["algorithm", "search"],
        )

        self.assertIsNotNone(mem_id)

        # Recall by query
        results = self.memory.recall_procedure(query="binary search algorithm in python", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("binary_search", results[0])
        self.assertIn("Procedure Name: binary_search_algorithm", results[0])

    # -------------------------------------------------------------------------
    # 2. Filtering by User ID and Language
    # -------------------------------------------------------------------------
    def test_filter_by_user_id_and_language(self):
        """Procedures should be isolated per user and language when requested."""
        self.memory.store_procedure(
            name="auth_handler",
            procedure="function authUser() { return true; }",
            user_id=101,
            language="javascript",
        )
        self.memory.store_procedure(
            name="auth_handler",
            procedure="def auth_user(): return True",
            user_id=102,
            language="python",
        )

        # User 101 query
        res_101 = self.memory.recall_procedure("auth user", user_id=101)
        self.assertEqual(len(res_101), 1)
        self.assertIn("function authUser", res_101[0])

        # User 102 query
        res_102 = self.memory.recall_procedure("auth user", user_id=102)
        self.assertEqual(len(res_102), 1)
        self.assertIn("def auth_user", res_102[0])

        # Language filter query
        res_js = self.memory.recall_procedure("auth", language="javascript")
        self.assertEqual(len(res_js), 1)
        self.assertIn("function authUser", res_js[0])

    # -------------------------------------------------------------------------
    # 3. Procedure Versioning by Name
    # -------------------------------------------------------------------------
    def test_get_procedure_versions(self):
        """Multiple stored versions of the same named procedure should be retrievable."""
        self.memory.store_procedure(
            name="deploy_workflow",
            procedure="step 1: build docker image v1",
            metadata={"version": 1},
        )
        self.memory.store_procedure(
            name="deploy_workflow",
            procedure="step 1: build docker image v2 with multi-stage caching",
            metadata={"version": 2},
        )

        versions = self.memory.get_procedure_versions("deploy_workflow")
        docs = versions.get("documents", [])
        self.assertEqual(len(docs), 2)
        self.assertTrue(any("v1" in d for d in docs))
        self.assertTrue(any("v2" in d for d in docs))

    # -------------------------------------------------------------------------
    # 4. Deletion, Update and Listing
    # -------------------------------------------------------------------------
    def test_delete_and_list_procedures(self):
        """Deleting a procedure should remove it from the collection."""
        id1 = self.memory.store_procedure("proc_1", "content 1")
        id2 = self.memory.store_procedure("proc_2", "content 2")

        all_procs = self.memory.list_all_procedures()
        self.assertEqual(len(all_procs), 2)

        deleted = self.memory.delete_procedure(id1)
        self.assertTrue(deleted)

        remaining = self.memory.list_all_procedures()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], id2)

    def test_collection_update_and_upsert(self):
        """InMemoryProceduralCollection should support update and upsert."""
        col = InMemoryProceduralCollection()
        col.add(["doc A"], [{"name": "A", "ver": 1}], ["id_A"])

        col.update(["id_A"], documents=["doc A updated"], metadatas=[{"ver": 2}])
        res = col.get(where={"name": "A"})
        self.assertEqual(res["documents"][0], "doc A updated")
        self.assertEqual(res["metadatas"][0]["ver"], 2)

        col.upsert(["doc B"], [{"name": "B"}], ["id_B"])
        self.assertEqual(len(col.get()["ids"]), 2)

    # -------------------------------------------------------------------------
    # 5. Mock DB Client Injection
    # -------------------------------------------------------------------------
    def test_mock_db_client_injection(self):
        """ProceduralMemory should accept an injected mock client."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "documents": [["Mock procedure content for testing"]]
        }

        memory = ProceduralMemory(client=mock_client)
        results = memory.recall_procedure("test query")

        mock_collection.query.assert_called_once()
        self.assertEqual(results, ["Mock procedure content for testing"])


if __name__ == "__main__":
    unittest.main()
