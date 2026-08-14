"""``memory_units.updated_at`` must move whenever a memory actually changes (#3490).

Consumers chase the column (``WHERE updated_at > watermark``) for incremental
export, cache invalidation and the mental-model staleness check, so a write path
that changes a memory without stamping it makes the chase silently skip that
change. Several did: the document tag propagation, ``set_memory_embedding``,
``set_invalidation_reason`` and the transfer importer's fixups.

The one deliberate exception is consolidation bookkeeping (``consolidated_at`` /
``consolidation_failed_at``): stamping it would make every consolidation pass
look like an edit to every fact it folded, re-flagging mental models stale and
re-feeding unchanged facts to a delta refresh. Both halves are pinned here —
the paths that must stamp, and the ones that must not.

The assertions read ``updated_at`` with raw SQL because it is not part of
``StoredMemory``; these are the SQL store's statements, which is where the fix
lives.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table

# A backdated baseline every case starts from, so "did this statement stamp the
# column" is a comparison against a value no ``now()`` can collide with.
_BASELINE = datetime(2020, 1, 1, tzinfo=timezone.utc)


async def _seed_memory(
    memory: MemoryEngine, conn, bank_id: str, text: str, *, document_id: str | None = None
) -> uuid.UUID:
    """Insert one fact through the store, bypassing the LLM retain pipeline."""
    store = get_memories()
    fact = SimpleNamespace(
        fact_text=text,
        embedding=memory.embeddings.encode([text])[0],
        fact_type="experience",
        tags=[],
        context=None,
        document_id=document_id,
        chunk_id=None,
        metadata=None,
        observation_scopes=None,
        entities=[],
        causal_relations=[],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )
    unit_ids = await store.insert_facts(
        conn=conn, ops=memory._backend.ops, bank_id=bank_id, facts=[fact], document_id=document_id
    )
    return uuid.UUID(unit_ids[0])


async def _backdate(conn, memory_id: uuid.UUID, table: str = "memory_units") -> None:
    """Park ``updated_at`` in the past so a later stamp is unambiguous."""
    await conn.execute(f"UPDATE {table} SET updated_at = $1 WHERE id = $2", _BASELINE, memory_id)


async def _updated_at(conn, memory_id: uuid.UUID, table: str = "memory_units") -> datetime:
    return await conn.fetchval(f"SELECT updated_at FROM {table} WHERE id = $1", memory_id)


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


class TestWritesThatMustStamp:
    @pytest.mark.asyncio
    async def test_document_tag_propagation_stamps_updated_at(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Retagging a document changes its memories' tags — and their updated_at."""
        bank_id = f"test-mu-updated-tags-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO documents (id, bank_id, original_text, content_hash, created_at, updated_at) "
                "VALUES ($1, $2, 'some doc', 'hash123', NOW(), NOW())",
                doc_id,
                bank_id,
            )
            mem_id = await _seed_memory(memory, conn, bank_id, "Alice loves hiking.", document_id=doc_id)
            await _backdate(conn, mem_id)

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            assert await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            assert await _updated_at(conn, mem_id) > _BASELINE

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_embedding_write_stamps_updated_at(self, memory: MemoryEngine, request_context: RequestContext):
        """A re-embed (curation edit, revert) rewrites the stored vector — that is a change."""
        bank_id = f"test-mu-updated-embed-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            mem_id = await _seed_memory(memory, conn, bank_id, "Alice loves hiking.")
            await _backdate(conn, mem_id)

            embedding = str(list(map(float, memory.embeddings.encode(["Alice loves climbing."])[0])))
            await get_memories().set_memory_embedding(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=str(mem_id), embedding=embedding
            )

            assert await _updated_at(conn, mem_id) > _BASELINE

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_invalidation_reason_change_stamps_updated_at(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Editing an archived memory's reason is an edit to the archive row."""
        bank_id = f"test-mu-updated-reason-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            store = get_memories()
            mem_id = await _seed_memory(memory, conn, bank_id, "Alice loves hiking.")
            await _backdate(conn, mem_id)
            # The archive row is copied from the live one, so it inherits the baseline.
            await store.invalidate_memory(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=str(mem_id), reason="wrong"
            )
            assert await _updated_at(conn, mem_id, "invalidated_memory_units") == _BASELINE

            await store.set_invalidation_reason(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=str(mem_id), reason="outdated"
            )

            assert await _updated_at(conn, mem_id, "invalidated_memory_units") > _BASELINE

        await memory.delete_bank(bank_id, request_context=request_context)


class TestConsolidationBookkeepingIsExempt:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "when,failed",
        [
            (datetime.now(timezone.utc), False),  # folded into an observation
            (None, False),  # requeued after that observation was invalidated
            (datetime.now(timezone.utc), True),  # the LLM could not consolidate it
        ],
        ids=["consolidated", "requeued", "failed"],
    )
    async def test_mark_consolidated_leaves_updated_at_alone(
        self, memory: MemoryEngine, request_context: RequestContext, when: datetime | None, failed: bool
    ):
        """Scheduler state is not an edit: stamping it would re-flag every mental model stale."""
        bank_id = f"test-mu-updated-mark-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            mem_id = await _seed_memory(memory, conn, bank_id, "Alice loves hiking.")
            await _backdate(conn, mem_id)

            await get_memories().mark_consolidated(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(mem_id)], when=when, failed=failed
            )

            assert await _updated_at(conn, mem_id) == _BASELINE

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_requeue_after_observation_cleanup_leaves_updated_at_alone(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The engine's own inline requeue (clear_observations_for_memory) is bookkeeping too."""
        bank_id = f"test-mu-updated-requeue-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            mem_id = await _seed_memory(memory, conn, bank_id, "Alice loves hiking.")
            await get_memories().mark_consolidated(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                unit_ids=[str(mem_id)],
                when=datetime.now(timezone.utc),
            )
            await conn.execute(
                "INSERT INTO memory_units "
                "(id, bank_id, text, fact_type, event_date, source_memory_ids, proof_count, created_at, updated_at) "
                "VALUES ($1, $2, 'Alice is a hiker.', 'observation', NOW(), $3, 1, NOW(), NOW())",
                uuid.uuid4(),
                bank_id,
                [mem_id],
            )
            await _backdate(conn, mem_id)

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            result = await memory.clear_observations_for_memory(bank_id, str(mem_id), request_context=request_context)
        assert result["deleted_count"] == 1

        async with pool.acquire() as conn:
            # The requeue happened (consolidated_at cleared) but the memory did not change.
            assert await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", mem_id) is None
            assert await _updated_at(conn, mem_id) == _BASELINE

        await memory.delete_bank(bank_id, request_context=request_context)
