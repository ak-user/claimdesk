"""Builds the components once, for the API and the eval script alike."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .agent import Agent
from .azure_search import build_retriever as build_search_retriever
from .claims import ClaimAssessor
from .config import Settings
from .ingest import parse_booklet
from .llm import build_clients
from .retrieval import BookletIndex, Retriever
from .review_queue import ReviewQueue
from .store import load_booklet, load_members, load_rules
from .tools import Toolbox

log = logging.getLogger(__name__)


@dataclass
class Components:
    settings: Settings
    retriever: BookletIndex
    toolbox: Toolbox
    agent: Agent
    assessor: ClaimAssessor
    queue: ReviewQueue

    @property
    def provider(self) -> str:
        return self.agent.provider


def build_components(settings: Settings) -> Components:
    try:
        chat, embedder = build_clients(settings)
    except Exception:  # noqa: BLE001 - a bad key shouldn't stop the offline demo
        log.exception("Couldn't create the model client; running offline.")
        chat, embedder = None, None
    chunks = parse_booklet(load_booklet())
    retriever: BookletIndex | None = None
    if settings.azure_search_endpoint:
        try:
            retriever = build_search_retriever(settings, chunks, embedder)
        except Exception:  # noqa: BLE001 - same as above: fall back rather than fail to start
            log.exception("Couldn't connect to Azure AI Search; using the local index.")
    if retriever is None:
        retriever = Retriever(chunks, embedder)
    toolbox = Toolbox(
        retriever,
        load_rules(),
        load_members(),
        settings.claimdesk_as_of,
        settings.retrieval_top_k,
    )
    agent = Agent(toolbox, retriever, chat, settings.agent_max_steps)
    assessor = ClaimAssessor(toolbox, retriever, chat)
    log.info(
        "ClaimDesk ready: provider=%s retrieval=%s sections=%d",
        agent.provider,
        retriever.mode,
        len(retriever.chunks),
    )
    queue = ReviewQueue(settings.claimdesk_db_path)
    return Components(settings, retriever, toolbox, agent, assessor, queue)
