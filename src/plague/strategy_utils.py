import logging
import random
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from .strategy_generator.base import AdversarialStrategy

logger = logging.getLogger("NewMultiTurnAttackStratUtils")


def merge_strategy(
    strategy_library, strategy: AdversarialStrategy, MERGE_THRESHOLD=0.85
):
    strategies_embeddings = [
        i.strategy_embedding for i in list(strategy_library.values())
    ]
    similarities = calculate_similarity(
        np.array(strategy.strategy_embedding), np.array(strategies_embeddings)
    )
    strategies = strategy_library.values()
    combined_stats = list(zip(strategies, similarities))
    combined_stats.sort(key=lambda x: x[1], reverse=True)
    combined_stats = [strat for strat, sim in combined_stats if sim > MERGE_THRESHOLD]
    if combined_stats:
        best_match = combined_stats[0]
        strategy_library[best_match.category].examples.append(strategy.examples)
        return True
    else:
        return False


def calculate_similarity(embedding1: NDArray, embedding2: NDArray) -> NDArray:
    if embedding1.size == 0 or embedding2.size == 0:
        return np.zeros((embedding2.shape[0], 1))

    # Ensure correct shapes
    embedding1 = embedding1.reshape(1, -1)  # (1, D)
    embedding2 = embedding2.reshape(-1, embedding1.shape[1])  # (N, D)

    # Compute norms
    norm1 = np.linalg.norm(embedding1, axis=1, keepdims=True)  # (1, 1)
    norm2 = np.linalg.norm(embedding2, axis=1, keepdims=True)  # (N, 1)

    # Avoid division by zero
    if norm1[0, 0] == 0 or np.any(norm2 == 0):
        return np.zeros((embedding2.shape[0], 1))

    # Compute dot products and cosine similarities
    dot_products = embedding2 @ embedding1.T  # (N, 1)
    similarities = dot_products / (norm2 * norm1)  # (N, 1)

    return similarities


def select_random_strategies(
    strategy_library: Dict[str, AdversarialStrategy], max_results: int = 2
):
    return random.sample(list(strategy_library.values()), max_results)


def find_similar_strategies(
    target_embedding: List[float],
    strategy_library: Dict[str, AdversarialStrategy],
    threshold: float = 0.6,
    max_results: int = 2,
) -> List[AdversarialStrategy]:
    if not target_embedding or len(strategy_library) == 0:
        return []

    strategies_with_embeddings = []
    for strategy_name, strategy in strategy_library.items():
        if strategy.strategy_embedding and strategy.retrieval_embedding:
            strategies_with_embeddings.append((strategy_name, strategy))

    if len(strategies_with_embeddings) == 0:
        logger.info("No strategies with embeddings found")
        return []

    similarities = []
    question_embeddings = [
        strat.retrieval_embedding for _, strat in strategies_with_embeddings
    ]
    similarities = calculate_similarity(
        np.array(target_embedding), np.array(question_embeddings)
    )
    logger.debug(f"The similarities are {similarities}")

    combined_stats = [
        (strategy_name, strategy, sim)
        for (strategy_name, strategy), sim in zip(
            strategies_with_embeddings, similarities
        )
    ]
    combined_stats.sort(key=lambda x: x[2], reverse=True)
    for strategy_name, strategy, sim in combined_stats[:max_results]:
        logger.info(strategy.goals)
        logger.info(f"similarity is {sim}")

    req_combined_stats = [
        strategy
        for _, strategy, similarity in combined_stats
        if similarity >= threshold
    ][:max_results]

    return req_combined_stats
