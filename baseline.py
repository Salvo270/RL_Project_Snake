"""Heuristic baseline policy for the Snake environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Action:
    value: int
    delta: Tuple[int, int]


class BaselineAgent:
    """Greedy + safety-check baseline for fully observable Snake boards."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    def select_actions(self, env) -> np.ndarray:
        """Return an action for each board in the environment."""
        actions = np.full(env.n_boards, env.NONE, dtype=int)
        for idx, board in enumerate(env.boards):
            actions[idx] = self._select_action_for_board(env, board)
        return actions

    def _select_action_for_board(self, env, board: np.ndarray) -> int:
        head = self._find_single(board, env.HEAD)
        fruit = self._find_single(board, env.FRUIT)
        if head is None or fruit is None:
            return self._fallback_action(env, board, head)

        path = self._bfs_shortest_path(board, head, fruit, env)
        if path:
            next_step = path[1]
            return self._delta_to_action(env, next_step[0] - head[0], next_step[1] - head[1])

        return self._fallback_action(env, board, head)

    def _find_single(self, board: np.ndarray, value: int) -> Optional[Tuple[int, int]]:
        positions = np.argwhere(board == value)
        if positions.size == 0:
            return None
        return tuple(positions[0])

    def _bfs_shortest_path(
        self,
        board: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        env,
    ) -> Optional[Sequence[Tuple[int, int]]]:
        blocked = {env.WALL, env.BODY}
        queue = deque([start])
        parents = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                return self._reconstruct_path(parents, current)
            for nx, ny in self._neighbors(board.shape, current):
                if (nx, ny) in parents:
                    continue
                if board[nx, ny] in blocked:
                    continue
                parents[(nx, ny)] = current
                queue.append((nx, ny))
        return None

    def _reconstruct_path(
        self,
        parents: dict,
        end: Tuple[int, int],
    ) -> Sequence[Tuple[int, int]]:
        path = [end]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])
        return list(reversed(path))

    def _neighbors(
        self,
        shape: Tuple[int, int],
        position: Tuple[int, int],
    ) -> Iterable[Tuple[int, int]]:
        x, y = position
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                yield nx, ny

    def _fallback_action(self, env, board: np.ndarray, head: Optional[Tuple[int, int]]) -> int:
        if head is None:
            return env.NONE

        candidates = []
        for action in self._actions(env):
            nx, ny = head[0] + action.delta[0], head[1] + action.delta[1]
            if not (0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]):
                continue
            if board[nx, ny] in {env.WALL, env.BODY}:
                continue
            area = self._flood_fill_area(board, (nx, ny), env)
            candidates.append((action.value, area))

        if not candidates:
            return env.NONE

        best_area = max(area for _, area in candidates)
        best_actions = [action for action, area in candidates if area == best_area]
        return int(self._rng.choice(best_actions))

    def _flood_fill_area(self, board: np.ndarray, start: Tuple[int, int], env) -> int:
        open_cells = {env.EMPTY, env.FRUIT, env.HEAD}
        visited = set([start])
        queue = deque([start])
        count = 0
        while queue:
            current = queue.popleft()
            if board[current] not in open_cells:
                continue
            count += 1
            for nx, ny in self._neighbors(board.shape, current):
                if (nx, ny) in visited:
                    continue
                if board[nx, ny] in {env.WALL, env.BODY}:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return count

    def _actions(self, env) -> Sequence[Action]:
        return (
            Action(env.UP, (1, 0)),
            Action(env.RIGHT, (0, 1)),
            Action(env.DOWN, (-1, 0)),
            Action(env.LEFT, (0, -1)),
        )

    def _delta_to_action(self, env, dx: int, dy: int) -> int:
        for action in self._actions(env):
            if action.delta == (dx, dy):
                return action.value
        return env.NONE