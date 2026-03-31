from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from ordered_set import OrderedSet


@dataclass
class DirectCompilerDiscovery:
    player_count: int
    children: list[dict[tuple, OrderedSet]]
    actions: list[dict[Any, OrderedSet]]
    infoset_order: list[list[Any]]
    infoset_labels: list[dict[Any, str]]
    utilities_hashed: defaultdict[tuple, np.ndarray]
    stats: dict[str, Any]


def direct_compiler_eligibility_reason(game: Any) -> str | None:
    try:
        short_name = str(game.get_type().short_name)
    except Exception:
        return "game type is unavailable"
    if short_name != "pbs":
        return f"unsupported game short_name={short_name!r}"

    try:
        player_count = int(game.num_players())
    except Exception:
        return "game.num_players() is unavailable"
    if player_count != 2:
        return f"requires exactly 2 players; found {player_count}"

    required_attrs = (
        "Nv",
        "Nb",
        "num_stages",
        "latency",
        "k_t",
        "q_t",
        "value_levels",
        "bid_levels",
        "max_bid_index_by_value",
        "valuation_dist",
    )
    for attr in required_attrs:
        if not hasattr(game, attr):
            return f"game is missing attribute {attr!r}"

    try:
        latency = tuple(int(x) for x in getattr(game, "latency"))
    except Exception:
        return "latency is not a valid integer sequence"
    if len(latency) != 2:
        return f"requires latency length 2; found {len(latency)}"
    if latency[0] != latency[1]:
        return f"requires equal latency; found {latency}"

    return None


def can_direct_compile_pbs_passive(game: Any) -> bool:
    return direct_compiler_eligibility_reason(game) is None


def discover_pbs_passive_direct(
    game: Any,
    *,
    emit_policy_specs: bool,
    state_key_from_infoset: Callable[[int, str], Any],
) -> DirectCompilerDiscovery:
    reason = direct_compiler_eligibility_reason(game)
    if reason is not None:
        raise ValueError(reason)

    player_count = int(game.num_players())
    nv = int(game.Nv)
    num_stages = int(game.num_stages)
    latency = tuple(int(x) for x in game.latency)
    value_levels = tuple(float(x) for x in game.value_levels)
    bid_levels = tuple(float(x) for x in game.bid_levels)
    max_bid_index_by_value = tuple(int(x) for x in game.max_bid_index_by_value)
    k_t = tuple(float(x) for x in game.k_t)
    q_t = tuple(float(x) for x in game.q_t)
    raw_dist = getattr(game, "valuation_dist", None)

    if raw_dist is None:
        valuation_dist = tuple(
            1.0 / float(nv ** player_count)
            for _ in range(nv ** player_count)
        )
    else:
        valuation_dist = tuple(float(x) for x in raw_dist)

    children: list[dict[tuple, OrderedSet]] = [{(): OrderedSet()} for _ in range(player_count)]
    actions: list[dict[Any, OrderedSet]] = [{} for _ in range(player_count)]
    infoset_order: list[list[Any]] = [[] for _ in range(player_count)]
    infoset_labels: list[dict[Any, str]] = [{} for _ in range(player_count)] if emit_policy_specs else []

    def _observed_bids(player: int, stage: int, bids0: tuple[int, ...], bids1: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        available = max(0, int(stage) - int(latency[player]))
        if player == 0:
            return tuple(bids0), tuple(bids1[:available])
        return tuple(bids0[:available]), tuple(bids1)

    def _infoset_string(player: int, value_idx: int, stage: int, bids0: tuple[int, ...], bids1: tuple[int, ...]) -> str:
        obs0, obs1 = _observed_bids(player, stage, bids0, bids1)
        obs_enc = ";".join(
            (
                f"0:{','.join(map(str, obs0))}",
                f"1:{','.join(map(str, obs1))}",
            ),
        )
        return f"B{player}|v:{int(value_idx)}|stage:{int(stage)}|obs:[{obs_enc}]"

    def _register_infoset(player: int, parent_sequence: tuple, infoset_str: str) -> Any:
        infoset_key = state_key_from_infoset(player, infoset_str)
        children[player].setdefault(parent_sequence, OrderedSet()).add(infoset_key)
        if infoset_key not in actions[player]:
            actions[player][infoset_key] = OrderedSet()
            infoset_order[player].append(infoset_key)
            if emit_policy_specs:
                infoset_labels[player][infoset_key] = infoset_str
        return infoset_key

    def _legal_actions(value_idx: int, own_bids: tuple[int, ...]) -> range:
        prev_bid = int(own_bids[-1]) if own_bids else 0
        max_bid = int(max_bid_index_by_value[int(value_idx)])
        if prev_bid > max_bid:
            raise ValueError("Previous bid exceeds valuation cap")
        return range(prev_bid, max_bid + 1)

    def _terminal_payoffs(
        valuation_prob: float,
        v0: int,
        v1: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
    ) -> np.ndarray:
        acc = np.zeros(player_count, dtype=np.float32)
        values = (value_levels[int(v0)], value_levels[int(v1)])
        for t_idx in range(num_stages):
            stage_bids = (int(bids0[t_idx]), int(bids1[t_idx]))
            stage_bid_vals = (bid_levels[stage_bids[0]], bid_levels[stage_bids[1]])
            max_bid = max(stage_bids)
            winners = [idx for idx, bid in enumerate(stage_bids) if bid == max_bid]
            scale = float(valuation_prob) * float(q_t[t_idx]) * float(k_t[t_idx])
            if not winners:
                continue
            if len(winners) == 1:
                w = winners[0]
                acc[w] += scale * float(values[w] - stage_bid_vals[w])
            else:
                split = float(len(winners))
                for w in winners:
                    acc[w] += scale * float(values[w] - stage_bid_vals[w]) / split
        return acc

    valuation_outcomes = nv ** player_count
    terminal_histories = 0
    expanded_frames = 0

    for valuation_action in range(valuation_outcomes):
        valuation_prob = float(valuation_dist[valuation_action])
        remaining = int(valuation_action)
        v1 = remaining % nv
        remaining //= nv
        v0 = remaining % nv

        stack: list[tuple[int, int, tuple[int, ...], tuple[int, ...], tuple, tuple]] = [
            (1, 0, (), (), (), ()),
        ]
        while stack:
            stage, player, bids0, bids1, seq0, seq1 = stack.pop()
            expanded_frames += 1

            if player == 0:
                infoset0_str = _infoset_string(0, v0, stage, bids0, bids1)
                infoset0 = _register_infoset(0, seq0, infoset0_str)
                frames: list[tuple[int, int, tuple[int, ...], tuple[int, ...], tuple, tuple]] = []
                for action0 in _legal_actions(v0, bids0):
                    action0 = int(action0)
                    actions[0][infoset0].add(action0)
                    children[0].setdefault((infoset0, action0), OrderedSet())
                    frames.append((stage, 1, bids0 + (action0,), bids1, (infoset0, action0), seq1))
                stack.extend(reversed(frames))
                continue

            infoset1_str = _infoset_string(1, v1, stage, bids0, bids1)
            infoset1 = _register_infoset(1, seq1, infoset1_str)
            frames = []
            for action1 in _legal_actions(v1, bids1):
                action1 = int(action1)
                actions[1][infoset1].add(action1)
                children[1].setdefault((infoset1, action1), OrderedSet())
                new_seq1 = (infoset1, action1)
                new_bids1 = bids1 + (action1,)

                if stage == num_stages:
                    terminal_histories += 1
                else:
                    frames.append((stage + 1, 0, bids0, new_bids1, seq0, new_seq1))
            stack.extend(reversed(frames))

    stats = {
        "valuation_outcomes": int(valuation_outcomes),
        "expanded_frames": int(expanded_frames),
        "terminal_histories": int(terminal_histories),
        "utility_profiles": 0,
    }

    return DirectCompilerDiscovery(
        player_count=player_count,
        children=children,
        actions=actions,
        infoset_order=infoset_order,
        infoset_labels=infoset_labels,
        utilities_hashed=defaultdict(lambda: np.zeros(player_count, dtype=np.float32)),
        stats=stats,
    )


def emit_pbs_passive_direct_rows(
    game: Any,
    *,
    state_key_from_infoset: Callable[[int, str], Any],
    external_sequence_index: list[dict[tuple, int]],
    emit_row: Callable[[list[int], np.ndarray], None],
) -> None:
    reason = direct_compiler_eligibility_reason(game)
    if reason is not None:
        raise ValueError(reason)

    player_count = int(game.num_players())
    nv = int(game.Nv)
    num_stages = int(game.num_stages)
    latency = tuple(int(x) for x in game.latency)
    value_levels = tuple(float(x) for x in game.value_levels)
    bid_levels = tuple(float(x) for x in game.bid_levels)
    max_bid_index_by_value = tuple(int(x) for x in game.max_bid_index_by_value)
    k_t = tuple(float(x) for x in game.k_t)
    q_t = tuple(float(x) for x in game.q_t)
    raw_dist = getattr(game, "valuation_dist", None)

    if raw_dist is None:
        valuation_dist = tuple(
            1.0 / float(nv ** player_count)
            for _ in range(nv ** player_count)
        )
    else:
        valuation_dist = tuple(float(x) for x in raw_dist)

    def _observed_bids(player: int, stage: int, bids0: tuple[int, ...], bids1: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        available = max(0, int(stage) - int(latency[player]))
        if player == 0:
            return tuple(bids0), tuple(bids1[:available])
        return tuple(bids0[:available]), tuple(bids1)

    def _infoset_string(player: int, value_idx: int, stage: int, bids0: tuple[int, ...], bids1: tuple[int, ...]) -> str:
        obs0, obs1 = _observed_bids(player, stage, bids0, bids1)
        obs_enc = ";".join(
            (
                f"0:{','.join(map(str, obs0))}",
                f"1:{','.join(map(str, obs1))}",
            ),
        )
        return f"B{player}|v:{int(value_idx)}|stage:{int(stage)}|obs:[{obs_enc}]"

    def _legal_actions(value_idx: int, own_bids: tuple[int, ...]) -> range:
        prev_bid = int(own_bids[-1]) if own_bids else 0
        max_bid = int(max_bid_index_by_value[int(value_idx)])
        if prev_bid > max_bid:
            raise ValueError("Previous bid exceeds valuation cap")
        return range(prev_bid, max_bid + 1)

    def _terminal_payoffs(
        valuation_prob: float,
        v0: int,
        v1: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
    ) -> np.ndarray:
        acc = np.zeros(player_count, dtype=np.float32)
        values = (value_levels[int(v0)], value_levels[int(v1)])
        for t_idx in range(num_stages):
            stage_bids = (int(bids0[t_idx]), int(bids1[t_idx]))
            stage_bid_vals = (bid_levels[stage_bids[0]], bid_levels[stage_bids[1]])
            max_bid = max(stage_bids)
            winners = [idx for idx, bid in enumerate(stage_bids) if bid == max_bid]
            scale = float(valuation_prob) * float(q_t[t_idx]) * float(k_t[t_idx])
            if not winners:
                continue
            if len(winners) == 1:
                winner = winners[0]
                acc[winner] += scale * float(values[winner] - stage_bid_vals[winner])
            else:
                split = float(len(winners))
                for winner in winners:
                    acc[winner] += scale * float(values[winner] - stage_bid_vals[winner]) / split
        return acc

    valuation_outcomes = nv ** player_count
    for valuation_action in range(valuation_outcomes):
        valuation_prob = float(valuation_dist[valuation_action])
        remaining = int(valuation_action)
        v1 = remaining % nv
        remaining //= nv
        v0 = remaining % nv

        stack: list[tuple[int, int, tuple[int, ...], tuple[int, ...], tuple, tuple]] = [
            (1, 0, (), (), (), ()),
        ]
        while stack:
            stage, player, bids0, bids1, seq0, seq1 = stack.pop()

            if player == 0:
                infoset0 = state_key_from_infoset(0, _infoset_string(0, v0, stage, bids0, bids1))
                frames: list[tuple[int, int, tuple[int, ...], tuple[int, ...], tuple, tuple]] = []
                for action0 in _legal_actions(v0, bids0):
                    action0 = int(action0)
                    frames.append((stage, 1, bids0 + (action0,), bids1, (infoset0, action0), seq1))
                stack.extend(reversed(frames))
                continue

            infoset1 = state_key_from_infoset(1, _infoset_string(1, v1, stage, bids0, bids1))
            frames = []
            for action1 in _legal_actions(v1, bids1):
                action1 = int(action1)
                new_seq1 = (infoset1, action1)
                new_bids1 = bids1 + (action1,)

                if stage == num_stages:
                    coords_row = [
                        int(external_sequence_index[0][seq0]),
                        int(external_sequence_index[1][new_seq1]),
                    ]
                    emit_row(
                        coords_row,
                        _terminal_payoffs(valuation_prob, v0, v1, bids0, new_bids1),
                    )
                else:
                    frames.append((stage + 1, 0, bids0, new_bids1, seq0, new_seq1))
            stack.extend(reversed(frames))
