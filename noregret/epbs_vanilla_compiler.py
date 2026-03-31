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
    if short_name != "epbs_vanilla":
        return f"unsupported game short_name={short_name!r}"

    try:
        player_count = int(game.num_players())
    except Exception:
        return "game.num_players() is unavailable"
    if player_count != 3:
        return f"requires exactly 3 players; found {player_count}"

    required_attrs = (
        "num_builders",
        "Nv",
        "Nb",
        "num_stages",
        "latency",
        "k_t",
        "value_levels",
        "bid_levels",
        "max_bid_index_by_value",
        "valuation_dist",
        "message_mask_bound",
    )
    for attr in required_attrs:
        if not hasattr(game, attr):
            return f"game is missing attribute {attr!r}"

    try:
        num_builders = int(game.num_builders)
    except Exception:
        return "num_builders is unavailable"
    if num_builders != 2:
        return f"requires exactly 2 builders; found {num_builders}"

    try:
        latency = tuple(int(x) for x in getattr(game, "latency"))
    except Exception:
        return "latency is not a valid integer sequence"
    if len(latency) != num_builders:
        return f"requires latency length {num_builders}; found {len(latency)}"
    if any(lat < 1 for lat in latency):
        return f"requires latency >= 1; found {latency}"

    try:
        num_stages = int(game.num_stages)
    except Exception:
        return "num_stages is unavailable"
    if num_stages < 1:
        return f"requires num_stages >= 1; found {num_stages}"

    return None


def can_direct_compile_epbs_vanilla(game: Any) -> bool:
    return direct_compiler_eligibility_reason(game) is None


def discover_epbs_vanilla_direct(
    game: Any,
    *,
    emit_policy_specs: bool,
    state_key_from_infoset: Callable[[int, str], Any],
) -> DirectCompilerDiscovery:
    reason = direct_compiler_eligibility_reason(game)
    if reason is not None:
        raise ValueError(reason)

    player_count = int(game.num_players())
    num_builders = int(game.num_builders)
    nv = int(game.Nv)
    num_stages = int(game.num_stages)
    latency = tuple(int(x) for x in game.latency)
    value_levels = tuple(float(x) for x in game.value_levels)
    bid_levels = tuple(float(x) for x in game.bid_levels)
    max_bid_index_by_value = tuple(int(x) for x in game.max_bid_index_by_value)
    k_t = tuple(float(x) for x in game.k_t)
    message_mask_bound = int(game.message_mask_bound)
    raw_dist = getattr(game, "valuation_dist", None)

    if raw_dist is None:
        valuation_dist = tuple(
            1.0 / float(nv ** num_builders)
            for _ in range(nv ** num_builders)
        )
    else:
        valuation_dist = tuple(float(x) for x in raw_dist)

    children: list[dict[tuple, OrderedSet]] = [{(): OrderedSet()} for _ in range(player_count)]
    actions: list[dict[Any, OrderedSet]] = [{} for _ in range(player_count)]
    infoset_order: list[list[Any]] = [[] for _ in range(player_count)]
    infoset_labels: list[dict[Any, str]] = [{} for _ in range(player_count)] if emit_policy_specs else []

    def _submasks(mask: int):
        submask = int(mask)
        while True:
            yield submask
            if submask == 0:
                break
            submask = (submask - 1) & int(mask)

    def _encode_message_pair(mask1: int, mask2: int) -> int:
        return 1 + int(mask1) * message_mask_bound + int(mask2)

    def _delivered_mask(receiver_idx: int, stage: int, message_masks: tuple[int, ...]) -> int:
        delivered_count = max(0, int(stage) - int(latency[receiver_idx]))
        delivered_count = min(delivered_count, len(message_masks))
        mask = 0
        for sent_mask in message_masks[:delivered_count]:
            mask |= int(sent_mask)
        return mask

    def _known_opponent_bids(
        receiver_idx: int,
        stage: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
        msg0: tuple[int, ...],
        msg1: tuple[int, ...],
    ) -> list[int]:
        opponent_idx = 1 - int(receiver_idx)
        delivered_mask = _delivered_mask(receiver_idx, stage, msg0 if receiver_idx == 0 else msg1)
        opponent_bids = bids1 if opponent_idx == 1 else bids0
        known: list[int] = []
        for stage_no in range(1, num_stages + 1):
            if delivered_mask & (1 << (stage_no - 1)):
                if len(opponent_bids) >= stage_no:
                    known.append(int(opponent_bids[stage_no - 1]))
                else:
                    known.append(-1)
            else:
                known.append(-1)
        return known

    def _proposer_infoset_string(
        stage: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
        msg0: tuple[int, ...],
        msg1: tuple[int, ...],
    ) -> str:
        bids_enc = ";".join(
            (
                f"0:{','.join(map(str, bids0))}",
                f"1:{','.join(map(str, bids1))}",
            ),
        )
        msg_enc = ";".join(
            (
                f"0:{','.join(map(str, msg0))}",
                f"1:{','.join(map(str, msg1))}",
            ),
        )
        return f"P|stage:{int(stage)}|phase:PROPOSER|bids:[{bids_enc}]|m:[{msg_enc}]"

    def _builder_infoset_string(
        builder_idx: int,
        value_idx: int,
        stage: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
        msg0: tuple[int, ...],
        msg1: tuple[int, ...],
    ) -> str:
        own_bids = bids0 if builder_idx == 0 else bids1
        known = _known_opponent_bids(builder_idx, stage, bids0, bids1, msg0, msg1)
        known_enc = ",".join("?" if bid < 0 else str(bid) for bid in known)
        delivered_mask = _delivered_mask(builder_idx, stage, msg0 if builder_idx == 0 else msg1)
        return (
            f"B{int(builder_idx)}|v:{int(value_idx)}|stage:{int(stage)}"
            f"|b:[{','.join(map(str, own_bids))}]|opp:[{known_enc}]|d:{int(delivered_mask)}"
        )

    def _register_infoset(player: int, parent_sequence: tuple, infoset_str: str) -> Any:
        infoset_key = state_key_from_infoset(player, infoset_str)
        children[player].setdefault(parent_sequence, OrderedSet()).add(infoset_key)
        if infoset_key not in actions[player]:
            actions[player][infoset_key] = OrderedSet()
            infoset_order[player].append(infoset_key)
            if emit_policy_specs:
                infoset_labels[player][infoset_key] = infoset_str
        return infoset_key

    def _legal_builder_actions(value_idx: int, own_bids: tuple[int, ...]) -> range:
        prev_bid = int(own_bids[-1]) if own_bids else 0
        max_bid = int(max_bid_index_by_value[int(value_idx)])
        if prev_bid > max_bid:
            raise ValueError("Previous bid exceeds valuation cap")
        return range(prev_bid, max_bid + 1)

    def _receiver_has_time_to_use_message(receiver_idx: int, stage: int) -> bool:
        return int(stage) + int(latency[receiver_idx]) <= num_stages

    def _scheduled_mask(message_masks: tuple[int, ...]) -> int:
        mask = 0
        for sent_mask in message_masks:
            mask |= int(sent_mask)
        return mask

    def _available_reveal_mask(receiver_idx: int, stage: int, message_masks: tuple[int, ...]) -> int:
        if not _receiver_has_time_to_use_message(receiver_idx, stage):
            return 0
        upto_stage_mask = (1 << int(stage)) - 1
        return upto_stage_mask & ~_scheduled_mask(message_masks)

    def _terminal_payoffs(
        valuation_prob: float,
        terminal_stage: int,
        v0: int,
        v1: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
    ) -> np.ndarray:
        acc = np.zeros(player_count, dtype=np.float32)
        stage_idx = int(terminal_stage) - 1
        stage_bids = (int(bids0[stage_idx]), int(bids1[stage_idx]))
        stage_bid_vals = (bid_levels[stage_bids[0]], bid_levels[stage_bids[1]])
        val_vals = (value_levels[int(v0)], value_levels[int(v1)])

        max_bid = max(stage_bids)
        winners = [idx for idx, bid in enumerate(stage_bids) if bid == max_bid]
        scale = float(valuation_prob) * float(k_t[stage_idx])

        acc[0] += scale * float(max(stage_bid_vals))
        if len(winners) == 1:
            winner = winners[0]
            acc[1 + winner] += scale * float(val_vals[winner] - stage_bid_vals[winner])
        else:
            split = float(len(winners))
            for winner in winners:
                acc[1 + winner] += scale * float(val_vals[winner] - stage_bid_vals[winner]) / split
        return acc

    valuation_outcomes = nv ** num_builders
    terminal_histories = 0
    expanded_frames = 0

    for valuation_action in range(valuation_outcomes):
        valuation_prob = float(valuation_dist[valuation_action])
        remaining = int(valuation_action)
        v1 = remaining % nv
        remaining //= nv
        v0 = remaining % nv

        stack: list[tuple[str, int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple, tuple, tuple]] = [
            ("B0", 1, (), (), (), (), (), (), ()),
        ]
        while stack:
            phase, stage, bids0, bids1, msg0, msg1, seq_p, seq_b0, seq_b1 = stack.pop()
            expanded_frames += 1

            if phase == "B0":
                infoset_str = _builder_infoset_string(0, v0, stage, bids0, bids1, msg0, msg1)
                infoset = _register_infoset(1, seq_b0, infoset_str)
                frames: list[tuple[str, int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple, tuple, tuple]] = []
                for action0 in _legal_builder_actions(v0, bids0):
                    action0 = int(action0)
                    actions[1][infoset].add(action0)
                    children[1].setdefault((infoset, action0), OrderedSet())
                    frames.append((
                        "B1",
                        stage,
                        bids0 + (action0,),
                        bids1,
                        msg0,
                        msg1,
                        seq_p,
                        (infoset, action0),
                        seq_b1,
                    ))
                stack.extend(reversed(frames))
                continue

            if phase == "B1":
                infoset_str = _builder_infoset_string(1, v1, stage, bids0, bids1, msg0, msg1)
                infoset = _register_infoset(2, seq_b1, infoset_str)
                frames = []
                for action1 in _legal_builder_actions(v1, bids1):
                    action1 = int(action1)
                    actions[2][infoset].add(action1)
                    children[2].setdefault((infoset, action1), OrderedSet())
                    new_seq_b1 = (infoset, action1)
                    new_bids1 = bids1 + (action1,)

                    if stage == num_stages:
                        terminal_histories += 1
                    else:
                        frames.append((
                            "P",
                            stage,
                            bids0,
                            new_bids1,
                            msg0,
                            msg1,
                            seq_p,
                            seq_b0,
                            new_seq_b1,
                        ))
                stack.extend(reversed(frames))
                continue

            if phase != "P":
                raise ValueError(f"unsupported phase {phase!r}")

            infoset_str = _proposer_infoset_string(stage, bids0, bids1, msg0, msg1)
            infoset = _register_infoset(0, seq_p, infoset_str)
            allowed_b0 = _available_reveal_mask(0, stage, msg0)
            allowed_b1 = _available_reveal_mask(1, stage, msg1)

            frames = []
            actions[0][infoset].add(0)
            children[0].setdefault((infoset, 0), OrderedSet())
            terminal_histories += 1

            for mask0 in _submasks(allowed_b0):
                for mask1 in _submasks(allowed_b1):
                    action = _encode_message_pair(mask0, mask1)
                    actions[0][infoset].add(action)
                    children[0].setdefault((infoset, action), OrderedSet())
                    frames.append((
                        "B0",
                        stage + 1,
                        bids0,
                        bids1,
                        msg0 + (int(mask0),),
                        msg1 + (int(mask1),),
                        (infoset, action),
                        seq_b0,
                        seq_b1,
                    ))
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


def emit_epbs_vanilla_direct_rows(
    game: Any,
    *,
    state_key_from_infoset: Callable[[int, str], Any],
    external_sequence_index: list[dict[tuple, int]],
    emit_row: Callable[[list[int], np.ndarray], None],
    debug: bool = False,
) -> None:
    reason = direct_compiler_eligibility_reason(game)
    if reason is not None:
        raise ValueError(reason)

    player_count = int(game.num_players())
    num_builders = int(game.num_builders)
    nv = int(game.Nv)
    num_stages = int(game.num_stages)
    latency = tuple(int(x) for x in game.latency)
    value_levels = tuple(float(x) for x in game.value_levels)
    bid_levels = tuple(float(x) for x in game.bid_levels)
    max_bid_index_by_value = tuple(int(x) for x in game.max_bid_index_by_value)
    k_t = tuple(float(x) for x in game.k_t)
    message_mask_bound = int(game.message_mask_bound)
    raw_dist = getattr(game, "valuation_dist", None)

    if raw_dist is None:
        valuation_dist = tuple(
            1.0 / float(nv ** num_builders)
            for _ in range(nv ** num_builders)
        )
    else:
        valuation_dist = tuple(float(x) for x in raw_dist)

    def _submasks(mask: int):
        submask = int(mask)
        while True:
            yield submask
            if submask == 0:
                break
            submask = (submask - 1) & int(mask)

    def _encode_message_pair(mask1: int, mask2: int) -> int:
        return 1 + int(mask1) * message_mask_bound + int(mask2)

    def _delivered_mask(receiver_idx: int, stage: int, message_masks: tuple[int, ...]) -> int:
        delivered_count = max(0, int(stage) - int(latency[receiver_idx]))
        delivered_count = min(delivered_count, len(message_masks))
        mask = 0
        for sent_mask in message_masks[:delivered_count]:
            mask |= int(sent_mask)
        return mask

    def _known_opponent_bids(
        receiver_idx: int,
        stage: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
        msg0: tuple[int, ...],
        msg1: tuple[int, ...],
    ) -> list[int]:
        opponent_idx = 1 - int(receiver_idx)
        delivered_mask = _delivered_mask(receiver_idx, stage, msg0 if receiver_idx == 0 else msg1)
        opponent_bids = bids1 if opponent_idx == 1 else bids0
        known: list[int] = []
        for stage_no in range(1, num_stages + 1):
            if delivered_mask & (1 << (stage_no - 1)):
                if len(opponent_bids) >= stage_no:
                    known.append(int(opponent_bids[stage_no - 1]))
                else:
                    known.append(-1)
            else:
                known.append(-1)
        return known

    def _proposer_infoset_string(
        stage: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
        msg0: tuple[int, ...],
        msg1: tuple[int, ...],
    ) -> str:
        bids_enc = ";".join(
            (
                f"0:{','.join(map(str, bids0))}",
                f"1:{','.join(map(str, bids1))}",
            ),
        )
        msg_enc = ";".join(
            (
                f"0:{','.join(map(str, msg0))}",
                f"1:{','.join(map(str, msg1))}",
            ),
        )
        return f"P|stage:{int(stage)}|phase:PROPOSER|bids:[{bids_enc}]|m:[{msg_enc}]"

    def _builder_infoset_string(
        builder_idx: int,
        value_idx: int,
        stage: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
        msg0: tuple[int, ...],
        msg1: tuple[int, ...],
    ) -> str:
        own_bids = bids0 if builder_idx == 0 else bids1
        known = _known_opponent_bids(builder_idx, stage, bids0, bids1, msg0, msg1)
        known_enc = ",".join("?" if bid < 0 else str(bid) for bid in known)
        delivered_mask = _delivered_mask(builder_idx, stage, msg0 if builder_idx == 0 else msg1)
        return (
            f"B{int(builder_idx)}|v:{int(value_idx)}|stage:{int(stage)}"
            f"|b:[{','.join(map(str, own_bids))}]|opp:[{known_enc}]|d:{int(delivered_mask)}"
        )

    def _legal_builder_actions(value_idx: int, own_bids: tuple[int, ...]) -> range:
        prev_bid = int(own_bids[-1]) if own_bids else 0
        max_bid = int(max_bid_index_by_value[int(value_idx)])
        if prev_bid > max_bid:
            raise ValueError("Previous bid exceeds valuation cap")
        return range(prev_bid, max_bid + 1)

    def _receiver_has_time_to_use_message(receiver_idx: int, stage: int) -> bool:
        return int(stage) + int(latency[receiver_idx]) <= num_stages

    def _scheduled_mask(message_masks: tuple[int, ...]) -> int:
        mask = 0
        for sent_mask in message_masks:
            mask |= int(sent_mask)
        return mask

    def _available_reveal_mask(receiver_idx: int, stage: int, message_masks: tuple[int, ...]) -> int:
        if not _receiver_has_time_to_use_message(receiver_idx, stage):
            return 0
        upto_stage_mask = (1 << int(stage)) - 1
        return upto_stage_mask & ~_scheduled_mask(message_masks)

    def _terminal_payoffs(
        valuation_prob: float,
        terminal_stage: int,
        v0: int,
        v1: int,
        bids0: tuple[int, ...],
        bids1: tuple[int, ...],
    ) -> np.ndarray:
        acc = np.zeros(player_count, dtype=np.float32)
        stage_idx = int(terminal_stage) - 1
        stage_bids = (int(bids0[stage_idx]), int(bids1[stage_idx]))
        stage_bid_vals = (bid_levels[stage_bids[0]], bid_levels[stage_bids[1]])
        val_vals = (value_levels[int(v0)], value_levels[int(v1)])

        max_bid = max(stage_bids)
        winners = [idx for idx, bid in enumerate(stage_bids) if bid == max_bid]
        scale = float(valuation_prob) * float(k_t[stage_idx])

        acc[0] += scale * float(max(stage_bid_vals))
        if len(winners) == 1:
            winner = winners[0]
            acc[1 + winner] += scale * float(val_vals[winner] - stage_bid_vals[winner])
        else:
            split = float(len(winners))
            for winner in winners:
                acc[1 + winner] += scale * float(val_vals[winner] - stage_bid_vals[winner]) / split
        return acc

    valuation_outcomes = nv ** num_builders
    emitted_rows = 0
    progress_every = max(1, valuation_outcomes // 20)
    for valuation_action in range(valuation_outcomes):
        valuation_prob = float(valuation_dist[valuation_action])
        remaining = int(valuation_action)
        v1 = remaining % nv
        remaining //= nv
        v0 = remaining % nv

        if debug and (
            valuation_action == 0
            or valuation_action + 1 == valuation_outcomes
            or (valuation_action + 1) % progress_every == 0
        ):
            print(
                '[nogret.serial] Direct rows progress | '
                f'game=epbs_vanilla valuation={valuation_action + 1}/{valuation_outcomes} '
                f'emitted_rows={emitted_rows}',
                flush=True,
            )

        stack: list[tuple[str, int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple, tuple, tuple]] = [
            ("B0", 1, (), (), (), (), (), (), ()),
        ]
        while stack:
            phase, stage, bids0, bids1, msg0, msg1, seq_p, seq_b0, seq_b1 = stack.pop()

            if phase == "B0":
                infoset = state_key_from_infoset(1, _builder_infoset_string(0, v0, stage, bids0, bids1, msg0, msg1))
                frames: list[tuple[str, int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple, tuple, tuple]] = []
                for action0 in _legal_builder_actions(v0, bids0):
                    action0 = int(action0)
                    frames.append((
                        "B1",
                        stage,
                        bids0 + (action0,),
                        bids1,
                        msg0,
                        msg1,
                        seq_p,
                        (infoset, action0),
                        seq_b1,
                    ))
                stack.extend(reversed(frames))
                continue

            if phase == "B1":
                infoset = state_key_from_infoset(2, _builder_infoset_string(1, v1, stage, bids0, bids1, msg0, msg1))
                frames = []
                for action1 in _legal_builder_actions(v1, bids1):
                    action1 = int(action1)
                    new_seq_b1 = (infoset, action1)
                    new_bids1 = bids1 + (action1,)

                    if stage == num_stages:
                        coords_row = [
                            int(external_sequence_index[0][seq_p]),
                            int(external_sequence_index[1][seq_b0]),
                            int(external_sequence_index[2][new_seq_b1]),
                        ]
                        emit_row(coords_row, _terminal_payoffs(valuation_prob, stage, v0, v1, bids0, new_bids1))
                        emitted_rows += 1
                    else:
                        frames.append((
                            "P",
                            stage,
                            bids0,
                            new_bids1,
                            msg0,
                            msg1,
                            seq_p,
                            seq_b0,
                            new_seq_b1,
                        ))
                stack.extend(reversed(frames))
                continue

            if phase != "P":
                raise ValueError(f"unsupported phase {phase!r}")

            infoset = state_key_from_infoset(0, _proposer_infoset_string(stage, bids0, bids1, msg0, msg1))
            allowed_b0 = _available_reveal_mask(0, stage, msg0)
            allowed_b1 = _available_reveal_mask(1, stage, msg1)

            coords_row = [
                int(external_sequence_index[0][(infoset, 0)]),
                int(external_sequence_index[1][seq_b0]),
                int(external_sequence_index[2][seq_b1]),
            ]
            emit_row(coords_row, _terminal_payoffs(valuation_prob, stage, v0, v1, bids0, bids1))
            emitted_rows += 1

            frames = []
            for mask0 in _submasks(allowed_b0):
                for mask1 in _submasks(allowed_b1):
                    action = _encode_message_pair(mask0, mask1)
                    frames.append((
                        "B0",
                        stage + 1,
                        bids0,
                        bids1,
                        msg0 + (int(mask0),),
                        msg1 + (int(mask1),),
                        (infoset, action),
                        seq_b0,
                        seq_b1,
                    ))
            stack.extend(reversed(frames))
