from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import auto, IntEnum
from functools import partial
import gzip
import hashlib
from importlib import import_module
from json import dump, dumps, load, loads
from math import inf
import os
from pathlib import Path
import pickle
from random import choices
import threading
from typing import Any, Callable

from ordered_set import OrderedSet
import numpy as np
import numpy.linalg as LA

from noregret.epbs_vanilla_compiler import (
    can_direct_compile_epbs_vanilla,
    direct_compiler_eligibility_reason as epbs_vanilla_direct_compiler_eligibility_reason,
    discover_epbs_vanilla_direct,
    emit_epbs_vanilla_direct_rows,
)
from noregret.pbs_passive_compiler import (
    can_direct_compile_pbs_passive,
    direct_compiler_eligibility_reason,
    discover_pbs_passive_direct,
    emit_pbs_passive_direct_rows,
)
from noregret.utilities import *
from pyspiel import GameType, SpielError



def _fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return 'NA'
    try:
        n = float(n)
    except Exception:
        return 'NA'
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f'{n:.2f}{units[i]}'


def _dprint(enabled: bool, msg: str, **meta: Any) -> None:
    if not enabled:
        return
    if meta:
        payload = ' '.join(f'{k}={v}' for k, v in meta.items())
        print(f'[nogret.serial] {msg} | {payload}', flush=True)
    else:
        print(f'[nogret.serial] {msg}', flush=True)


class _ProfilePerPlayerRowSink:
    """Append-only sink for packed sparse utility profile rows."""

    def append(self, coords_row: list[int], values_row: np.ndarray) -> None:
        raise NotImplementedError

    def finalize(self) -> dict[str, Any]:
        raise NotImplementedError


class _InMemoryProfilePerPlayerRowSink(_ProfilePerPlayerRowSink):
    """In-memory row sink that preserves the current packed bundle schema."""

    def __init__(
        self,
        *,
        row_count: int,
        player_count: int,
        coord_dtype: np.dtype = np.int64,
        value_dtype: np.dtype = np.float32,
    ) -> None:
        self._row_count = int(row_count)
        self._player_count = int(player_count)
        self._coords = np.empty((self._row_count, self._player_count), dtype=coord_dtype)
        self._values = [
            np.empty(self._row_count, dtype=value_dtype)
            for _ in range(self._player_count)
        ]
        self._cursor = 0

    def append(self, coords_row: list[int], values_row: np.ndarray) -> None:
        row_idx = int(self._cursor)
        if row_idx >= self._row_count:
            raise ValueError('utility row sink capacity exceeded')
        if len(coords_row) != self._player_count:
            raise ValueError('coords_row does not match player_count')
        if len(values_row) != self._player_count:
            raise ValueError('values_row does not match player_count')

        self._coords[row_idx, :] = coords_row
        for p in range(self._player_count):
            self._values[p][row_idx] = np.float32(values_row[p])
        self._cursor = row_idx + 1

    def finalize(self) -> dict[str, Any]:
        nnz = int(self._cursor)
        if nnz != self._row_count:
            coords = self._coords[:nnz].copy()
            values = [self._values[p][:nnz].copy() for p in range(self._player_count)]
        else:
            coords = self._coords
            values = self._values

        indptr = np.arange(nnz + 1, dtype=np.int64)
        indices = np.zeros(nnz, dtype=np.int32)
        payloads = [
            {
                'type': 'csr',
                'shape': (nnz, 1),
                'dtype': str(values[p].dtype),
                'data': values[p],
                'indices': indices,
                'indptr': indptr,
            }
            for p in range(self._player_count)
        ]
        return {
            'kind': 'scipy.sparse.profile_per_player',
            'player_count': int(self._player_count),
            'zero_sum': False,
            'coords': coords,
            'values': payloads,
        }


@dataclass
class _BuiltDecisionProcesses:
    tfsdps: list[TreeFormSequentialDecisionProcess]
    raw_tfsdps: list[Any]
    external_sequence_index: list[dict[tuple, int]]


def _sort_profile_per_player_packed_utilities(raw_utilities: dict[str, Any]) -> dict[str, Any]:
    coords = np.asarray(raw_utilities['coords'])
    if coords.ndim != 2 or coords.shape[0] <= 1:
        return raw_utilities

    order = np.lexsort(tuple(coords[:, col] for col in range(coords.shape[1] - 1, -1, -1)))
    raw_utilities['coords'] = coords[order]
    for payload in raw_utilities['values']:
        payload['data'] = np.asarray(payload['data'])[order]
    return raw_utilities


def _make_state_key_from_infoset(
    *,
    player_count: int,
    hash_infosets: bool,
    hash_digest_size: int,
    check_hash_collisions: bool,
):
    if hash_infosets and hash_digest_size < 8:
        raise ValueError('hash_digest_size must be >= 8 bytes')

    infoset_strings: list[dict[Any, str]] = [
        {} for _ in range(player_count)
    ] if check_hash_collisions and hash_infosets else []

    def _state_key_from_infoset(player: int, infoset_str: str) -> Any:
        if not hash_infosets:
            return infoset_str

        digest = hashlib.blake2b(
            infoset_str.encode('utf-8', errors='strict'),
            digest_size=hash_digest_size,
            person=b'history',
        ).digest()
        infoset = int.from_bytes(digest, 'big', signed=False)

        if infoset_strings:
            prev = infoset_strings[player].get(infoset)
            if prev is None:
                infoset_strings[player][infoset] = infoset_str
            elif prev != infoset_str:
                raise ValueError(
                    'Information state hash collision detected; increase hash_digest_size.',
                )

        return infoset

    return _state_key_from_infoset


def _discover_game_via_state_dfs(
    game: Any,
    *,
    emit_policy_specs: bool,
    state_key_from_infoset,
    debug: bool,
) -> dict[str, Any]:
    player_count = game.num_players()
    children: list[dict[tuple, OrderedSet]] = [
        {(): OrderedSet()} for _ in range(player_count)
    ]
    actions: list[dict[int, OrderedSet]] = [
        {} for _ in range(player_count)
    ]
    infoset_order: list[list[Any]] = [
        [] for _ in range(player_count)
    ]
    infoset_labels: list[dict[Any, str]] = [
        {} for _ in range(player_count)
    ] if emit_policy_specs else []

    utilities_hashed: defaultdict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(player_count, dtype=np.float32),
    )

    init_sequences: list[tuple] = [()] * player_count
    stack: list[tuple[Any, float, list[tuple]]] = [
        (game.new_initial_state(), 1.0, init_sequences),
    ]

    expanded_states = 0
    terminal_states = 0
    chance_states = 0
    decision_states = 0
    max_stack = len(stack)

    while stack:
        state, chance_prob, sequences = stack.pop()
        expanded_states += 1
        if len(stack) > max_stack:
            max_stack = len(stack)

        if state.is_terminal():
            terminal_states += 1
            acc = utilities_hashed[tuple(sequences)]
            rewards = state.rewards()
            for i in range(player_count):
                acc[i] += chance_prob * float(rewards[i])
            continue

        if state.is_chance_node():
            chance_states += 1
            frames: list[tuple[Any, float, list[tuple]]] = []
            for action, prob in state.chance_outcomes():
                frames.append((
                    state.child(action),
                    chance_prob * float(prob),
                    sequences,
                ))
            stack.extend(reversed(frames))
            continue

        player = state.current_player()
        decision_states += 1
        try:
            infoset_str = state.information_state_string()
        except SpielError:
            infoset_str = state.history_str() if hasattr(state, 'history_str') else str(state)

        infoset = state_key_from_infoset(player, infoset_str)
        parent_sequence = sequences[player]
        children[player].setdefault(parent_sequence, OrderedSet()).add(infoset)
        if infoset not in actions[player]:
            actions[player][infoset] = OrderedSet()
            infoset_order[player].append(infoset)
            if infoset_labels:
                infoset_labels[player][infoset] = infoset_str

        frames = []
        for action in state.legal_actions():
            a = int(action)
            actions[player][infoset].add(a)
            children[player].setdefault((infoset, a), OrderedSet())

            child = state.child(action)
            child_sequences = sequences.copy()
            child_sequences[player] = (infoset, a)
            frames.append((child, chance_prob, child_sequences))

        stack.extend(reversed(frames))

    _dprint(
        debug,
        'DFS finished',
        expanded_states=int(expanded_states),
        terminal_states=int(terminal_states),
        chance_states=int(chance_states),
        decision_states=int(decision_states),
        utilities_profiles=int(len(utilities_hashed)),
        max_stack=int(max_stack),
    )

    return {
        'player_count': player_count,
        'children': children,
        'actions': actions,
        'infoset_order': infoset_order,
        'infoset_labels': infoset_labels,
        'utilities_hashed': utilities_hashed,
    }


def _build_decision_processes_from_discovery(
    *,
    children: list[dict[tuple, OrderedSet]],
    actions: list[dict[Any, OrderedSet]],
    debug: bool,
) -> _BuiltDecisionProcesses:
    player_count = len(actions)
    decision_ids: list[dict[Any, int]] = []
    action_event_ids: list[dict[Any, dict[int, int]]] = []
    tfsdps: list[TreeFormSequentialDecisionProcess] = []
    external_sequence_index: list[dict[tuple, int]] = []

    for p in range(player_count):
        children_map = children[p]
        actions_map = actions[p]
        infosets = list(actions_map.keys())
        decision_id = {h: i for i, h in enumerate(infosets)}

        action_event_id: dict[Any, dict[int, int]] = {}
        for h in infosets:
            ordered_actions = list(actions_map.get(h, OrderedSet()))
            action_event_id[h] = {a: i for i, a in enumerate(ordered_actions)}

        END = -1
        node_types: dict[int, TreeFormSequentialDecisionProcess.NodeType] = {
            END: TreeFormSequentialDecisionProcess.NodeType.END_OF_THE_DECISION_PROCESS,
        }
        for _, nid in decision_id.items():
            node_types[nid] = TreeFormSequentialDecisionProcess.NodeType.DECISION_POINT

        transitions: dict[tuple, int] = {}
        obs_next = len(decision_id)
        pending_obs_children: dict[int, list[int]] = {}

        def _target_node(parent_sequence: tuple) -> int:
            nonlocal obs_next
            next_infosets = children_map.get(parent_sequence, OrderedSet())
            if not next_infosets:
                return END
            if len(next_infosets) == 1:
                h = next(iter(next_infosets))
                return decision_id[h]

            obs_id = obs_next
            obs_next += 1
            node_types[obs_id] = TreeFormSequentialDecisionProcess.NodeType.OBSERVATION_POINT
            pending_obs_children[obs_id] = [decision_id[h] for h in next_infosets]
            return obs_id

        def _expand_from_root() -> None:
            stack_edges: list[tuple] = [()]
            while stack_edges:
                parent_edge = stack_edges.pop()
                child_node = transitions[parent_edge]
                if child_node == END:
                    continue

                ntype = node_types[child_node]
                if ntype == TreeFormSequentialDecisionProcess.NodeType.DECISION_POINT:
                    infoset_hash = infosets[child_node]
                    edges: list[tuple] = []
                    for action in actions_map.get(infoset_hash, OrderedSet()):
                        eid = action_event_id[infoset_hash][action]
                        edge = (child_node, eid)
                        if edge not in transitions:
                            transitions[edge] = _target_node((infoset_hash, action))
                        edges.append(edge)
                    stack_edges.extend(reversed(edges))
                elif ntype == TreeFormSequentialDecisionProcess.NodeType.OBSERVATION_POINT:
                    if (child_node, 0) not in transitions:
                        for i, nid in enumerate(pending_obs_children.get(child_node, [])):
                            transitions[(child_node, i)] = nid
                    i = 0
                    edges = []
                    while (child_node, i) in transitions:
                        edges.append((child_node, i))
                        i += 1
                    stack_edges.extend(reversed(edges))

        transitions[()] = _target_node(())
        _expand_from_root()

        tfsdp = TreeFormSequentialDecisionProcess(transitions, node_types)
        tfsdps.append(tfsdp)
        decision_ids.append(decision_id)
        action_event_ids.append(action_event_id)
        children_map.clear()

        seq_index = {seq: i for i, seq in enumerate(tfsdp.sequences)}
        ext_index: dict[tuple, int] = {(): int(seq_index[()])}
        for h in infosets:
            for action in actions_map.get(h, OrderedSet()):
                ext_seq = (h, int(action))
                nid = decision_id[h]
                eid = action_event_id[h][int(action)]
                ext_index[ext_seq] = int(seq_index[(nid, eid)])
        external_sequence_index.append(ext_index)

        try:
            seq_count = len(tfsdp.sequences)
        except Exception:
            seq_count = 'NA'
        obs_count = sum(
            1 for t in node_types.values()
            if t == TreeFormSequentialDecisionProcess.NodeType.OBSERVATION_POINT
        )
        _dprint(
            debug,
            'TFSDP built',
            player=int(p),
            infosets=int(len(infosets)),
            decision_points=int(len(decision_id)),
            observation_points=int(obs_count),
            sequences=seq_count,
            transitions=int(len(transitions)),
        )

    return _BuiltDecisionProcesses(
        tfsdps=tfsdps,
        raw_tfsdps=[t.to_list() for t in tfsdps],
        external_sequence_index=external_sequence_index,
    )


def _assemble_raw_game(
    *,
    raw_tfsdps: list[Any],
    raw_utilities: Any,
    player_count: int,
    infoset_order: list[list[Any]],
    infoset_labels: list[dict[Any, str]],
    actions: list[dict[Any, OrderedSet]],
    emit_policy_specs: bool,
    hash_digest_size: int,
    hash_infosets: bool,
    check_hash_collisions: bool,
    sort_utilities: bool,
    zero_sum: bool,
) -> dict[str, Any]:
    raw_game = {
        'tree_form_sequential_decision_processes': raw_tfsdps,
        'utilities': raw_utilities,
        'meta': {
            'format': 'nogret.openspiel.game.per_agent',
            'version': 5 if emit_policy_specs else 4,
            'player_count': player_count,
            'hash_digest_size': hash_digest_size,
            'hash_infosets': bool(hash_infosets),
            'check_hash_collisions': bool(check_hash_collisions),
            'sort_utilities': bool(sort_utilities),
            'zero_sum': bool(zero_sum),
            'emit_policy_specs': bool(emit_policy_specs),
        },
    }
    if emit_policy_specs:
        raw_game['policy_specs'] = [
            {
                'infosets': [
                    infoset_labels[p][infoset]
                    for infoset in infoset_order[p]
                ],
                'actions': [
                    list(actions[p].get(infoset, OrderedSet()))
                    for infoset in infoset_order[p]
                ],
            }
            for p in range(player_count)
        ]
    return raw_game


def _build_raw_game_from_discovery(
    game: Any,
    *,
    player_count: int,
    children: list[dict[tuple, OrderedSet]],
    actions: list[dict[Any, OrderedSet]],
    infoset_order: list[list[Any]],
    infoset_labels: list[dict[Any, str]],
    utilities_hashed: defaultdict[tuple, np.ndarray],
    emit_policy_specs: bool,
    hash_digest_size: int,
    hash_infosets: bool,
    check_hash_collisions: bool,
    sort_utilities: bool,
    debug: bool,
) -> dict[str, Any]:
    built = _build_decision_processes_from_discovery(
        children=children,
        actions=actions,
        debug=bool(debug),
    )

    utilities: defaultdict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(player_count, dtype=np.float32),
    )

    for hashed_sequences, vals in utilities_hashed.items():
        coords_row: list[int] = []
        for p, seq in enumerate(hashed_sequences):
            ext_seq = seq if seq else ()
            coords_row.append(int(built.external_sequence_index[p][ext_seq]))
        utilities[tuple(coords_row)] += vals
    utilities_hashed.clear()

    _dprint(
        debug,
        'Utilities mapped to internal sequences',
        nonzero_profiles=int(len(utilities)),
        shape=tuple(int(len(t.sequences)) for t in built.tfsdps),
    )

    zero_sum = (
        player_count == 2
        and game.get_type().utility == GameType.Utility.ZERO_SUM
    )

    raw_utilities: Any
    shape = tuple(len(t.sequences) for t in built.tfsdps)

    def _pack_csr_parts(
            *,
            shape_: tuple[int, int],
            dtype_: str,
            data: Any,
            indices: Any,
            indptr: Any,
    ) -> dict[str, Any]:
        return {
            'type': 'csr',
            'shape': tuple(int(x) for x in shape_),
            'dtype': str(dtype_),
            'data': data,
            'indices': indices,
            'indptr': indptr,
        }

    if zero_sum:
        try:
            from scipy.sparse import csr_array  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                'scipy is required to persist 2-player zero-sum utilities in CSR form.',
            ) from e

        nnz = len(utilities)
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=np.float32)

        for k, ((c0, c1), vals) in enumerate(utilities.items()):
            rows[k] = int(c0)
            cols[k] = int(c1)
            data[k] = np.float32(vals[0])
        utilities.clear()

        m = csr_array((data, (rows, cols)), shape=shape)
        raw_utilities = {
            'kind': 'scipy.sparse.csr',
            'zero_sum': True,
            'player_count': 2,
            'utility': _pack_csr_parts(
                shape_=tuple(int(x) for x in m.shape),
                dtype_=str(m.dtype),
                data=m.data,
                indices=m.indices,
                indptr=m.indptr,
            ),
        }
        _dprint(
            debug,
            'Packed utilities (2p zero-sum csr)',
            shape=tuple(int(x) for x in m.shape),
            nnz=int(m.nnz),
        )
    else:
        items: Any = utilities.items()
        if sort_utilities:
            items = list(items)
            items.sort(key=lambda kv: kv[0])

        nnz = len(utilities)
        row_sink = _InMemoryProfilePerPlayerRowSink(
            row_count=nnz,
            player_count=player_count,
            coord_dtype=np.int64,
            value_dtype=np.float32,
        )

        for coords_row, vals in items:
            row_sink.append(coords_row, np.asarray(vals, dtype=np.float32))
        utilities.clear()
        if sort_utilities:
            items.clear()

        raw_utilities = row_sink.finalize()
        nnz_each = [int(nnz)] * player_count
        _dprint(
            debug,
            'Packed utilities (profile_per_player)',
            nnz=int(nnz),
            coords_shape=tuple(int(x) for x in raw_utilities['coords'].shape),
            per_player_vec_nnz=nnz_each,
        )
    return _assemble_raw_game(
        raw_tfsdps=built.raw_tfsdps,
        raw_utilities=raw_utilities,
        player_count=player_count,
        infoset_order=infoset_order,
        infoset_labels=infoset_labels,
        actions=actions,
        emit_policy_specs=emit_policy_specs,
        hash_digest_size=hash_digest_size,
        hash_infosets=hash_infosets,
        check_hash_collisions=check_hash_collisions,
        sort_utilities=sort_utilities,
        zero_sum=zero_sum,
    )


def _build_raw_game_from_direct_rows(
    game: Any,
    *,
    player_count: int,
    children: list[dict[tuple, OrderedSet]],
    actions: list[dict[Any, OrderedSet]],
    infoset_order: list[list[Any]],
    infoset_labels: list[dict[Any, str]],
    emit_policy_specs: bool,
    hash_digest_size: int,
    hash_infosets: bool,
    check_hash_collisions: bool,
    sort_utilities: bool,
    terminal_histories: int,
    emit_direct_rows: Callable[[list[dict[tuple, int]], Callable[[list[int], np.ndarray], None]], None],
    debug: bool,
) -> dict[str, Any]:
    built = _build_decision_processes_from_discovery(
        children=children,
        actions=actions,
        debug=bool(debug),
    )

    row_sink = _InMemoryProfilePerPlayerRowSink(
        row_count=int(terminal_histories),
        player_count=int(player_count),
        coord_dtype=np.int64,
        value_dtype=np.float32,
    )
    emit_direct_rows(
        built.external_sequence_index,
        row_sink.append,
    )
    raw_utilities = row_sink.finalize()
    if sort_utilities:
        raw_utilities = _sort_profile_per_player_packed_utilities(raw_utilities)

    _dprint(
        debug,
        'Packed utilities (profile_per_player direct rows)',
        nnz=int(raw_utilities['coords'].shape[0]),
        coords_shape=tuple(int(x) for x in raw_utilities['coords'].shape),
        per_player_vec_nnz=[int(raw_utilities['coords'].shape[0])] * int(player_count),
    )

    return _assemble_raw_game(
        raw_tfsdps=built.raw_tfsdps,
        raw_utilities=raw_utilities,
        player_count=player_count,
        infoset_order=infoset_order,
        infoset_labels=infoset_labels,
        actions=actions,
        emit_policy_specs=emit_policy_specs,
        hash_digest_size=hash_digest_size,
        hash_infosets=hash_infosets,
        check_hash_collisions=check_hash_collisions,
        sort_utilities=sort_utilities,
        zero_sum=False,
    )



def persist_openspiel_game_per_agent(
    game: Any,
    out_dir: str | os.PathLike,
    *,
    file_prefix: str = 'openspiel_game',
    hash_digest_size: int = 8,
    hash_infosets: bool = True,
    check_hash_collisions: bool = False,
    compress: bool = False,
    sort_utilities: bool = False,
    emit_policy_specs: bool = True,
    compiler: str = 'dfs',
    debug: bool = False,
) -> Path:
    """Persist an OpenSpiel extensive-form game in a template-compatible schema.

    This mirrors the JSON structure produced by `scripts/from-open-spiel.py`:

    - `tree_form_sequential_decision_processes`: list[ list[transition] ] (one per player)
    - `utilities`: sparse list of terminal utilities keyed by per-player sequences

    but uses a more compact encoding:
    - decision points are hashed integers (via blake2b of information state strings)
    - actions are OpenSpiel action ids (integers)
    - observation points / END nodes are still represented in the TFSDP tree

    The output is pickled as `{file_prefix}.pkl[.gz]` in `out_dir`.
    """
  

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    player_count = int(game.num_players())
    state_key_from_infoset = _make_state_key_from_infoset(
        player_count=player_count,
        hash_infosets=bool(hash_infosets),
        hash_digest_size=int(hash_digest_size),
        check_hash_collisions=bool(check_hash_collisions),
    )

    compiler_requested = str(compiler)
    compiler_used = 'openspiel_dfs'
    compiler_fallback_reason: str | None = None

    _dprint(
        debug,
        'persist_openspiel_game_per_agent: start',
        out_dir=str(out_path),
        file_prefix=file_prefix,
        player_count=int(player_count),
        hash_digest_size=int(hash_digest_size),
        hash_infosets=bool(hash_infosets),
        check_hash_collisions=bool(check_hash_collisions),
        sort_utilities=bool(sort_utilities),
        compress=bool(compress),
        emit_policy_specs=bool(emit_policy_specs),
        compiler=compiler_requested,
    )

    if compiler_requested not in {'dfs', 'auto', 'pbs_passive_direct', 'epbs_vanilla_direct'}:
        raise ValueError(f'unsupported compiler mode: {compiler_requested!r}')

    if compiler_requested == 'dfs':
        discovery = _discover_game_via_state_dfs(
            game,
            emit_policy_specs=bool(emit_policy_specs),
            state_key_from_infoset=state_key_from_infoset,
            debug=bool(debug),
        )
    elif compiler_requested == 'pbs_passive_direct':
        discovery = discover_pbs_passive_direct(
            game,
            emit_policy_specs=bool(emit_policy_specs),
            state_key_from_infoset=state_key_from_infoset,
        ).__dict__
        compiler_used = 'pbs_passive_direct'
    elif compiler_requested == 'epbs_vanilla_direct':
        discovery = discover_epbs_vanilla_direct(
            game,
            emit_policy_specs=bool(emit_policy_specs),
            state_key_from_infoset=state_key_from_infoset,
        ).__dict__
        compiler_used = 'epbs_vanilla_direct'
    else:
        if can_direct_compile_pbs_passive(game):
            discovery = discover_pbs_passive_direct(
                game,
                emit_policy_specs=bool(emit_policy_specs),
                state_key_from_infoset=state_key_from_infoset,
            ).__dict__
            compiler_used = 'pbs_passive_direct'
        elif can_direct_compile_epbs_vanilla(game):
            discovery = discover_epbs_vanilla_direct(
                game,
                emit_policy_specs=bool(emit_policy_specs),
                state_key_from_infoset=state_key_from_infoset,
            ).__dict__
            compiler_used = 'epbs_vanilla_direct'
        else:
            pbs_reason = direct_compiler_eligibility_reason(game)
            epbs_reason = epbs_vanilla_direct_compiler_eligibility_reason(game)
            compiler_fallback_reason = epbs_reason if epbs_reason is not None and pbs_reason and "unsupported game short_name" in pbs_reason else pbs_reason
            discovery = _discover_game_via_state_dfs(
                game,
                emit_policy_specs=bool(emit_policy_specs),
                state_key_from_infoset=state_key_from_infoset,
                debug=bool(debug),
            )

    if compiler_used in {'pbs_passive_direct', 'epbs_vanilla_direct'}:
        stats = discovery.get('stats', {})
        _dprint(
            debug,
            'Direct compiler finished',
            valuation_outcomes=stats.get('valuation_outcomes', 'NA'),
            expanded_frames=stats.get('expanded_frames', 'NA'),
            terminal_histories=stats.get('terminal_histories', 'NA'),
            utility_profiles=stats.get('utility_profiles', 'NA'),
        )

    if compiler_used == 'pbs_passive_direct':
        raw_game = _build_raw_game_from_direct_rows(
            game,
            player_count=int(discovery['player_count']),
            children=discovery['children'],
            actions=discovery['actions'],
            infoset_order=discovery['infoset_order'],
            infoset_labels=discovery['infoset_labels'],
            emit_policy_specs=bool(emit_policy_specs),
            hash_digest_size=int(hash_digest_size),
            hash_infosets=bool(hash_infosets),
            check_hash_collisions=bool(check_hash_collisions),
            sort_utilities=bool(sort_utilities),
            terminal_histories=int(discovery.get('stats', {}).get('terminal_histories', 0)),
            emit_direct_rows=lambda external_sequence_index, emit_row: emit_pbs_passive_direct_rows(
                game,
                state_key_from_infoset=state_key_from_infoset,
                external_sequence_index=external_sequence_index,
                emit_row=emit_row,
            ),
            debug=bool(debug),
        )
    elif compiler_used == 'epbs_vanilla_direct':
        raw_game = _build_raw_game_from_direct_rows(
            game,
            player_count=int(discovery['player_count']),
            children=discovery['children'],
            actions=discovery['actions'],
            infoset_order=discovery['infoset_order'],
            infoset_labels=discovery['infoset_labels'],
            emit_policy_specs=bool(emit_policy_specs),
            hash_digest_size=int(hash_digest_size),
            hash_infosets=bool(hash_infosets),
            check_hash_collisions=bool(check_hash_collisions),
            sort_utilities=bool(sort_utilities),
            terminal_histories=int(discovery.get('stats', {}).get('terminal_histories', 0)),
            emit_direct_rows=lambda external_sequence_index, emit_row: emit_epbs_vanilla_direct_rows(
                game,
                state_key_from_infoset=state_key_from_infoset,
                external_sequence_index=external_sequence_index,
                emit_row=emit_row,
            ),
            debug=bool(debug),
        )
    else:
        raw_game = _build_raw_game_from_discovery(
            game,
            player_count=int(discovery['player_count']),
            children=discovery['children'],
            actions=discovery['actions'],
            infoset_order=discovery['infoset_order'],
            infoset_labels=discovery['infoset_labels'],
            utilities_hashed=discovery['utilities_hashed'],
            emit_policy_specs=bool(emit_policy_specs),
            hash_digest_size=int(hash_digest_size),
            hash_infosets=bool(hash_infosets),
            check_hash_collisions=bool(check_hash_collisions),
            sort_utilities=bool(sort_utilities),
            debug=bool(debug),
        )
    raw_game['meta']['compiler_requested'] = compiler_requested
    raw_game['meta']['compiler_used'] = compiler_used
    if compiler_fallback_reason is not None:
        raw_game['meta']['compiler_fallback_reason'] = compiler_fallback_reason
    discovery.clear()

    suffix = '.pkl.gz' if compress else '.pkl'
    out_file = out_path / f'{file_prefix}{suffix}'
    tmp = out_file.with_suffix(out_file.suffix + '.tmp')

    if compress:
        with gzip.open(tmp, 'wb', compresslevel=1) as f:
            pickle.dump(raw_game, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(tmp, 'wb') as f:
            pickle.dump(raw_game, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_file)

    try:
        size_b = out_file.stat().st_size
    except Exception:
        size_b = None
    _dprint(
        debug,
        'persist_openspiel_game_per_agent: done',
        out_file=str(out_file),
        file_size=_fmt_bytes(size_b),
        compiler_used=compiler_used,
    )

    return out_file


def load_openspiel_game_per_agent(
        path: str | os.PathLike,
        *,
        restore_raw_utilities: bool = False,
        vectorized_raw_restore: bool = True,
        debug: bool = False,
) -> dict[str, Any]:
    """Load a bundle written by persist_openspiel_game_per_agent()."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    _dprint(debug, 'load_openspiel_game_per_agent: start', path=str(p))

    def _restore_raw_utilities(bundle: dict[str, Any]) -> dict[str, Any]:
        utilities = bundle.get('utilities')
        if not isinstance(utilities, dict):
            return bundle

        # 2-player packed CSR format.
        if utilities.get('kind') == 'scipy.sparse.csr':
            tfsdps = TreeFormSequentialDecisionProcess.deserialize_all(
                bundle['tree_form_sequential_decision_processes'],
            )
            raw_list = []

            def _iter_csr(payload):
                # Iterate nonzeros as (i, j, value)
                from scipy.sparse import csr_array  # type: ignore
                mat = csr_array(
                    (payload['data'], payload['indices'], payload['indptr']),
                    shape=tuple(payload['shape']),
                )
                for i, j in zip(*mat.nonzero()):
                    yield int(i), int(j), mat[i, j].item()

            if bool(utilities.get('zero_sum', False)):
                for i, j, v in _iter_csr(utilities['utility']):
                    raw_list.append({
                        'sequences': [tfsdps[0].sequences[i], tfsdps[1].sequences[j]],
                        'value': float(v),
                    })
            else:
                row = utilities['row_utility']
                col = utilities['column_utility']
                # Build a dict of row payoffs first, then attach column.
                tmp: dict[tuple[int, int], list[float]] = {}
                for i, j, v in _iter_csr(row):
                    tmp[(i, j)] = [float(v), 0.0]
                for i, j, v in _iter_csr(col):
                    tmp.setdefault((i, j), [0.0, 0.0])[1] = float(v)
                for (i, j), vals in tmp.items():
                    raw_list.append({
                        'sequences': [tfsdps[0].sequences[i], tfsdps[1].sequences[j]],
                        'values': vals,
                    })

            bundle['raw_utilities'] = raw_list
            return bundle

        # n-player packed profile/tuple format.
        if utilities.get('kind') == 'scipy.sparse.profile_tuples':
            tfsdps = TreeFormSequentialDecisionProcess.deserialize_all(
                bundle['tree_form_sequential_decision_processes'],
            )
            coords = np.asarray(utilities['coords'])
            payload = utilities['values']
            from scipy.sparse import csr_array  # type: ignore
            v = csr_array(
                (payload['data'], payload['indices'], payload['indptr']),
                shape=tuple(payload['shape']),
            )

            raw_list = []
            # `v` is (nnz x 1) with object tuples at nonzero entries.
            for row_idx, _ in zip(*v.nonzero()):
                tup = v[int(row_idx), 0].item()
                seqs = [
                    tfsdps[p].sequences[int(coords[int(row_idx), p])]
                    for p in range(int(utilities['player_count']))
                ]
                raw_list.append({'sequences': seqs, 'values': list(tup)})

            bundle['raw_utilities'] = raw_list
            return bundle

        # Unified per-player sparse values format.
        if utilities.get('kind') == 'scipy.sparse.profile_per_player':
            tfsdps = TreeFormSequentialDecisionProcess.deserialize_all(
                bundle['tree_form_sequential_decision_processes'],
            )
            coords = np.asarray(utilities['coords'])
            player_count = int(utilities['player_count'])
            payloads = list(utilities['values'])

            def _csr_vector_payload_to_dense(payload: dict[str, Any], length: int) -> np.ndarray:
                if payload.get('type') != 'csr':
                    raise ValueError('unsupported sparse utility type')
                shape = tuple(payload['shape'])
                if shape != (length, 1):
                    raise ValueError('unexpected sparse vector shape')
                indptr = np.asarray(payload['indptr'], dtype=np.int64)
                data = np.asarray(payload['data'], dtype=np.float32)
                out = np.zeros(length, dtype=np.float32)
                if vectorized_raw_restore:
                    row_nnz = indptr[1:] > indptr[:-1]
                    if bool(row_nnz.any()):
                        out[row_nnz] = data[indptr[:-1][row_nnz]]
                else:
                    for i in range(length):
                        start = indptr[i]
                        end = indptr[i + 1]
                        if end > start:
                            out[i] = np.float32(data[start])
                return out

            nnz = int(coords.shape[0])
            per_player_values = [
                _csr_vector_payload_to_dense(payloads[p], nnz)
                for p in range(player_count)
            ]

            raw_list = []
            for k in range(nnz):
                seqs = [
                    tfsdps[p].sequences[int(coords[k, p])]
                    for p in range(player_count)
                ]
                vals = [float(per_player_values[p][k]) for p in range(player_count)]
                raw_list.append({'sequences': seqs, 'values': vals})

            bundle['raw_utilities'] = raw_list
            return bundle

        return bundle

    if p.suffixes[-2:] == ['.pkl', '.gz'] or p.suffix == '.gz':
        with gzip.open(p, 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(p, 'rb') as f:
            obj = pickle.load(f)

    if isinstance(obj, dict):
        meta = obj.get('meta', {}) if isinstance(obj.get('meta', {}), dict) else {}
        utilities = obj.get('utilities', {}) if isinstance(obj.get('utilities', {}), dict) else {}
        _dprint(
            debug,
            'Bundle loaded (pre-restore)',
            format=meta.get('format', 'NA'),
            version=meta.get('version', 'NA'),
            player_count=meta.get('player_count', 'NA'),
            utilities_kind=utilities.get('kind', 'NA'),
        )
        if restore_raw_utilities:
            obj = _restore_raw_utilities(obj)
            if isinstance(obj, dict) and 'raw_utilities' in obj:
                _dprint(
                    debug,
                    'Utilities restored to raw list',
                    raw_len=int(len(obj.get('raw_utilities', []))),
                )
        else:
            _dprint(debug, 'Skipped raw utility restoration')
    _dprint(debug, 'load_openspiel_game_per_agent: done')
    return obj


def load_openspiel_tfsdp_per_agent(
        in_dir: str | os.PathLike,
        *,
        file_prefix: str = 'tfsdp_player',
        player_count: int | None = None,
        num_workers: int | None = None,
) -> list[TreeFormSequentialDecisionProcess]:
    """Load TFSDP(s) previously written by persist_openspiel_tfsdp_per_agent()."""
    in_path = Path(in_dir)

    if player_count is None:
        meta_path = in_path / f'{file_prefix}.meta.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            player_count = int(meta['player_count'])
        else:
            raise FileNotFoundError(
                f'Missing `{meta_path}`; pass player_count=... explicitly.',
            )

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1))
    num_workers = max(1, int(num_workers))

    # Prefer parallel reads (per-agent files) to reduce wall-clock on large trees.
    lock = threading.Lock()
    loaded: list[TreeFormSequentialDecisionProcess | None] = [None] * player_count

    def _load_one(p: int):
        pkl = in_path / f'{file_prefix}{p}.pkl'
        pkl_gz = in_path / f'{file_prefix}{p}.pkl.gz'
        if pkl_gz.exists():
            with gzip.open(pkl_gz, 'rb') as f:
                obj = pickle.load(f)
        elif pkl.exists():
            with open(pkl, 'rb') as f:
                obj = pickle.load(f)
        else:
            raise FileNotFoundError(f'Missing TFSDP file for player {p}')

        with lock:
            loaded[p] = obj

    if player_count <= 1 or num_workers <= 1:
        for p in range(player_count):
            _load_one(p)
    else:
        import concurrent.futures as _fut

        with _fut.ThreadPoolExecutor(max_workers=min(num_workers, player_count)) as ex:
            list(ex.map(_load_one, range(player_count)))

    # mypy-friendly cast; runtime guarantees all entries filled or an exception raised.
    return [t for t in loaded if t is not None]
