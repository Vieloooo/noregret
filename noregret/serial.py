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
from typing import Any

from ordered_set import OrderedSet
import numpy as np
import numpy.linalg as LA

from noregret.utilities import *



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



def persist_openspiel_game_per_agent(
        game: Any,
        out_dir: str | os.PathLike,
        *,
        file_prefix: str = 'openspiel_game',
        hash_digest_size: int = 8,
        compress: bool = False,
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
    try:
        from pyspiel import GameType, SpielError  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            'pyspiel is required to persist OpenSpiel games.',
        ) from e

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if hash_digest_size < 8:
        raise ValueError('hash_digest_size must be >= 8 bytes')

    player_count = game.num_players()

    _dprint(
        debug,
        'persist_openspiel_game_per_agent: start',
        out_dir=str(out_path),
        file_prefix=file_prefix,
        player_count=int(player_count),
        hash_digest_size=int(hash_digest_size),
        compress=bool(compress),
    )

    def _state_key(state: Any) -> int:
        try:
            s = state.information_state_string()
        except SpielError:
            s = state.history_str() if hasattr(state, 'history_str') else str(state)

        digest = hashlib.blake2b(
            s.encode('utf-8', errors='strict'),
            digest_size=hash_digest_size,
            person=b'history',
        ).digest()
        return int.from_bytes(digest, 'big', signed=False)

    # DFS over the OpenSpiel tree (pre-order), matching `scripts/from-open-spiel.py`:
    # - preserve encounter order of infosets under each parent sequence
    # - preserve OpenSpiel's `legal_actions()` order per infoset
    # - accumulate terminal utilities (expected over chance)
    #
    # IMPORTANT: downstream TFSDP logic assumes a topologically sorted / DFS-like
    # ordering of transitions. We therefore carry encounter order through the
    # whole pipeline and avoid sorting.
    children: list[dict[tuple, OrderedSet]] = [
        {(): OrderedSet()} for _ in range(player_count)
    ]
    actions: list[dict[int, OrderedSet]] = [
        {} for _ in range(player_count)
    ]

    # Optional: store original information state strings for collision detection.
    # (We still persist hashed ids for compactness.)
    infoset_strings: list[dict[int, str]] = [
        {} for _ in range(player_count)
    ]

    utilities_hashed: defaultdict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(player_count, dtype=float),
    )

    init_sequences: list[tuple] = [()] * player_count
    stack: list[tuple[Any, float, list[tuple]]] = [
        (game.new_initial_state(), 1.0, init_sequences),
    ]

    # Lightweight traversal stats for debugging.
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
            utilities_hashed[tuple(sequences)] += (
                chance_prob * np.array(state.rewards(), dtype=float)
            )
            continue

        if state.is_chance_node():
            chance_states += 1
            # Match recursive DFS order: iterate outcomes in the given order.
            frames: list[tuple[Any, float, list[tuple]]] = []
            for action, prob in state.chance_outcomes():
                frames.append((
                    state.child(action),
                    chance_prob * float(prob),
                    sequences,
                ))
            # LIFO stack -> push reversed so first outcome is processed first.
            stack.extend(reversed(frames))
            continue

        player = state.current_player()
        decision_states += 1
        try:
            infoset_str = state.information_state_string()
        except SpielError:
            infoset_str = state.history_str() if hasattr(state, 'history_str') else str(state)

        infoset = _state_key(state)
        prev = infoset_strings[player].get(infoset)
        if prev is None:
            infoset_strings[player][infoset] = infoset_str
        elif prev != infoset_str:
            raise ValueError(
                'Information state hash collision detected; increase hash_digest_size.',
            )

        parent_sequence = sequences[player]
        children[player].setdefault(parent_sequence, OrderedSet()).add(infoset)
        actions[player].setdefault(infoset, OrderedSet())

        # Match recursive DFS order: record children in legal_actions order, then
        # push to stack in reverse.
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

    # 2) Build per-player TFSDPs using the existing compact encoding
    #    (contiguous decision ids + per-node event ids).
    decision_ids: list[dict[int, int]] = []
    action_event_ids: list[dict[int, dict[int, int]]] = []
    tfsdps: list[TreeFormSequentialDecisionProcess] = []

    for p in range(player_count):
        children_map = children[p]
        actions_map = actions[p]

        # Preserve encounter order (DFS) for decision points and per-infoset actions.
        infosets = list(actions_map.keys())
        decision_id = {h: i for i, h in enumerate(infosets)}

        action_event_id: dict[int, dict[int, int]] = {}
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

            # Preserve encounter order within this observation split.
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

        tfsdps.append(TreeFormSequentialDecisionProcess(transitions, node_types))
        decision_ids.append(decision_id)
        action_event_ids.append(action_event_id)

        # Per-player TFSDP summary.
        try:
            seq_count = len(tfsdps[-1].sequences)
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

    # Map utilities keyed by hashed sequences to utilities keyed by internal TFSDP
    # sequence edges (node_id, event_id), without re-walking the OpenSpiel tree.
    utilities: defaultdict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(player_count, dtype=float),
    )

    for hashed_sequences, vals in utilities_hashed.items():
        internal_sequences: list[tuple] = []
        for p, seq in enumerate(hashed_sequences):
            if not seq:
                internal_sequences.append(())
                continue
            infoset_hash, action_id = seq
            nid = decision_ids[p][int(infoset_hash)]
            eid = action_event_ids[p][int(infoset_hash)][int(action_id)]
            internal_sequences.append((nid, eid))

        utilities[tuple(internal_sequences)] += vals

    _dprint(
        debug,
        'Utilities mapped to internal sequences',
        nonzero_profiles=int(len(utilities)),
        shape=tuple(int(x) for x in (tuple(len(t.sequences) for t in tfsdps))),
    )

    # Serialize utilities in the same sparse list style as the template.
    zero_sum = (
        player_count == 2
        and game.get_type().utility == GameType.Utility.ZERO_SUM
    )

    # Persist utilities in a packed sparse representation.
    # - 2-player ZERO_SUM: store a single payoff matrix as SciPy CSR (float).
    # - Otherwise: store a sparse list of nonzero profiles (coords) plus one
    #   SciPy CSR sparse vector (nnz x 1) per player (float). This avoids tuple
    #   payloads and works uniformly for 2-player general-sum and n-player games.
    raw_utilities: Any
    try:
        from scipy.sparse import lil_array  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            'scipy is required to persist utilities in sparse form.',
        ) from e

    shape = tuple(len(t.sequences) for t in tfsdps)
    seq_index = [
        {seq: i for i, seq in enumerate(t.sequences)}
        for t in tfsdps
    ]

    def _pack_csr(mat: Any) -> dict[str, Any]:
        csr = mat.tocsr()
        return {
            'type': 'csr',
            'shape': tuple(int(x) for x in csr.shape),
            'dtype': str(csr.dtype),
            'data': csr.data,
            'indices': csr.indices,
            'indptr': csr.indptr,
        }

    if zero_sum:
        # Keep the special-case compact matrix representation.
        m = lil_array(shape, dtype=float)
        for (s0, s1), vals in utilities.items():
            i = seq_index[0][s0]
            j = seq_index[1][s1]
            m[i, j] = float(vals[0])
        raw_utilities = {
            'kind': 'scipy.sparse.csr',
            'zero_sum': True,
            'player_count': 2,
            'utility': _pack_csr(m),
        }
        _dprint(
            debug,
            'Packed utilities (2p zero-sum csr)',
            shape=tuple(int(x) for x in m.shape),
            nnz=int(m.nnz),
        )
    else:
        # Unified representation: sparse profiles + per-player sparse values.
        # Coords is (nnz, player_count) of per-player sequence indices.
        nnz = len(utilities)
        coords = np.empty((nnz, player_count), dtype=np.int64)
        per_player_vecs = [lil_array((nnz, 1), dtype=float) for _ in range(player_count)]

        for k, (seqs, vals) in enumerate(sorted(utilities.items(), key=lambda kv: kv[0])):
            for p in range(player_count):
                coords[k, p] = int(seq_index[p][seqs[p]])
                per_player_vecs[p][k, 0] = float(vals[p])

        raw_utilities = {
            'kind': 'scipy.sparse.profile_per_player',
            'player_count': int(player_count),
            'zero_sum': False,
            'coords': coords,
            'values': [_pack_csr(v) for v in per_player_vecs],
        }
        # Best-effort nnz from packed vectors.
        nnz_each = []
        try:
            nnz_each = [int(v.nnz) for v in per_player_vecs]
        except Exception:
            nnz_each = []
        _dprint(
            debug,
            'Packed utilities (profile_per_player)',
            nnz=int(nnz),
            coords_shape=tuple(int(x) for x in coords.shape),
            per_player_vec_nnz=nnz_each,
        )

    raw_tfsdps = [t.to_list() for t in tfsdps]
    raw_game = {
        'tree_form_sequential_decision_processes': raw_tfsdps,
        'utilities': raw_utilities,
        'meta': {
            'format': 'nogret.openspiel.game.per_agent',
            'version': 4,
            'player_count': player_count,
            'hash_digest_size': hash_digest_size,
            'zero_sum': bool(zero_sum),
        },
    }

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
    )

    return out_file


def load_openspiel_game_per_agent(
        path: str | os.PathLike,
        *,
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
                indptr = payload['indptr']
                data = payload['data']
                out = np.zeros(length, dtype=float)
                for i in range(length):
                    start = indptr[i]
                    end = indptr[i + 1]
                    if end > start:
                        out[i] = float(data[start])
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
        obj = _restore_raw_utilities(obj)
        if isinstance(obj, dict) and 'raw_utilities' in obj:
            _dprint(
                debug,
                'Utilities restored to raw list',
                raw_len=int(len(obj.get('raw_utilities', []))),
            )
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
