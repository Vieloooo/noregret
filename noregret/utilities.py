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


def euclidean_projection_on_probability_simplex(input_):
    """Euclidean projection of the input on a probability simplex.

    >>> euclidean_projection_on_probability_simplex(np.array([0.2, 0.5, 0.3]))
    array([0.2, 0.5, 0.3])
    >>> euclidean_projection_on_probability_simplex(np.array([0.2, -0.3, 2]))
    array([0., 0., 1.])
    >>> euclidean_projection_on_probability_simplex(np.array([5, 5, 5, 5]))
    array([0.25, 0.25, 0.25, 0.25])
    >>> euclidean_projection_on_probability_simplex(np.array([10, 0, 0]))
    array([1., 0., 0.])
    >>> euclidean_projection_on_probability_simplex(np.array([0.6]))
    array([1.])
    >>> euclidean_projection_on_probability_simplex(np.array([0, 0, 0, 0, 0]))
    array([0.2, 0.2, 0.2, 0.2, 0.2])

    :param input_: The input to be projected.
    :return: The projection output.
    """
    sorted_input = np.flip(np.sort(input_))
    cumsum_sorted_input = sorted_input.cumsum()
    indices = np.arange(1, input_.size + 1)
    conditions = (sorted_input + (1 - cumsum_sorted_input) / indices) > 0
    rho = np.where(conditions)[0].max() + 1
    lambda_ = (1 - cumsum_sorted_input[rho - 1]) / rho
    output = (input_ + lambda_).clip(0)

    return output


def stationary_distribution(stochastic_matrix):
    P = stochastic_matrix
    eigenvalues, eigenvectors = LA.eig(P)
    pi = eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0]
    pi /= pi.sum()
    pi = pi.real

    assert np.allclose(pi @ P, pi)

    return pi


def import_string(dotted_path):
    """Import an object from a module.

    >>> import_string('math.inf')
    inf

    :param dotted_path: The dotted path of the object to import.
    :return: Imported object.
    """
    module_path, class_name = dotted_path.rsplit('.', 1)

    return getattr(import_module(module_path), class_name)


def split(values, counts):
    """Split concatenated values.

    >>> split([0, 1, 2, 3, 4, 5], [3, 0, 1, 2])
    [[0, 1, 2], [], [3], [4, 5]]

    :param values: Values to be split.
    :param counts: The size of the partitions.
    :return: The split values.
    """
    splits = []
    begin = 0

    for count in counts:
        end = begin + count

        splits.append(values[begin:end])

        begin = end

    return splits


def sample(values, probabilities):
    return choices(values, probabilities)[0]


class Serializable(ABC):
    @classmethod
    @abstractmethod
    def deserialize(cls, raw_data):
        pass

    @classmethod
    def load(cls, *args, **kwargs):
        return cls.deserialize(load(*args, **kwargs))

    @classmethod
    def loads(cls, *args, **kwargs):
        return cls.deserialize(loads(*args, **kwargs))

    @abstractmethod
    def serialize(self):
        pass

    def dump(self, *args, **kwargs):
        return dump(self.serialize(), *args, **kwargs)

    def dumps(self, *args, **kwargs):
        return dumps(self.serialize(), *args, **kwargs)


@dataclass
class TreeFormSequentialDecisionProcess(Serializable):
    class NodeType(IntEnum):
        DECISION_POINT = auto()
        OBSERVATION_POINT = auto()
        END_OF_THE_DECISION_PROCESS = auto()

    @classmethod
    def deserialize_all(cls, raw_data):
        return list(map(cls.deserialize, raw_data))

    @classmethod
    def deserialize(cls, raw_data):
        transitions = {}
        node_types = {}

        for raw_transition in raw_data:
            parent_edge = tuple(raw_transition['parent_edge'])
            node_id = raw_transition['node']['id']
            node_type = getattr(cls.NodeType, raw_transition['node']['type'])
            transitions[parent_edge] = node_id
            node_types[node_id] = node_type

        return cls(transitions, node_types)

    transitions: Any
    """Entries are assumed be topologically sorted."""
    node_types: Any
    nodes: Any = field(init=False, default_factory=OrderedSet)
    decision_points: Any = field(init=False, default_factory=OrderedSet)
    observation_points: Any = field(init=False, default_factory=OrderedSet)
    sequences: Any = field(init=False, default_factory=OrderedSet)
    parent_sequences: Any = field(init=False, default_factory=dict)
    actions: Any = field(
        init=False,
        default_factory=partial(defaultdict, OrderedSet),
    )
    signals: Any = field(
        init=False,
        default_factory=partial(defaultdict, OrderedSet),
    )

    def __post_init__(self):
        self.nodes.update(self.transitions.values())

        for parent_edge, p in self.transitions.items():
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    self.decision_points.add(p)
                case self.NodeType.OBSERVATION_POINT:
                    self.observation_points.add(p)
                case self.NodeType.END_OF_THE_DECISION_PROCESS:
                    pass
                case _:
                    raise ValueError('unknown node type')

            if parent_edge:
                parent, event = parent_edge

                match self.node_types[parent]:
                    case self.NodeType.DECISION_POINT:
                        is_sequence = True
                        parent_sequence = parent_edge

                        self.actions[parent].add(event)
                    case self.NodeType.OBSERVATION_POINT:
                        is_sequence = False
                        parent_sequence = self.parent_sequences[parent]

                        self.signals[parent].add(event)
                    case self.NodeType.END_OF_THE_DECISION_PROCESS:
                        raise ValueError('parent is an end of the tfsdp')
                    case _:
                        raise ValueError('unknown parent node type')
            else:
                is_sequence = True
                parent_sequence = parent_edge

            if is_sequence:
                self.sequences.add(parent_edge)

            self.parent_sequences[p] = parent_sequence

    def behavioral_uniform_strategy(self):
        strategy = {
            j: np.full(len(self.actions[j]), 1 / len(self.actions[j]))
            for j in self.decision_points
        }

        return strategy

    def behavioral_best_response(self, utility):
        strategy = {}
        V = defaultdict(int)

        for p in reversed(self.nodes):
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    V[p] = -inf
                    index = None

                    for i, a in enumerate(self.actions[p]):
                        value = (
                            utility[self.sequences.index((p, a))]
                            + V[self.transitions[p, a]]
                        )

                        if V[p] < value:
                            V[p] = value
                            index = i

                    strategy[p] = np.zeros(len(self.actions[p]))
                    strategy[p][index] = 1
                case self.NodeType.OBSERVATION_POINT:
                    for s in self.signals[p]:
                        V[p] += V[self.transitions[p, s]]

        return strategy, V[self.nodes[0]]

    def sequence_form_best_response(self, utility):
        strategy, value = self.behavioral_best_response(utility)

        return self.behavioral_to_sequence_form(strategy), value

    def behavioral_to_sequence_form(self, behavioral_strategy):
        strategy = np.zeros(len(self.sequences))
        strategy[self.sequences.index(())] = 1

        for j in self.decision_points:
            p_j = self.parent_sequences[j]

            for i, a in enumerate(self.actions[j]):
                strategy[self.sequences.index((j, a))] = (
                    behavioral_strategy[j][i]
                    * strategy[self.sequences.index(p_j)]
                )

        return strategy

    def counterfactual_utilities(self, behavioral_strategy, utility):
        V = defaultdict(int)

        for p in reversed(self.nodes):
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    for i, a in enumerate(self.actions[p]):
                        V[p] += (
                            behavioral_strategy[p][i]
                            * (
                                utility[self.sequences.index((p, a))]
                                + V[self.transitions[p, a]]
                            )
                        )
                case self.NodeType.OBSERVATION_POINT:
                    for s in self.signals[p]:
                        V[p] += V[self.transitions[p, s]]

        utilities = {}

        for j in self.decision_points:
            utilities[j] = np.empty(len(self.actions[j]))

            for i, a in enumerate(self.actions[j]):
                utilities[j][i] = (
                    utility[self.sequences.index((j, a))]
                    + V[self.transitions[j, a]]
                )

        return utilities

    def serialize(self):
        raw_data = []

        for parent_edge, p in self.transitions.items():
            raw_data.append(
                {
                    'parent_edge': parent_edge,
                    'node': {'id': p, 'type': self.node_types[p].name},
                },
            )

        return raw_data


def persist_openspiel_tfsdp_per_agent(
        game: Any,
        out_dir: str | os.PathLike,
        *,
        file_prefix: str = 'tfsdp_player',
        split_depth: int = 1,
        num_workers: int | None = None,
        hash_digest_size: int = 16,
        compress: bool = False,
) -> list[Path]:
    """Build and persist TFSDP(s) from an OpenSpiel game, one file per player.

    This is a faster/smaller alternative to round-tripping through JSON:
    - encodes information sets / history nodes with a fixed-size hash;
    - encodes actions/signals with per-node small integers;
    - writes TFSDPs per player to avoid sequential read bottlenecks.

    Files are written atomically as `{file_prefix}{p}.pkl[.gz]` under `out_dir`.
    """
    try:
        from pyspiel import SpielError  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            'pyspiel is required to build TFSDPs from an OpenSpiel game.',
        ) from e

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if split_depth < 0:
        raise ValueError('split_depth must be >= 0')
    if hash_digest_size < 8:
        # 64-bit digests are already tiny; going below that is asking for trouble.
        raise ValueError('hash_digest_size must be >= 8 bytes')

    player_count = game.num_players()
    # children[p][parent_sequence] = set(child_infoset_hashes)
    children: list[dict[tuple, set[int]]] = [
        defaultdict(set) for _ in range(player_count)
    ]
    # actions[p][infoset_hash] = set(action_int)
    actions: list[dict[int, set[int]]] = [
        defaultdict(set) for _ in range(player_count)
    ]

    def _state_key(state: Any) -> int:
        # Prefer OpenSpiel's information state for the current player. Some
        # (perfect-information) games might not implement it, so fall back to a
        # stable textual representation.
        try:
            s = state.information_state_string()
        except SpielError:
            # history_str() is available in many games and is more stable than str().
            s = state.history_str() if hasattr(state, 'history_str') else str(state)

        digest = hashlib.blake2b(
            s.encode('utf-8', errors='strict'),
            digest_size=hash_digest_size,
            person=b'nogret-openspiel',
        ).digest()
        return int.from_bytes(digest, 'big', signed=False)

    def _process_decision_node(state: Any, sequences: list[tuple]) -> None:
        player = state.current_player()
        infoset = _state_key(state)
        parent_sequence = sequences[player]

        children[player][parent_sequence].add(infoset)
        # Ensure the infoset exists even if it has no legal actions (rare, but
        # avoids KeyError in downstream indexing).
        actions[player].setdefault(infoset, set())

        # Record legal actions for this information set and ensure a key exists
        # for each continuation sequence.
        for action in state.legal_actions():
            actions[player][infoset].add(int(action))
            children[player].setdefault((infoset, int(action)), set())

    def _explore_many(roots: list[tuple[Any, list[tuple]]]):
        local_children: list[dict[tuple, set[int]]] = [
            defaultdict(set) for _ in range(player_count)
        ]
        local_actions: list[dict[int, set[int]]] = [
            defaultdict(set) for _ in range(player_count)
        ]

        # Small helper that mirrors _process_decision_node but writes locally.
        def _local_process(state: Any, sequences: list[tuple]) -> None:
            player = state.current_player()
            infoset = _state_key(state)
            parent_sequence = sequences[player]

            local_children[player][parent_sequence].add(infoset)
            local_actions[player].setdefault(infoset, set())

            for action in state.legal_actions():
                local_actions[player][infoset].add(int(action))
                local_children[player].setdefault((infoset, int(action)), set())

        stack: list[tuple[Any, list[tuple]]] = roots[:]

        while stack:
            state, sequences = stack.pop()
            if state.is_terminal():
                continue
            if state.is_chance_node():
                for action, _prob in state.chance_outcomes():
                    stack.append((state.child(action), sequences))
                continue

            _local_process(state, sequences)
            player = state.current_player()
            infoset = _state_key(state)

            for action in state.legal_actions():
                child = state.child(action)
                child_sequences = sequences.copy()
                child_sequences[player] = (infoset, int(action))
                stack.append((child, child_sequences))

        return local_children, local_actions

    # Build a frontier up to split_depth, processing nodes above the frontier
    # sequentially to avoid double counting.
    init_sequences: list[tuple] = [()] * player_count
    frontier: list[tuple[Any, list[tuple]]] = []
    stack: list[tuple[Any, list[tuple], int]] = [(game.new_initial_state(), init_sequences, 0)]

    while stack:
        state, sequences, depth = stack.pop()
        if depth >= split_depth:
            frontier.append((state, sequences))
            continue
        if state.is_terminal():
            continue
        if state.is_chance_node():
            for action, _prob in state.chance_outcomes():
                stack.append((state.child(action), sequences, depth + 1))
            continue

        _process_decision_node(state, sequences)
        player = state.current_player()
        infoset = _state_key(state)

        for action in state.legal_actions():
            child = state.child(action)
            child_sequences = sequences.copy()
            child_sequences[player] = (infoset, int(action))
            stack.append((child, child_sequences, depth + 1))

    # Parallel exploration of the frontier.
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1))
    num_workers = max(1, int(num_workers))

    # Chunk frontier to avoid "one task per node" overhead when the root fans out.
    chunks: list[list[tuple[Any, list[tuple]]]] = []
    if frontier:
        chunk_count = min(num_workers, len(frontier))
        chunks = [[] for _ in range(chunk_count)]
        for i, item in enumerate(frontier):
            chunks[i % chunk_count].append(item)

    if chunks and len(chunks) > 1:
        import concurrent.futures as _fut

        results = []
        with _fut.ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            for chunk in chunks:
                results.append(ex.submit(_explore_many, chunk))
            for fut in results:
                local_children, local_actions = fut.result()
                for p in range(player_count):
                    for k, v in local_children[p].items():
                        children[p][k].update(v)
                    for k, v in local_actions[p].items():
                        actions[p][k].update(v)
    elif frontier or split_depth == 0:
        # Either split_depth=0 (nothing has been processed yet), or a small
        # frontier that we explore sequentially in one worker to avoid overhead.
        local_children, local_actions = _explore_many(
            frontier or [(game.new_initial_state(), init_sequences)],
        )
        for p in range(player_count):
            for k, v in local_children[p].items():
                children[p][k].update(v)
            for k, v in local_actions[p].items():
                actions[p][k].update(v)
    else:
        # The full tree was processed sequentially while building the frontier.
        pass

    def _build_tfsdp(
            player: int,
            children_map: dict[tuple, set[int]],
            actions_map: dict[int, set[int]],
    ) -> TreeFormSequentialDecisionProcess:
        # Deterministic node ids: sort by hashed infoset key.
        infoset_set: set[int] = set(actions_map.keys())
        for parent_seq, next_infos in children_map.items():
            if parent_seq:
                infoset_set.add(int(parent_seq[0]))
            infoset_set.update(next_infos)
        infosets = sorted(infoset_set)
        decision_id = {h: i for i, h in enumerate(infosets)}

        # Deterministic per-node event ids: sort by OpenSpiel action id.
        action_event_id: dict[int, dict[int, int]] = {}
        for h in infosets:
            ordered = sorted(actions_map.get(h, set()))
            action_event_id[h] = {a: i for i, a in enumerate(ordered)}

        END = -1
        node_types: dict[int, TreeFormSequentialDecisionProcess.NodeType] = {
            END: TreeFormSequentialDecisionProcess.NodeType.END_OF_THE_DECISION_PROCESS,
        }
        for _, nid in decision_id.items():
            node_types[nid] = TreeFormSequentialDecisionProcess.NodeType.DECISION_POINT

        transitions: dict[tuple, int] = {}
        obs_next = len(decision_id)

        def _target_node(parent_sequence: tuple) -> int:
            nonlocal obs_next
            next_infosets = children_map.get(parent_sequence, set())
            if not next_infosets:
                return END
            if len(next_infosets) == 1:
                h = next(iter(next_infosets))
                return decision_id[h]

            obs_id = obs_next
            obs_next += 1
            node_types[obs_id] = TreeFormSequentialDecisionProcess.NodeType.OBSERVATION_POINT

            # Deterministic signal ordering based on child decision node id.
            ordered = sorted(next_infosets, key=lambda x: decision_id[x])
            for i, h in enumerate(ordered):
                transitions[(obs_id, i)] = decision_id[h]
            return obs_id

        def _expand_from_root():
            # Iterative DFS to avoid Python recursion limits on large games.
            stack_edges: list[tuple] = [()]

            while stack_edges:
                parent_edge = stack_edges.pop()
                child_node = transitions[parent_edge]
                if child_node == END:
                    continue

                ntype = node_types[child_node]
                if ntype == TreeFormSequentialDecisionProcess.NodeType.DECISION_POINT:
                    # Enumerate actions from this decision point in a stable order.
                    infoset_hash = infosets[child_node]
                    # event ids are already 0..k-1, so sort by eid for stability.
                    ordered = sorted(
                        action_event_id[infoset_hash].items(),
                        key=lambda kv: kv[1],
                    )
                    edges: list[tuple] = []
                    for action, eid in ordered:
                        edge = (child_node, eid)
                        if edge not in transitions:
                            transitions[edge] = _target_node((infoset_hash, action))
                        edges.append(edge)
                    # DFS stack is LIFO; push in reverse so smaller eid is processed first.
                    stack_edges.extend(reversed(edges))
                elif ntype == TreeFormSequentialDecisionProcess.NodeType.OBSERVATION_POINT:
                    # Expand each signal edge (added in _target_node).
                    i = 0
                    edges = []
                    while (child_node, i) in transitions:
                        edges.append((child_node, i))
                        i += 1
                    stack_edges.extend(reversed(edges))

        # Build all reachable transitions from the root.
        transitions[()] = _target_node(())
        _expand_from_root()

        return TreeFormSequentialDecisionProcess(transitions, node_types)

    tfsdps = [
        _build_tfsdp(p, children[p], actions[p])
        for p in range(player_count)
    ]

    # Write one file per player; include minimal metadata for future proofing.
    meta = {
        'format': 'nogret.tfsdp.openspiel.per_agent',
        'version': 1,
        'player_count': player_count,
        'hash_digest_size': hash_digest_size,
    }

    meta_path = out_path / f'{file_prefix}.meta.pkl'
    tmp_meta = meta_path.with_suffix(meta_path.suffix + '.tmp')
    with open(tmp_meta, 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_meta, meta_path)

    paths: list[Path] = []
    for p, tfsdp in enumerate(tfsdps):
        suffix = '.pkl.gz' if compress else '.pkl'
        path = out_path / f'{file_prefix}{p}{suffix}'
        tmp = path.with_suffix(path.suffix + '.tmp')

        if compress:
            with gzip.open(tmp, 'wb', compresslevel=1) as f:
                pickle.dump(tfsdp, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(tmp, 'wb') as f:
                pickle.dump(tfsdp, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        paths.append(path)

    return paths


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
