"""
Reactive Behavior Tree Engine for NetHack Policies.
Provides modular, decoupled decision execution:
- NodeStatus: SUCCESS, FAILURE, RUNNING
- Blackboard: shared state dictionary across nodes
- PrioritySelector (Fallback): ticks children until one succeeds/runs
- Sequence: ticks children until one fails
- ActionLeaf: executes leaf action routines
- ConditionLeaf: evaluates tactical preconditions
- BehaviorTree: root coordinator that executes tree ticks
"""

from __future__ import annotations

import inspect
import os
import traceback
from enum import Enum
from typing import Any, Callable


class NodeStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class Blackboard(dict[str, Any]):
    """Shared blackboard data store for inter-node communication and state persistence."""

    def __init__(self, *args: Any, strict_errors: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strict_errors = strict_errors
        self.leaf_errors: list[tuple[str, str]] = []
        self.last_active_leaf: str = ""
        self._selector_commits: dict[int, tuple[int, int]] = {}
        self._active_children: dict[int, int] = {}
        self._seq_indices: dict[int, int] = {}
        self._leaf_gens: dict[int, Any] = {}

    def record_leaf_error(self, node_name: str, exc: Exception) -> None:
        if len(self.leaf_errors) < 50:
            tb_loc = ""
            try:
                frames = traceback.extract_tb(exc.__traceback__)
                if frames:
                    last = frames[-1]
                    tb_loc = f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}: "
            except Exception:
                tb_loc = ""
            self.leaf_errors.append(
                (str(node_name), f"{tb_loc}{type(exc).__name__}: {exc}")
            )

    def floor_key(self, state: Any) -> tuple[int, int]:
        dnum = int(getattr(state, "dungeon_num", 0) or 0)
        dlvl = int(getattr(state, "level_num", 1) or 1)
        return (dnum, dlvl)

    def floor_get(self, state: Any, key: str, default: Any = None) -> Any:
        dnum, dlvl = self.floor_key(state)
        return self.get(("__floor__", dnum, dlvl, key), default)

    def floor_set(self, state: Any, key: str, value: Any) -> None:
        dnum, dlvl = self.floor_key(state)
        self[("__floor__", dnum, dlvl, key)] = value

    def floor_update(
        self,
        state: Any,
        key: str,
        fn: Callable[[Any], Any],
        default: Any = None,
    ) -> Any:
        cur = self.floor_get(state, key, default)
        nxt = fn(cur)
        self.floor_set(state, key, nxt)
        return nxt

    def get_var(self, key: str, default: Any = None) -> Any:
        return self.get(key, default)

    def set_var(self, key: str, value: Any) -> None:
        self[key] = value

    def has_var(self, key: str) -> bool:
        return key in self

    def floor_blacklist_add(
        self, state: Any, pos: tuple[int, int], ttl: int = 50
    ) -> None:
        """Marks `(y, x)` blacklisted on the current `(dungeon_num, level_num)` for `ttl` turns."""
        turn_now = int(getattr(state, "turn", 0) or 0)
        bl = self.floor_get(state, "__blacklist", None)
        if not isinstance(bl, dict):
            bl = {}
            self.floor_set(state, "__blacklist", bl)
        bl[(int(pos[0]), int(pos[1]))] = turn_now + max(1, int(ttl))

    def floor_is_blacklisted(self, state: Any, pos: tuple[int, int]) -> bool:
        """True if `(y, x)` has an active unexpired blacklist cooldown on the current floor."""
        bl = self.floor_get(state, "__blacklist", None)
        if not isinstance(bl, dict):
            return False
        key_p = (int(pos[0]), int(pos[1]))
        exp = bl.get(key_p)
        if exp is None:
            return False
        turn_now = int(getattr(state, "turn", 0) or 0)
        if turn_now >= int(exp):
            bl.pop(key_p, None)
            return False
        return True

    def clear_vars(self) -> None:
        self.clear()
        self.last_active_leaf = ""
        self._selector_commits.clear()
        self._active_children.clear()
        self._seq_indices.clear()
        for gen in list(self._leaf_gens.values()):
            try:
                gen.close()
            except Exception:
                pass
        self._leaf_gens.clear()


class BehaviorNode:
    """Base class for all Behavior Tree nodes."""

    def __init__(
        self,
        name: str = "",
        min_commit: int = 0,
        priority: float = 0.0,
        before: str | None = None,
        after: str | None = None,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.min_commit = max(0, int(min_commit))
        self.priority = float(priority)
        self.before = before
        self.after = after

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        """
        Ticks the node with current game state and blackboard.
        Returns `(status, action_or_none)`.
        """
        raise NotImplementedError

    def halt(self, blackboard: Blackboard) -> None:
        """Called when a higher-priority branch preempts or aborts this node."""
        return None

    def should_preempt(self, state: Any, blackboard: Blackboard) -> bool:
        """Optional predicate indicating an active `min_commit`/coroutine should abort early."""
        return False


def bt_leaf(
    priority: float = 0.0,
    before: str | None = None,
    after: str | None = None,
    name: str = "",
    min_commit: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering metadata for `PrioritySelector.from_modules(...)`."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(
            fn,
            "_lox_bt_meta",
            {
                "priority": float(priority),
                "before": str(before) if before else None,
                "after": str(after) if after else None,
                "name": str(name) if name else getattr(fn, "__name__", "leaf"),
                "min_commit": max(0, int(min_commit)),
            },
        )
        return fn

    return _decorator


def _callable_takes_two_args(fn: Any) -> bool:
    """Safely checks if a function, bound method, or callable accepts `(state, blackboard)`."""
    if fn is None or not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
        pos_cnt = 0
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                return True
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                pos_cnt += 1
        return pos_cnt >= 2
    except Exception:
        code = getattr(fn, "__code__", None)
        offset = 1 if inspect.ismethod(fn) else 0
        return bool(code and (code.co_argcount - offset) >= 2)


class ActionLeaf(BehaviorNode):
    """
    Leaf node that executes an action function or multi-turn generator coroutine.
    Function signature: `fn(state, blackboard)` or `fn(state)` -> Action | (NodeStatus, Action) | generator | None.
    """

    def __init__(
        self,
        action_fn: Callable[..., Any],
        name: str = "",
        is_running: bool = False,
        min_commit: int | None = None,
        priority: float | None = None,
        before: str | None = None,
        after: str | None = None,
        preempt_if: Callable[..., bool] | None = None,
    ) -> None:
        meta = getattr(action_fn, "_lox_bt_meta", None) or {}
        resolved_min_commit = (
            int(min_commit)
            if min_commit is not None
            else int(meta.get("min_commit", 0))
        )
        resolved_priority = (
            float(priority)
            if priority is not None
            else float(meta.get("priority", 0.0))
        )
        super().__init__(
            name=name
            or meta.get("name")
            or getattr(action_fn, "__name__", "ActionLeaf"),
            min_commit=resolved_min_commit,
            priority=resolved_priority,
            before=before if before is not None else meta.get("before"),
            after=after if after is not None else meta.get("after"),
        )
        self.action_fn = action_fn
        self.is_running = is_running
        self._takes_bb = _callable_takes_two_args(action_fn)
        self.preempt_if = preempt_if
        self._preempt_takes_bb = _callable_takes_two_args(preempt_if)

    def should_preempt(self, state: Any, blackboard: Blackboard) -> bool:
        if self.preempt_if is None:
            return False
        try:
            return bool(
                self.preempt_if(state, blackboard)
                if self._preempt_takes_bb
                else self.preempt_if(state)
            )
        except Exception:
            return False

    def halt(self, blackboard: Blackboard) -> None:
        gens = getattr(blackboard, "_leaf_gens", None)
        if isinstance(gens, dict):
            gen = gens.pop(id(self), None)
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass

    def _interpret_result(self, res: Any) -> tuple[NodeStatus, Any]:
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], NodeStatus):
            return res[0], res[1]
        if res is not None:
            status = NodeStatus.RUNNING if self.is_running else NodeStatus.SUCCESS
            return status, res
        return NodeStatus.FAILURE, None

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        gens = getattr(blackboard, "_leaf_gens", None)
        leaf_id = id(self)
        try:
            active_gen = gens.get(leaf_id) if isinstance(gens, dict) else None
            if active_gen is not None and self.should_preempt(state, blackboard):
                self.halt(blackboard)
                active_gen = None
            if active_gen is not None:
                try:
                    yielded = active_gen.send(state)
                    if (
                        isinstance(yielded, tuple)
                        and len(yielded) == 2
                        and isinstance(yielded[0], NodeStatus)
                    ):
                        if yielded[0] != NodeStatus.RUNNING:
                            gens.pop(leaf_id, None)
                            try:
                                active_gen.close()
                            except Exception:
                                pass
                        if (
                            yielded[0] != NodeStatus.FAILURE
                            and hasattr(blackboard, "last_active_leaf")
                            and yielded[1] is not None
                        ):
                            blackboard.last_active_leaf = self.name
                        return yielded[0], yielded[1]
                    if yielded is not None:
                        if hasattr(blackboard, "last_active_leaf"):
                            blackboard.last_active_leaf = self.name
                        return NodeStatus.RUNNING, yielded
                    gens.pop(leaf_id, None)
                    try:
                        active_gen.close()
                    except Exception:
                        pass
                    return NodeStatus.FAILURE, None
                except StopIteration as stop:
                    gens.pop(leaf_id, None)
                    status, act = self._interpret_result(stop.value)
                    if act is not None and hasattr(blackboard, "last_active_leaf"):
                        blackboard.last_active_leaf = self.name
                    return status, act

            raw = (
                self.action_fn(state, blackboard)
                if self._takes_bb
                else self.action_fn(state)
            )
            if inspect.isgenerator(raw):
                try:
                    first_yield = next(raw)
                except StopIteration as stop:
                    status, act = self._interpret_result(stop.value)
                    if act is not None and hasattr(blackboard, "last_active_leaf"):
                        blackboard.last_active_leaf = self.name
                    return status, act
                if isinstance(gens, dict):
                    gens[leaf_id] = raw
                if (
                    isinstance(first_yield, tuple)
                    and len(first_yield) == 2
                    and isinstance(first_yield[0], NodeStatus)
                ):
                    if first_yield[0] != NodeStatus.RUNNING and isinstance(gens, dict):
                        gens.pop(leaf_id, None)
                        try:
                            raw.close()
                        except Exception:
                            pass
                    if first_yield[1] is not None and hasattr(
                        blackboard, "last_active_leaf"
                    ):
                        blackboard.last_active_leaf = self.name
                    return first_yield[0], first_yield[1]
                if first_yield is not None:
                    if hasattr(blackboard, "last_active_leaf"):
                        blackboard.last_active_leaf = self.name
                    return NodeStatus.RUNNING, first_yield
                if isinstance(gens, dict):
                    gens.pop(leaf_id, None)
                try:
                    raw.close()
                except Exception:
                    pass
                return NodeStatus.FAILURE, None
        except Exception as exc:
            if isinstance(gens, dict):
                bad_gen = gens.pop(leaf_id, None)
                if bad_gen is not None:
                    try:
                        bad_gen.close()
                    except Exception:
                        pass
            if hasattr(blackboard, "record_leaf_error"):
                blackboard.record_leaf_error(self.name, exc)
            if getattr(blackboard, "strict_errors", False):
                raise
            return NodeStatus.FAILURE, None

        status, act = self._interpret_result(raw)
        if (
            status != NodeStatus.FAILURE
            and act is not None
            and hasattr(blackboard, "last_active_leaf")
        ):
            blackboard.last_active_leaf = self.name
        return status, act


class ConditionLeaf(BehaviorNode):
    """
    Leaf node that evaluates a boolean precondition.
    Function signature: `cond_fn(state, blackboard)` or `cond_fn(state)` -> bool.
    Returns SUCCESS if True, FAILURE if False, action is always None.
    """

    def __init__(
        self,
        cond_fn: Callable[..., bool],
        name: str = "",
    ) -> None:
        super().__init__(name=name or getattr(cond_fn, "__name__", "ConditionLeaf"))
        self.cond_fn = cond_fn
        self._takes_bb = _callable_takes_two_args(cond_fn)

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        try:
            if self._takes_bb:
                ok = bool(self.cond_fn(state, blackboard))
            else:
                ok = bool(self.cond_fn(state))
        except Exception as exc:
            if hasattr(blackboard, "record_leaf_error"):
                blackboard.record_leaf_error(self.name, exc)
            if getattr(blackboard, "strict_errors", False):
                raise
            ok = False

        status = NodeStatus.SUCCESS if ok else NodeStatus.FAILURE
        return status, None


class PrioritySelector(BehaviorNode):
    """
    Composite Selector / Fallback node.
    Ticks children sequentially from left to right.
    Tracks active child per blackboard and calls `child.halt(blackboard)` on preemption.
    Supports native hysteresis commitment (`hysteresis` or child `min_commit`).
    """

    def __init__(
        self,
        children: list[BehaviorNode] | None = None,
        name: str = "",
        hysteresis: int = 0,
        min_commit: int = 0,
    ) -> None:
        super().__init__(name=name, min_commit=min_commit)
        self.children: list[BehaviorNode] = children or []
        self.hysteresis = max(0, int(hysteresis))

    def add_child(self, child: BehaviorNode) -> PrioritySelector:
        self.children.append(child)
        return self

    def halt(self, blackboard: Blackboard) -> None:
        commits = getattr(blackboard, "_selector_commits", None)
        if isinstance(commits, dict):
            commits.pop(id(self), None)
        actives = getattr(blackboard, "_active_children", None)
        prev_idx = actives.pop(id(self), None) if isinstance(actives, dict) else None
        if prev_idx is not None and 0 <= prev_idx < len(self.children):
            self.children[prev_idx].halt(blackboard)
        else:
            for c in self.children:
                c.halt(blackboard)

    @classmethod
    def from_modules(
        cls,
        *mods: Any,
        extra: list[BehaviorNode] | None = None,
        name: str = "root",
        hysteresis: int = 0,
    ) -> PrioritySelector:
        """Collects `@bt_leaf`-decorated functions and BehaviorNodes across modules in priority/topo order."""
        nodes: list[BehaviorNode] = []
        seen_names: set[str] = set()
        for mod in mods:
            if mod is None:
                continue
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                val = getattr(mod, attr_name, None)
                if val is None:
                    continue
                if isinstance(val, BehaviorNode):
                    if val.name not in seen_names:
                        seen_names.add(val.name)
                        nodes.append(val)
                elif callable(val) and hasattr(val, "_lox_bt_meta"):
                    meta = getattr(val, "_lox_bt_meta", {})
                    nm = str(meta.get("name") or attr_name)
                    if nm not in seen_names:
                        seen_names.add(nm)
                        nodes.append(ActionLeaf(val, name=nm))
        if extra:
            for node in extra:
                if node.name not in seen_names:
                    seen_names.add(node.name)
                    nodes.append(node)

        by_name: dict[str, int] = {n.name: idx for idx, n in enumerate(nodes)}
        adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
        indeg: dict[int, int] = {i: 0 for i in range(len(nodes))}
        for idx, node in enumerate(nodes):
            before = getattr(node, "before", None)
            after = getattr(node, "after", None)
            if before and before in by_name and by_name[before] != idx:
                tgt = by_name[before]
                adj[idx].append(tgt)
                indeg[tgt] += 1
            if after and after in by_name and by_name[after] != idx:
                src = by_name[after]
                adj[src].append(idx)
                indeg[idx] += 1

        # Kahn's topological sort breaking ties by higher priority descending, then original order
        ordered_indices: list[int] = []
        ready = sorted(
            [i for i, d in indeg.items() if d == 0],
            key=lambda i: (-float(getattr(nodes[i], "priority", 0.0)), i),
        )
        while ready:
            cur = ready.pop(0)
            ordered_indices.append(cur)
            for nxt in adj[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
            ready.sort(key=lambda i: (-float(getattr(nodes[i], "priority", 0.0)), i))

        if len(ordered_indices) != len(nodes):
            from lox.lang.validator import PolicyValidationError

            raise PolicyValidationError(
                "Circular `before`/`after` anchor dependency detected in @bt_leaf PrioritySelector.from_modules"
            )

        return cls(
            [nodes[i] for i in ordered_indices],
            name=name,
            hysteresis=hysteresis,
        )

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        commits = getattr(blackboard, "_selector_commits", None)
        actives = getattr(blackboard, "_active_children", None)
        sel_id = id(self)
        prev_active = actives.get(sel_id) if isinstance(actives, dict) else None

        failed_committed_idx: int | None = None
        if isinstance(commits, dict) and sel_id in commits:
            committed_idx, rem = commits[sel_id]
            if rem > 0 and 0 <= committed_idx < len(self.children):
                committed_child = self.children[committed_idx]
                if committed_child.should_preempt(state, blackboard):
                    committed_child.halt(blackboard)
                    commits.pop(sel_id, None)
                else:
                    status, act = committed_child.tick(state, blackboard)
                    if status in (NodeStatus.SUCCESS, NodeStatus.RUNNING):
                        if rem - 1 > 0:
                            commits[sel_id] = (committed_idx, rem - 1)
                        else:
                            commits.pop(sel_id, None)
                        if isinstance(actives, dict):
                            actives[sel_id] = committed_idx
                        return status, act
                    committed_child.halt(blackboard)
                    commits.pop(sel_id, None)
                    failed_committed_idx = committed_idx
            else:
                commits.pop(sel_id, None)

        for idx, child in enumerate(self.children):
            if idx == failed_committed_idx:
                continue
            status, act = child.tick(state, blackboard)
            if status in (NodeStatus.SUCCESS, NodeStatus.RUNNING):
                if (
                    prev_active is not None
                    and prev_active != idx
                    and 0 <= prev_active < len(self.children)
                ):
                    self.children[prev_active].halt(blackboard)
                if isinstance(actives, dict):
                    actives[sel_id] = idx
                commit_len = max(
                    int(getattr(child, "min_commit", 0) or 0),
                    self.hysteresis,
                )
                if commit_len > 1 and isinstance(commits, dict):
                    commits[sel_id] = (idx, commit_len - 1)
                return status, act

        if prev_active is not None and 0 <= prev_active < len(self.children):
            self.children[prev_active].halt(blackboard)
        if isinstance(actives, dict):
            actives.pop(sel_id, None)
        return NodeStatus.FAILURE, None


class Sequence(BehaviorNode):
    """
    Composite Sequence node.
    Ticks children sequentially from left to right.
    If any child returns FAILURE, immediately returns FAILURE (short-circuit).
    If a child returns RUNNING, returns RUNNING.
    Returns SUCCESS if all children return SUCCESS.
    """

    def __init__(
        self,
        children: list[BehaviorNode] | None = None,
        name: str = "",
        min_commit: int = 0,
    ) -> None:
        super().__init__(name=name, min_commit=min_commit)
        self.children: list[BehaviorNode] = children or []

    def add_child(self, child: BehaviorNode) -> Sequence:
        self.children.append(child)
        return self

    def halt(self, blackboard: Blackboard) -> None:
        for c in self.children:
            c.halt(blackboard)

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        last_action = None
        for child in self.children:
            status, act = child.tick(state, blackboard)
            if act is not None:
                last_action = act
            if status == NodeStatus.FAILURE:
                return NodeStatus.FAILURE, None
            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING, act
        return NodeStatus.SUCCESS, last_action


class MemorySequence(BehaviorNode):
    """
    Stateful Sequence node (`Sequence with memory`).
    Remembers the active child index when a child returns `RUNNING` and resumes
    directly from that index on subsequent ticks until completion, failure, or `halt()`.
    """

    def __init__(
        self,
        children: list[BehaviorNode] | None = None,
        name: str = "",
        min_commit: int = 0,
    ) -> None:
        super().__init__(name=name, min_commit=min_commit)
        self.children: list[BehaviorNode] = children or []

    def add_child(self, child: BehaviorNode) -> MemorySequence:
        self.children.append(child)
        return self

    def halt(self, blackboard: Blackboard) -> None:
        seq_indices = getattr(blackboard, "_seq_indices", None)
        idx = seq_indices.pop(id(self), None) if isinstance(seq_indices, dict) else None
        if idx is not None and 0 <= idx < len(self.children):
            self.children[idx].halt(blackboard)
        else:
            for c in self.children:
                c.halt(blackboard)

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        seq_indices = getattr(blackboard, "_seq_indices", None)
        seq_id = id(self)
        start_idx = seq_indices.get(seq_id, 0) if isinstance(seq_indices, dict) else 0
        if start_idx >= len(self.children):
            start_idx = 0

        last_action = None
        for i in range(start_idx, len(self.children)):
            child = self.children[i]
            status, act = child.tick(state, blackboard)
            if act is not None:
                last_action = act
            if status == NodeStatus.FAILURE:
                child.halt(blackboard)
                if isinstance(seq_indices, dict):
                    seq_indices.pop(seq_id, None)
                return NodeStatus.FAILURE, None
            if status == NodeStatus.RUNNING:
                if isinstance(seq_indices, dict):
                    seq_indices[seq_id] = i
                return NodeStatus.RUNNING, act

        if isinstance(seq_indices, dict):
            seq_indices.pop(seq_id, None)
        return NodeStatus.SUCCESS, last_action


class Inverter(BehaviorNode):
    """Decorator node that inverts SUCCESS and FAILURE."""

    def __init__(self, child: BehaviorNode, name: str = "") -> None:
        super().__init__(name=name or f"Invert_{child.name}")
        self.child = child

    def halt(self, blackboard: Blackboard) -> None:
        self.child.halt(blackboard)

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        status, act = self.child.tick(state, blackboard)
        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE, None
        if status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS, act
        return status, act


class BehaviorTree:
    """
    Root coordinator for executing a Reactive Behavior Tree.
    Manages the Blackboard and ticks the root node on each game turn.
    """

    def __init__(
        self,
        root: BehaviorNode,
        name: str = "RootTree",
        blackboard: Blackboard | None = None,
    ) -> None:
        self.root = root
        self.name = name
        self.blackboard = blackboard if blackboard is not None else Blackboard()

    def tick(self, state: Any) -> Any:
        """Ticks the tree and returns the resulting action (or None if all failed)."""
        if hasattr(self.blackboard, "last_active_leaf"):
            self.blackboard.last_active_leaf = ""
        status, act = self.root.tick(state, self.blackboard)
        return act


def extract_runner_leaf_errors(runner_or_instance: Any) -> list[tuple[str, str]]:
    """Extracts any swallowed `(node_name, error_str)` entries from a PolicyRunner or Policy instance's Blackboard."""
    if runner_or_instance is None:
        return []
    inst = getattr(runner_or_instance, "instance", runner_or_instance)
    candidates = [
        getattr(inst, "blackboard", None),
        getattr(inst, "bb", None),
        getattr(getattr(inst, "tree", None), "blackboard", None),
        getattr(runner_or_instance, "blackboard", None),
    ]
    seen: list[tuple[str, str]] = []
    for bb in candidates:
        errs = getattr(bb, "leaf_errors", None)
        if isinstance(errs, list):
            for item in errs:
                if isinstance(item, tuple) and len(item) == 2:
                    pair = (str(item[0]), str(item[1]))
                    if pair not in seen:
                        seen.append(pair)
    return seen
