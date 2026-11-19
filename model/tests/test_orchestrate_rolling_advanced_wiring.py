"""Structural consistency of the rolling worker's args tuple.

The tuple crosses a pickle boundary into a spawned process, is built at one
site and unpacked at another, and is documented in a third place. A mismatch
in arity or order does not fail at the construction site -- it fails inside a
worker, hours into a run, with a message about the wrong field. So the three
are compared programmatically rather than by eye.

Also checks that both rolling coordination types route to the same bulk
runner, and that the diagnostics CSV covers every RollingDiagnostics field
worth writing.
"""
import ast
import csv
import dataclasses
import inspect
import io
from pathlib import Path

import pytest

import model.orchestrate as orch
from model.components.rolling_optimisation import RollingDiagnostics
from model.core.scenario import ROLLING_COORDINATION_TYPES

WORKER = '_rolling_perfect_foresight_site_worker'
BULK = '_run_rolling_perfect_foresight_lp'


def _source_tree(name):
    return ast.parse(inspect.getsource(getattr(orch, name)))


def _unpack_targets(fn_name):
    """Names the worker unpacks its args tuple into, in order."""
    for node in ast.walk(_source_tree(fn_name)):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Tuple)
                and isinstance(node.value, ast.Name)
                and node.value.id == 'args'):
            return [e.id for e in node.targets[0].elts]
    raise AssertionError(f"no `(...) = args` unpack found in {fn_name}")


def _appended_tuple_lengths(fn_name, list_name):
    """Lengths of every tuple appended to `list_name` in the function."""
    out = []
    for node in ast.walk(_source_tree(fn_name)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'append'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == list_name
                and node.args
                and isinstance(node.args[0], ast.Tuple)):
            out.append(len(node.args[0].elts))
    return out


def test_worker_unpack_matches_every_construction_site():
    names = _unpack_targets(WORKER)
    built = _appended_tuple_lengths(BULK, 'site_args')
    assert built, "no site_args.append((...)) found; the check is vacuous"
    for length in built:
        assert length == len(names), (
            f"{BULK} builds a {length}-element args tuple but {WORKER} "
            f"unpacks {len(names)}: {names}"
        )


def _appended_tuple_element_names(fn_name, list_name):
    """Resolved name of each element of the appended tuple, in order.

    `site_id` resolves to 'site_id', `ctx.network_tariff` to
    'network_tariff'. Elements that resolve to nothing comparable (a call, a
    subscript) come back as None and are skipped.
    """
    for node in ast.walk(_source_tree(fn_name)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'append'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == list_name
                and node.args
                and isinstance(node.args[0], ast.Tuple)):
            out = []
            for e in node.args[0].elts:
                if isinstance(e, ast.Name):
                    out.append(e.id)
                elif isinstance(e, ast.Attribute):
                    out.append(e.attr)
                else:
                    out.append(None)
            return out
    raise AssertionError(f"no {list_name}.append((...)) found in {fn_name}")


def test_worker_args_are_in_the_same_ORDER_at_both_ends():
    """Arity alone is not enough.

    Two fields swapped at the construction site keeps the tuple the same
    length and unpacks silently, so `with_monthly_bills` becomes a
    coordination-type string and `coordination_type` becomes a bool. Nothing
    raises; the advanced branch simply never runs. This compares position by
    position for every element whose name is comparable at both ends.
    """
    names = _unpack_targets(WORKER)
    built = _appended_tuple_element_names(BULK, 'site_args')
    assert len(built) == len(names)
    compared = 0
    for i, (b, n) in enumerate(zip(built, names)):
        if b is None or b not in names:
            continue          # aliased on the way in, e.g. ctx.prices
        compared += 1
        assert b == n, (
            f"position {i}: construction passes {b!r} where the worker "
            f"unpacks {n!r}. Full order:\n  built:  {built}\n  unpack: {names}"
        )
    assert compared >= len(names) - 2, (
        f"only {compared} of {len(names)} positions were comparable; the "
        f"check has gone vacuous"
    )


def test_worker_docstring_lists_the_same_fields():
    """The docstring is what a reader trusts; drift there is silent."""
    names = _unpack_targets(WORKER)
    doc = getattr(orch, WORKER).__doc__
    body = doc.split('args: (', 1)[1].split(')', 1)[0]
    documented = [t.strip() for t in body.replace('\n', ' ').split(',')]
    assert documented == names, (
        f"docstring lists {documented}\nunpack uses  {names}"
    )


def test_later_fields_were_appended():
    """coordination_type and then apply_export_order_nudge were appended
    rather than inserted, so every existing position is unchanged. A field
    inserted mid-tuple would silently reassign the ones after it."""
    names = _unpack_targets(WORKER)
    assert names[-2:] == ['coordination_type', 'apply_export_order_nudge']


def test_both_rolling_types_route_to_one_bulk_runner():
    src = inspect.getsource(orch.run_scenario)
    tree = ast.parse(src)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id.startswith('_run_rolling')
    }
    # Two rolling runners now: the perfect-foresight pair share BULK, and the
    # VPP pair share their own. This used to assert exactly one, which held
    # while BULK was the only rolling runner; the property is which types
    # reach which runner, not how many runners exist.
    assert called == {BULK, '_run_rolling_vpp_event'}, (
        f"unexpected rolling runner set: {sorted(called)}"
    )
    assert 'ROLLING_COORDINATION_TYPES' in src, (
        "run_scenario should route on the shared constant rather than "
        "listing the coordination types again"
    )
    # Both perfect-foresight rolling types are handled by that one runner.
    # This used to assert the set had exactly two members, which held while
    # the set and the runner's scope were the same thing. Part C added two
    # rolling VPP types: they belong to the set (they need a RollingConfig and
    # the shared causality checks) but NOT to this runner, so the property to
    # assert is membership, not cardinality.
    for ct in ('rolling_perfect_foresight_lp',
               'rolling_perfect_foresight_lp_advanced'):
        assert ct in ROLLING_COORDINATION_TYPES


def test_rolling_vpp_types_are_intercepted_before_the_bulk_runner():
    """The rolling VPP types are in ROLLING_COORDINATION_TYPES, which is the
    set run_scenario routes on, so without an earlier branch they would fall
    into the wholesale/advanced bulk runner and produce a plausible wrong
    answer rather than an error. Asserted by source position because that is
    the property that matters: the guard must come FIRST."""
    src = inspect.getsource(orch.run_scenario)
    tree = ast.parse(src)

    guard_line = bulk_line = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if guard_line is None and 'rolling_vpp_sc_event' in test_src:
            guard_line = node.lineno
        if bulk_line is None and 'ROLLING_COORDINATION_TYPES' in test_src:
            bulk_line = node.lineno

    assert guard_line is not None, (
        "no branch in run_scenario tests for 'rolling_vpp_sc_event'; the "
        "rolling VPP types would route to " + BULK
    )
    assert bulk_line is not None, "no ROLLING_COORDINATION_TYPES branch found"
    assert guard_line < bulk_line, (
        f"the rolling VPP branch is at line {guard_line} but the "
        f"{BULK} branch is at line {bulk_line}; the guard must precede it"
    )


@pytest.mark.parametrize('ct', ['rolling_vpp_sc_event',
                                'rolling_vpp_tou_event'])
def test_rolling_vpp_types_reach_their_own_runner(ct):
    """Both VPP types dispatch to _run_rolling_vpp_event, not to the
    wholesale/advanced bulk runner. Replaces an earlier test that asserted a
    NotImplementedError placeholder; it was written to fail when the real
    runner landed rather than to keep passing against a stale contract."""
    src = inspect.getsource(orch.run_scenario)
    assert ct in src, f"{ct} is not mentioned in run_scenario"
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert '_run_rolling_vpp_event' in called
    assert 'NotImplementedError' not in src, (
        "the Part C placeholder guard is still present"
    )


def test_rolling_vpp_args_tuple_agrees_across_its_three_definitions():
    """The pickle boundary. The tuple is built in one place, unpacked in
    another and named in a third; arity alone would not catch two swapped
    fields, which keeps the length identical, unpacks silently, and sends the
    wrong value to the wrong parameter."""
    fields = orch._ROLLING_VPP_ARG_FIELDS

    # 1. The unpacker: the tuple-assignment target inside the worker.
    wsrc = inspect.getsource(orch._rolling_vpp_site_worker)
    wtree = ast.parse(wsrc)
    unpacked = None
    for node in ast.walk(wtree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Tuple)
                and any(isinstance(e, ast.Name) and e.id == 'site_id'
                        for e in node.targets[0].elts)):
            unpacked = tuple(e.id for e in node.targets[0].elts)
            break
    assert unpacked is not None, "no args unpacking found in the worker"
    assert unpacked == fields, (
        f"worker unpacks {len(unpacked)} fields in a different order:\n"
        f"  declared: {fields}\n  unpacked: {unpacked}"
    )

    # 2. The builder: the append inside the scenario runner.
    rsrc = inspect.getsource(orch._run_rolling_vpp_event)
    rtree = ast.parse(rsrc)
    built = None
    for node in ast.walk(rtree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'append'
                and node.args and isinstance(node.args[0], ast.Tuple)):
            built = node.args[0].elts
            break
    assert built is not None, "no site_args.append((...)) found in the runner"
    assert len(built) == len(fields), (
        f"runner builds {len(built)} fields, {len(fields)} declared"
    )

    # 3. The docstring names the shared constant rather than re-listing.
    assert '_ROLLING_VPP_ARG_FIELDS' in wsrc, (
        "the worker docstring should point at the shared field list, not "
        "restate it -- a second copy is a third place to drift"
    )


def test_rolling_vpp_worker_is_separate_from_the_perfect_foresight_one():
    """They carry different payloads and return different shapes. Asserted so
    a later merge is a deliberate decision rather than a quiet one."""
    assert orch._rolling_vpp_site_worker is not \
        orch._rolling_perfect_foresight_site_worker
    vpp_ret = inspect.getsource(orch._rolling_vpp_site_worker)
    assert 'roll_vt' in vpp_ret, "the VPP worker must return VppTraces"


def test_the_vpp_runner_always_builds_a_price_provider():
    """The advanced perfect-foresight path skips the provider for a plan with
    no critical rates. The VPP path must not: the operator declares events from
    the price forecast whatever the household plan reads."""
    src = inspect.getsource(orch._run_rolling_vpp_event)
    assert 'needs_prices' not in src, (
        "the VPP runner appears to have a conditional provider; it must "
        "always build one"
    )
    assert '_init_rolling_worker(*init_args)' in src
