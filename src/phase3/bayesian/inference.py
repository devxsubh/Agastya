"""Inference helpers for Phase 3 BN."""

from __future__ import annotations

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

VALID_STATES: dict[str, tuple[str, ...]] = {
    "Has_Payment_Clause": ("Present", "Absent"),
    "Has_Termination_Clause": ("Present", "Absent"),
    "Has_Liability_Clause": ("Present", "Absent"),
    "Has_Confidentiality_Clause": ("Present", "Absent"),
    "Has_Dispute_Resolution_Clause": ("Present", "Absent"),
    "Payment_Or_Termination_Risky": ("Risky", "Not_Risky"),
    "Liability_Or_Confidentiality_Risky": ("Risky", "Not_Risky"),
    "Cross_Clause_Conflict": ("Conflict", "No_Conflict"),
}


def validate_evidence(evidence: dict[str, str] | None) -> None:
    for node, val in (evidence or {}).items():
        allowed = VALID_STATES.get(node)
        if allowed and val not in allowed:
            raise ValueError(
                f"State mismatch: node '{node}' got '{val}', expected one of {set(allowed)}"
            )


def build_virtual_evidence(
    virtual_ev_dict: dict[str, list[float]] | None,
) -> list[TabularCPD]:
    """Convert {node: [p_negative, p_positive]} to pgmpy virtual-evidence CPDs."""
    if not virtual_ev_dict:
        return []

    ve_list: list[TabularCPD] = []
    for node, probs in virtual_ev_dict.items():
        if probs is None or len(probs) != 2:
            continue
        states = list(VALID_STATES.get(node, ()))
        if len(states) != 2:
            continue
        p_negative = float(probs[0])
        p_positive = float(probs[1])
        # Confidence mapper emits [p_negative, p_positive]. Re-order for state order.
        if states[0] in {"Present", "Risky", "Conflict"}:
            ordered_probs = [[p_positive], [p_negative]]
        else:
            ordered_probs = [[p_negative], [p_positive]]
        ve_list.append(
            TabularCPD(
                variable=node,
                variable_card=2,
                values=ordered_probs,
                state_names={node: states},
            )
        )
    return ve_list


def run_inference(
    model: BayesianNetwork,
    evidence: dict[str, str] | None = None,
    virtual_evidence: dict[str, list[float]] | None = None,
    query_var: str = "Contract_Risk_Level",
    *,
    bp_engine: BeliefPropagation | None = None,
) -> dict:
    """Run belief propagation and return a normalized risk payload.

    ``virtual_evidence`` is auto-converted to a list of state-aware
    ``TabularCPD`` objects. Pass ``bp_engine`` to reuse a pre-built engine.
    """
    hard_evidence = {k: v for k, v in (evidence or {}).items() if v is not None}
    validate_evidence(hard_evidence)
    bp = bp_engine if bp_engine is not None else BeliefPropagation(model)
    vcpds = build_virtual_evidence(virtual_evidence)
    kwargs: dict = {"variables": [query_var], "evidence": hard_evidence}
    if vcpds:
        kwargs["virtual_evidence"] = vcpds
    try:
        result = bp.query(**kwargs)
    except TypeError:
        # Older pgmpy versions don't accept virtual_evidence; fall back gracefully.
        result = bp.query(variables=[query_var], evidence=hard_evidence)
    factor = result[query_var] if isinstance(result, dict) else result
    states = factor.state_names[query_var]
    values = factor.values
    probabilities = {state: float(values[idx]) for idx, state in enumerate(states)}
    risk_level = max(probabilities, key=probabilities.get)
    return {
        "distribution": factor,
        "risk_level": risk_level,
        "probabilities": probabilities,
    }


def query_node_posterior(
    model: BayesianNetwork,
    node: str,
    *,
    bp_engine: BeliefPropagation | None = None,
) -> dict[str, float]:
    """Marginal P(node) under no evidence — used for cached priors."""
    bp = bp_engine if bp_engine is not None else BeliefPropagation(model)
    factor = bp.query(variables=[node], evidence={})
    factor = factor[node] if isinstance(factor, dict) else factor
    return {s: float(factor.values[i]) for i, s in enumerate(factor.state_names[node])}
