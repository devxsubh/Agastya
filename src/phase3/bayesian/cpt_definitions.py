"""Seed CPT definitions for the Phase 3 Bayesian network."""

from __future__ import annotations

from pgmpy.factors.discrete import TabularCPD


def get_seed_cpts():
    """Return domain-informed seed CPDs for all nodes."""
    cpt_payment = TabularCPD(
        "Has_Payment_Clause",
        2,
        [[0.6], [0.4]],
        state_names={"Has_Payment_Clause": ["Present", "Absent"]},
    )
    cpt_termination = TabularCPD(
        "Has_Termination_Clause",
        2,
        [[0.5], [0.5]],
        state_names={"Has_Termination_Clause": ["Present", "Absent"]},
    )
    cpt_liability = TabularCPD(
        "Has_Liability_Clause",
        2,
        [[0.55], [0.45]],
        state_names={"Has_Liability_Clause": ["Present", "Absent"]},
    )
    cpt_confidentiality = TabularCPD(
        "Has_Confidentiality_Clause",
        2,
        [[0.45], [0.55]],
        state_names={"Has_Confidentiality_Clause": ["Present", "Absent"]},
    )
    cpt_dispute = TabularCPD(
        "Has_Dispute_Resolution_Clause",
        2,
        [[0.5], [0.5]],
        state_names={"Has_Dispute_Resolution_Clause": ["Present", "Absent"]},
    )
    cpt_r1 = TabularCPD(
        "Payment_Or_Termination_Risky",
        2,
        [[0.75, 0.60, 0.55, 0.15], [0.25, 0.40, 0.45, 0.85]],
        evidence=["Has_Payment_Clause", "Has_Termination_Clause"],
        evidence_card=[2, 2],
        state_names={
            "Payment_Or_Termination_Risky": ["Risky", "Not_Risky"],
            "Has_Payment_Clause": ["Present", "Absent"],
            "Has_Termination_Clause": ["Present", "Absent"],
        },
    )
    cpt_r2 = TabularCPD(
        "Liability_Or_Confidentiality_Risky",
        2,
        [[0.80, 0.65, 0.60, 0.20], [0.20, 0.35, 0.40, 0.80]],
        evidence=["Has_Liability_Clause", "Has_Confidentiality_Clause"],
        evidence_card=[2, 2],
        state_names={
            "Liability_Or_Confidentiality_Risky": ["Risky", "Not_Risky"],
            "Has_Liability_Clause": ["Present", "Absent"],
            "Has_Confidentiality_Clause": ["Present", "Absent"],
        },
    )
    cpt_d1 = TabularCPD(
        "Cross_Clause_Conflict",
        3,
        [
            [0.65, 0.75, 0.30, 0.45, 0.25, 0.40, 0.03, 0.10],  # Conflict
            [0.25, 0.18, 0.50, 0.42, 0.50, 0.42, 0.17, 0.25],  # Partial
            [0.10, 0.07, 0.20, 0.13, 0.25, 0.18, 0.80, 0.65],  # No_Conflict
        ],
        evidence=[
            "Payment_Or_Termination_Risky",
            "Liability_Or_Confidentiality_Risky",
            "Has_Dispute_Resolution_Clause",
        ],
        evidence_card=[2, 2, 2],
        state_names={
            "Cross_Clause_Conflict": ["Conflict", "Partial", "No_Conflict"],
            "Payment_Or_Termination_Risky": ["Risky", "Not_Risky"],
            "Liability_Or_Confidentiality_Risky": ["Risky", "Not_Risky"],
            "Has_Dispute_Resolution_Clause": ["Present", "Absent"],
        },
    )
    cpt_f1 = TabularCPD(
        "Contract_Risk_Level",
        3,
        [[0.05, 0.20, 0.70], [0.20, 0.65, 0.25], [0.75, 0.15, 0.05]],
        evidence=["Cross_Clause_Conflict"],
        evidence_card=[3],
        state_names={
            "Contract_Risk_Level": ["Low", "Medium", "High"],
            "Cross_Clause_Conflict": ["Conflict", "Partial", "No_Conflict"],
        },
    )
    return [
        cpt_payment,
        cpt_termination,
        cpt_liability,
        cpt_confidentiality,
        cpt_dispute,
        cpt_r1,
        cpt_r2,
        cpt_d1,
        cpt_f1,
    ]

