"""
Assignment Q4: Bayesian Networks
Implements:
  - BayesNet class: DAG with CPTs (Conditional Probability Tables)
  - Exact Inference: Enumeration (full joint) and Variable Elimination
  - Approximate Inference: Prior Sampling, Rejection Sampling, Likelihood Weighting
Domain: Classic Alarm Network (Burglary, Earthquake, Alarm, JohnCalls, MaryCalls)
"""

import random
import math
from typing import Dict, List, Optional, Tuple
from functools import reduce


# ──────────────────────────────────────────────
# BAYESIAN NETWORK NODE
# ──────────────────────────────────────────────

class BayesNode:
    """
    A single node in a Bayesian Network.
    Stores the Conditional Probability Table (CPT).
    CPT is stored as dict: {parent_assignment_tuple: P(True | parents)}
    For a root node with no parents: {(): P(True)}
    """

    def __init__(self, name: str, parents: List[str], cpt: Dict):
        self.name = name
        self.parents = parents
        self.cpt = cpt  # {(T/F, T/F, ...): prob_of_true}

    def p(self, value: bool, parent_values: Dict[str, bool]) -> float:
        """Return P(self=value | parent_values)"""
        key = tuple(parent_values[p] for p in self.parents)
        prob_true = self.cpt[key]
        return prob_true if value else (1.0 - prob_true)


# ──────────────────────────────────────────────
# BAYESIAN NETWORK
# ──────────────────────────────────────────────

class BayesNet:
    """
    A Bayesian Network: DAG of BayesNodes.
    Topological order is maintained for correct sampling/inference.
    """

    def __init__(self):
        self.nodes: Dict[str, BayesNode] = {}
        self._order: List[str] = []  # topological order

    def add_node(self, name: str, parents: List[str], cpt: Dict):
        """Add node in topological order (parents must be added first)."""
        self.nodes[name] = BayesNode(name, parents, cpt)
        self._order.append(name)

    def variables(self) -> List[str]:
        return self._order

    def joint_probability(self, assignment: Dict[str, bool]) -> float:
        """P(X₁=x₁,...,Xₙ=xₙ) = ∏ P(Xᵢ | Parents(Xᵢ))"""
        prob = 1.0
        for var in self._order:
            node = self.nodes[var]
            parent_vals = {p: assignment[p] for p in node.parents}
            prob *= node.p(assignment[var], parent_vals)
        return prob


# ──────────────────────────────────────────────
# INFERENCE 1: Enumeration (Exact)
# ──────────────────────────────────────────────

class EnumerationInference:
    """
    Exact inference by full enumeration over the joint distribution.
    P(query | evidence) = α * Σ P(query, evidence, hidden)
    Exponential in number of hidden variables.
    """

    def __init__(self, bn: BayesNet):
        self.bn = bn

    def query(self, query_var: str, evidence: Dict[str, bool]) -> Dict[bool, float]:
        """Return {True: P(query=T|evidence), False: P(query=F|evidence)}"""
        dist = {}
        for val in [True, False]:
            assignment = dict(evidence)
            assignment[query_var] = val
            dist[val] = self._enumerate_all(self.bn.variables(), assignment)

        # Normalise
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}

    def _enumerate_all(self, variables: List[str], evidence: Dict[str, bool]) -> float:
        if not variables:
            return 1.0
        Y = variables[0]
        rest = variables[1:]
        node = self.bn.nodes[Y]

        if Y in evidence:
            parent_vals = {p: evidence[p] for p in node.parents}
            return node.p(evidence[Y], parent_vals) * self._enumerate_all(rest, evidence)
        else:
            total = 0.0
            for val in [True, False]:
                e2 = dict(evidence)
                e2[Y] = val
                parent_vals = {p: e2[p] for p in node.parents}
                total += node.p(val, parent_vals) * self._enumerate_all(rest, e2)
            return total


# ──────────────────────────────────────────────
# INFERENCE 2: Variable Elimination (Exact)
# ──────────────────────────────────────────────

class Factor:
    """A factor (function over a set of variables) used in Variable Elimination."""

    def __init__(self, variables: List[str], values: Dict):
        self.variables = variables
        self.values = values  # {assignment_tuple: probability}

    @staticmethod
    def _all_assignments(variables):
        if not variables:
            yield {}
            return
        for rest in Factor._all_assignments(variables[1:]):
            for val in [True, False]:
                yield {variables[0]: val, **rest}

    def restrict(self, var: str, value: bool) -> 'Factor':
        """Fix var=value, remove var from factor."""
        new_vars = [v for v in self.variables if v != v == var or v != var]
        new_vars = [v for v in self.variables if v != var]
        new_vals = {}
        for assign, prob in self.values.items():
            assignment = dict(zip(self.variables, assign))
            if assignment[var] == value:
                key = tuple(assignment[v] for v in new_vars)
                new_vals[key] = prob
        return Factor(new_vars, new_vals)

    def multiply(self, other: 'Factor') -> 'Factor':
        """Pointwise multiply two factors."""
        shared = [v for v in self.variables if v in other.variables]
        all_vars = self.variables + [v for v in other.variables if v not in self.variables]
        new_vals = {}
        for assign in Factor._all_assignments(all_vars):
            k1 = tuple(assign[v] for v in self.variables)
            k2 = tuple(assign[v] for v in other.variables)
            if k1 in self.values and k2 in other.values:
                key = tuple(assign[v] for v in all_vars)
                new_vals[key] = self.values[k1] * other.values[k2]
        return Factor(all_vars, new_vals)

    def sum_out(self, var: str) -> 'Factor':
        """Marginalise out var by summing over its values."""
        new_vars = [v for v in self.variables if v != var]
        new_vals = {}
        for assign, prob in self.values.items():
            assignment = dict(zip(self.variables, assign))
            key = tuple(assignment[v] for v in new_vars)
            new_vals[key] = new_vals.get(key, 0.0) + prob
        return Factor(new_vars, new_vals)

    def normalize(self) -> 'Factor':
        total = sum(self.values.values())
        return Factor(self.variables, {k: v / total for k, v in self.values.items()})


class VariableEliminationInference:
    """Variable Elimination: more efficient than enumeration for sparse networks."""

    def __init__(self, bn: BayesNet):
        self.bn = bn

    def _make_factor(self, var: str) -> Factor:
        node = self.bn.nodes[var]
        all_vars = node.parents + [var]
        vals = {}
        for assign in Factor._all_assignments(all_vars):
            parent_vals = {p: assign[p] for p in node.parents}
            for v in [True, False]:
                key = tuple(assign[p] for p in node.parents) + (v,)
                vals[key] = node.p(v, parent_vals)
            break

        # Rebuild properly
        vals = {}
        for assign in Factor._all_assignments(all_vars):
            key = tuple(assign[v] for v in all_vars)
            parent_vals = {p: assign[p] for p in node.parents}
            vals[key] = node.p(assign[var], parent_vals)
        return Factor(all_vars, vals)

    def query(self, query_var: str, evidence: Dict[str, bool]) -> Dict[bool, float]:
        # Build factors
        factors = [self._make_factor(var) for var in self.bn.variables()]

        # Restrict evidence
        for var, val in evidence.items():
            factors = [f.restrict(var, val) if var in f.variables else f for f in factors]

        # Eliminate hidden variables
        hidden = [v for v in self.bn.variables() if v != query_var and v not in evidence]
        for var in hidden:
            relevant = [f for f in factors if var in f.variables]
            others = [f for f in factors if var not in f.variables]
            if relevant:
                product = relevant[0]
                for f in relevant[1:]:
                    product = product.multiply(f)
                marginalised = product.sum_out(var)
                factors = others + [marginalised]

        # Multiply remaining factors
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)

        result = result.normalize()
        out = {}
        for assign, prob in result.values.items():
            assignment = dict(zip(result.variables, assign))
            out[assignment[query_var]] = prob
        return out


# ──────────────────────────────────────────────
# INFERENCE 3: Approximate – Prior Sampling
# ──────────────────────────────────────────────

class ApproximateInference:
    """
    Three approximate inference methods:
    1. Prior Sampling
    2. Rejection Sampling
    3. Likelihood Weighting
    """

    def __init__(self, bn: BayesNet):
        self.bn = bn

    def _sample_prior(self) -> Dict[str, bool]:
        """Sample from prior distribution P(X₁,...,Xₙ)"""
        sample = {}
        for var in self.bn.variables():
            node = self.bn.nodes[var]
            parent_vals = {p: sample[p] for p in node.parents}
            p_true = node.cpt[tuple(parent_vals[p] for p in node.parents)]
            sample[var] = random.random() < p_true
        return sample

    def prior_sampling(self, query_var: str, evidence: Dict[str, bool], n: int = 10000) -> Dict[bool, float]:
        """P(query|evidence) estimated by fraction of samples consistent with evidence."""
        counts = {True: 0, False: 0}
        for _ in range(n):
            sample = self._sample_prior()
            if all(sample[e] == v for e, v in evidence.items()):
                counts[sample[query_var]] += 1
        total = sum(counts.values())
        if total == 0:
            return {True: 0.5, False: 0.5}
        return {k: v / total for k, v in counts.items()}

    def rejection_sampling(self, query_var: str, evidence: Dict[str, bool], n: int = 10000) -> Dict[bool, float]:
        """Reject samples inconsistent with evidence. Identical to prior sampling in effect."""
        return self.prior_sampling(query_var, evidence, n)

    def likelihood_weighting(self, query_var: str, evidence: Dict[str, bool], n: int = 10000) -> Dict[bool, float]:
        """
        Likelihood Weighting: fix evidence variables, weight each sample by
        P(evidence|sampled_non_evidence). More efficient than rejection sampling.
        """
        weights = {True: 0.0, False: 0.0}
        for _ in range(n):
            sample = {}
            weight = 1.0
            for var in self.bn.variables():
                node = self.bn.nodes[var]
                parent_vals = {p: sample[p] for p in node.parents}
                p_true = node.cpt[tuple(parent_vals[p] for p in node.parents)]
                if var in evidence:
                    sample[var] = evidence[var]
                    weight *= p_true if evidence[var] else (1.0 - p_true)
                else:
                    sample[var] = random.random() < p_true
            weights[sample[query_var]] += weight
        total = sum(weights.values())
        if total == 0:
            return {True: 0.5, False: 0.5}
        return {k: v / total for k, v in weights.items()}


# ──────────────────────────────────────────────
# CLASSIC ALARM NETWORK
# ──────────────────────────────────────────────

def build_alarm_network() -> BayesNet:
    """
    Classic Alarm Bayesian Network (Russell & Norvig).
    Nodes: Burglary, Earthquake, Alarm, JohnCalls, MaryCalls
    """
    bn = BayesNet()

    # P(Burglary=T) = 0.001
    bn.add_node("Burglary", [], {(): 0.001})

    # P(Earthquake=T) = 0.002
    bn.add_node("Earthquake", [], {(): 0.002})

    # P(Alarm | Burglary, Earthquake)
    bn.add_node("Alarm", ["Burglary", "Earthquake"], {
        (True,  True):  0.95,
        (True,  False): 0.94,
        (False, True):  0.29,
        (False, False): 0.001,
    })

    # P(JohnCalls | Alarm)
    bn.add_node("JohnCalls", ["Alarm"], {
        (True,):  0.90,
        (False,): 0.05,
    })

    # P(MaryCalls | Alarm)
    bn.add_node("MaryCalls", ["Alarm"], {
        (True,):  0.70,
        (False,): 0.01,
    })

    return bn


# ──────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────

if __name__ == "__main__":
    bn = build_alarm_network()

    print("=" * 60)
    print("  BAYESIAN NETWORK: Classic Alarm Network")
    print("=" * 60)
    print(f"  Variables: {bn.variables()}")

    evidence = {"JohnCalls": True, "MaryCalls": True}
    query = "Burglary"

    print(f"\n  Query: P({query} | JohnCalls=T, MaryCalls=T)\n")

    # Exact: Enumeration
    ei = EnumerationInference(bn)
    r1 = ei.query(query, evidence)
    print(f"  Enumeration (Exact)        : P(T)={r1[True]:.5f}  P(F)={r1[False]:.5f}")

    # Exact: Variable Elimination
    ve = VariableEliminationInference(bn)
    r2 = ve.query(query, evidence)
    print(f"  Variable Elimination (Exact): P(T)={r2[True]:.5f}  P(F)={r2[False]:.5f}")

    # Approximate
    ai = ApproximateInference(bn)
    random.seed(42)

    r3 = ai.prior_sampling(query, evidence, n=50000)
    print(f"  Prior/Rejection Sampling   : P(T)={r3[True]:.5f}  P(F)={r3[False]:.5f}")

    r4 = ai.likelihood_weighting(query, evidence, n=50000)
    print(f"  Likelihood Weighting       : P(T)={r4[True]:.5f}  P(F)={r4[False]:.5f}")

    print(f"\n  Expected (textbook): P(Burglary=T|J=T,M=T) ≈ 0.284")

    # Second query
    print(f"\n  Query: P(Alarm | Burglary=T)\n")
    r5 = ei.query("Alarm", {"Burglary": True})
    print(f"  Enumeration: P(Alarm=T|Burglary=T) = {r5[True]:.5f}")
