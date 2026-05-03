"""Test Suite for Q4: Bayesian Networks"""
import sys, random
sys.path.insert(0, '.')
from bayesian_network import (BayesNet, BayesNode, EnumerationInference,
                               VariableEliminationInference, ApproximateInference,
                               build_alarm_network)

PASS = "✓ PASS"
FAIL = "✗ FAIL"
EPS = 0.01  # tolerance for exact inference
APPROX_EPS = 0.05  # tolerance for approximate inference


def test_node_probability_root():
    bn = build_alarm_network()
    node = bn.nodes["Burglary"]
    assert abs(node.p(True, {}) - 0.001) < 1e-9
    assert abs(node.p(False, {}) - 0.999) < 1e-9
    print(f"{PASS} node_probability_root (Burglary)")


def test_node_probability_conditional():
    bn = build_alarm_network()
    node = bn.nodes["Alarm"]
    p = node.p(True, {"Burglary": True, "Earthquake": False})
    assert abs(p - 0.94) < 1e-9, f"{FAIL} expected 0.94, got {p}"
    print(f"{PASS} node_probability_conditional (Alarm|B=T,E=F)")


def test_joint_probability_sums_to_one():
    """Sum over all 2^5 = 32 joint assignments must equal 1."""
    bn = build_alarm_network()
    total = 0.0
    variables = bn.variables()
    for bits in range(2**5):
        assignment = {variables[i]: bool((bits >> i) & 1) for i in range(5)}
        total += bn.joint_probability(assignment)
    assert abs(total - 1.0) < 1e-6, f"{FAIL} joint distribution sums to {total}"
    print(f"{PASS} joint_probability_sums_to_one (total={total:.8f})")


def test_enumeration_normalises():
    bn = build_alarm_network()
    ei = EnumerationInference(bn)
    result = ei.query("Burglary", {"JohnCalls": True, "MaryCalls": True})
    assert abs(sum(result.values()) - 1.0) < EPS, f"{FAIL} posterior doesn't sum to 1"
    print(f"{PASS} enumeration_normalises")


def test_enumeration_alarm_prior():
    """P(Alarm) should be approximately 0.00252 (prior, no evidence)."""
    bn = build_alarm_network()
    ei = EnumerationInference(bn)
    result = ei.query("Alarm", {})
    expected = 0.00252
    assert abs(result[True] - expected) < 0.001, f"{FAIL} P(Alarm)={result[True]:.5f}, expected ~{expected}"
    print(f"{PASS} enumeration_alarm_prior (P(Alarm)={result[True]:.5f})")


def test_enumeration_john_calls_prior():
    """P(JohnCalls=T) ≈ 0.052"""
    bn = build_alarm_network()
    ei = EnumerationInference(bn)
    result = ei.query("JohnCalls", {})
    assert abs(result[True] - 0.0521) < 0.002
    print(f"{PASS} enumeration_john_calls_prior (P(J)={result[True]:.4f})")


def test_enumeration_burglary_given_calls():
    """P(Burglary|J=T,M=T) ≈ 0.284"""
    bn = build_alarm_network()
    ei = EnumerationInference(bn)
    result = ei.query("Burglary", {"JohnCalls": True, "MaryCalls": True})
    assert abs(result[True] - 0.284) < 0.02, f"{FAIL} got {result[True]:.4f}"
    print(f"{PASS} enumeration_burglary_given_calls (P(B|J,M)={result[True]:.4f})")


def test_variable_elimination_matches_enumeration():
    bn = build_alarm_network()
    ei = EnumerationInference(bn)
    ve = VariableEliminationInference(bn)
    evidence = {"JohnCalls": True, "MaryCalls": True}
    r1 = ei.query("Burglary", evidence)
    r2 = ve.query("Burglary", evidence)
    assert abs(r1[True] - r2[True]) < EPS, \
        f"{FAIL} enumeration={r1[True]:.4f} vs VE={r2[True]:.4f}"
    print(f"{PASS} ve_matches_enumeration (VE={r2[True]:.4f})")


def test_likelihood_weighting_approximates():
    """Likelihood weighting P(B|J=T,M=T) should be close to exact 0.284"""
    bn = build_alarm_network()
    ai = ApproximateInference(bn)
    random.seed(42)
    result = ai.likelihood_weighting("Burglary", {"JohnCalls": True, "MaryCalls": True}, n=50000)
    assert abs(result[True] - 0.284) < APPROX_EPS, \
        f"{FAIL} LW={result[True]:.4f}, expected ~0.284"
    print(f"{PASS} likelihood_weighting_approximates (P(B|J,M)={result[True]:.4f})")


def test_prior_sampling_alarm_marginal():
    """Prior sampling P(Alarm) ≈ 0.00252"""
    bn = build_alarm_network()
    ai = ApproximateInference(bn)
    random.seed(0)
    result = ai.prior_sampling("Alarm", {}, n=100000)
    assert abs(result[True] - 0.00252) < 0.003, \
        f"{FAIL} Prior sampling P(Alarm)={result[True]:.5f}"
    print(f"{PASS} prior_sampling_alarm_marginal (P(Alarm)={result[True]:.5f})")


def test_custom_network():
    """Simple 2-node network: Rain → WetGrass"""
    bn = BayesNet()
    bn.add_node("Rain", [], {(): 0.3})
    bn.add_node("WetGrass", ["Rain"], {(True,): 0.9, (False,): 0.2})
    ei = EnumerationInference(bn)
    # P(WetGrass=T) = P(W|R=T)*P(R) + P(W|R=F)*P(¬R) = 0.9*0.3 + 0.2*0.7 = 0.27+0.14=0.41
    result = ei.query("WetGrass", {})
    assert abs(result[True] - 0.41) < EPS, f"{FAIL} P(WetGrass)={result[True]:.4f}, expected 0.41"
    print(f"{PASS} custom_rain_wetgrass_network (P(Wet)={result[True]:.4f})")


if __name__ == "__main__":
    print("=" * 55)
    print("  Q4 Bayesian Networks – Test Suite")
    print("=" * 55)
    test_node_probability_root()
    test_node_probability_conditional()
    test_joint_probability_sums_to_one()
    test_enumeration_normalises()
    test_enumeration_alarm_prior()
    test_enumeration_john_calls_prior()
    test_enumeration_burglary_given_calls()
    test_variable_elimination_matches_enumeration()
    test_likelihood_weighting_approximates()
    test_prior_sampling_alarm_marginal()
    test_custom_network()
    print("\nAll tests passed!")
