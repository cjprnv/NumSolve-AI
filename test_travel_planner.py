"""Test Suite for Q2: AI Travel Planner"""
import sys
sys.path.insert(0, '.')
from travel_planner import TravelPlannerKB, UserPreferences

PASS = "✓ PASS"
FAIL = "✗ FAIL"
planner = TravelPlannerKB()


def test_destination_ranking_returns_all():
    prefs = UserPreferences(budget_usd=2000, duration_days=5, interests=["cultural"], season="spring", name="Test")
    recs = planner.recommend_destinations(prefs)
    assert len(recs) == 5, f"{FAIL} should return all 5 destinations"
    print(f"{PASS} destination_ranking_returns_all ({len(recs)} destinations)")


def test_scores_between_0_and_1():
    prefs = UserPreferences(budget_usd=2000, duration_days=5, interests=["cultural", "food"], season="spring", name="Test")
    recs = planner.recommend_destinations(prefs)
    for r in recs:
        assert 0.0 <= r["score"] <= 1.0, f"{FAIL} score {r['score']} out of range"
    print(f"{PASS} scores_between_0_and_1")


def test_beach_lover_recommends_bali():
    prefs = UserPreferences(budget_usd=5000, duration_days=7, interests=["beach", "nature", "spiritual"], season="summer", name="Test")
    recs = planner.recommend_destinations(prefs)
    top = recs[0]["destination"]
    assert top == "Bali", f"{FAIL} beach lover should get Bali, got {top}"
    print(f"{PASS} beach_lover_recommends_bali (top={top})")


def test_budget_filter():
    prefs = UserPreferences(budget_usd=500, duration_days=5, interests=["cultural"], season="spring", name="Test")
    recs = planner.recommend_destinations(prefs)
    # Bali ($80/day * 5 = $400) should be feasible; NY ($250*5=$1250) should not
    bali = next(r for r in recs if r["destination"] == "Bali")
    ny = next(r for r in recs if r["destination"] == "New York")
    assert bali["budget_feasible"] is True, f"{FAIL} Bali should be budget feasible"
    assert ny["budget_feasible"] is False, f"{FAIL} NY should NOT be budget feasible at $500"
    print(f"{PASS} budget_filter_correct (Bali=feasible, NY=not feasible)")


def test_itinerary_day_count():
    prefs = UserPreferences(budget_usd=2000, duration_days=4, interests=["cultural"], season="spring", name="Test")
    itin = planner.build_itinerary("Paris", prefs)
    assert len(itin["itinerary"]) == 4, f"{FAIL} should have 4 days"
    print(f"{PASS} itinerary_day_count (4 days)")


def test_itinerary_has_required_fields():
    prefs = UserPreferences(budget_usd=2000, duration_days=3, interests=["cultural"], season="spring", name="Test")
    itin = planner.build_itinerary("Tokyo", prefs)
    for day in itin["itinerary"]:
        for key in ["day", "activity", "lunch_recommendation", "tip"]:
            assert key in day, f"{FAIL} missing key '{key}' in day {day}"
    print(f"{PASS} itinerary_has_required_fields")


def test_wine_pairing_works():
    prefs = UserPreferences(budget_usd=2000, duration_days=3, interests=["cultural"], season="spring", name="Test")
    itin = planner.build_itinerary("Rome", prefs)
    # Day 1 food in Rome is "Cacio e Pepe" which has a wine pairing
    day1 = itin["itinerary"][0]
    assert day1["wine_pairing"] is not None, f"{FAIL} Rome day 1 should have wine pairing"
    assert "wine" in day1["wine_pairing"], f"{FAIL} wine_pairing should have 'wine' key"
    print(f"{PASS} wine_pairing_works (wine={day1['wine_pairing']['wine']})")


def test_cost_assessment_accuracy():
    prefs = UserPreferences(budget_usd=2000, duration_days=5, interests=["cultural"], season="spring", name="Test")
    ca = planner.cost_assessment("Paris", prefs)
    expected = 200 * 5  # $1000
    assert ca["total_estimated_usd"] == expected, f"{FAIL} expected {expected}, got {ca['total_estimated_usd']}"
    assert ca["verdict"] == "Within budget ✓", f"{FAIL} should be within budget"
    print(f"{PASS} cost_assessment_accuracy (total=${ca['total_estimated_usd']})")


def test_over_budget_verdict():
    prefs = UserPreferences(budget_usd=500, duration_days=5, interests=["cultural"], season="spring", name="Test")
    ca = planner.cost_assessment("New York", prefs)  # $250*5=$1250 > $500
    assert ca["verdict"] == "Over budget ✗", f"{FAIL} should be over budget"
    assert ca["surplus_or_deficit_usd"] < 0, f"{FAIL} deficit should be negative"
    print(f"{PASS} over_budget_verdict (deficit=${ca['surplus_or_deficit_usd']})")


def test_full_plan_structure():
    prefs = UserPreferences(budget_usd=1500, duration_days=5, interests=["cultural", "food"], season="spring", name="Test")
    plan = planner.generate_plan(prefs)
    for key in ["recommended_destinations", "selected_destination", "itinerary", "cost_assessment"]:
        assert key in plan, f"{FAIL} missing key '{key}' in plan"
    assert len(plan["recommended_destinations"]) == 3
    print(f"{PASS} full_plan_structure (top={plan['selected_destination']})")


if __name__ == "__main__":
    print("=" * 50)
    print("  Q2 Travel Planner – Test Suite")
    print("=" * 50)
    test_destination_ranking_returns_all()
    test_scores_between_0_and_1()
    test_beach_lover_recommends_bali()
    test_budget_filter()
    test_itinerary_day_count()
    test_itinerary_has_required_fields()
    test_wine_pairing_works()
    test_cost_assessment_accuracy()
    test_over_budget_verdict()
    test_full_plan_structure()
    print("\nAll tests passed!")
