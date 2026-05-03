"""
Assignment Q2: AI Travel Planner
Reuses existing knowledge bases:
  - Wine Ontology (food/wine pairing)
  - Tourist Places KB
  - Food Recommendation KB
  - Personalised Tour Plans
  - Cost Assessment KB
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ──────────────────────────────────────────────
# KNOWLEDGE BASE 1: Tourist Places
# ──────────────────────────────────────────────

TOURIST_PLACES_KB = {
    "Paris": {
        "attractions": ["Eiffel Tower", "Louvre Museum", "Notre Dame", "Montmartre", "Versailles"],
        "type": ["romantic", "cultural", "historic"],
        "best_season": ["spring", "autumn"],
        "avg_cost_per_day_usd": 200,
        "language": "French",
        "currency": "EUR"
    },
    "Tokyo": {
        "attractions": ["Senso-ji Temple", "Shibuya Crossing", "TeamLab Planets", "Mt. Fuji Day Trip", "Tsukiji Market"],
        "type": ["cultural", "tech", "food"],
        "best_season": ["spring", "autumn"],
        "avg_cost_per_day_usd": 150,
        "language": "Japanese",
        "currency": "JPY"
    },
    "New York": {
        "attractions": ["Central Park", "Metropolitan Museum", "Times Square", "Brooklyn Bridge", "Statue of Liberty"],
        "type": ["urban", "cultural", "shopping"],
        "best_season": ["spring", "autumn"],
        "avg_cost_per_day_usd": 250,
        "language": "English",
        "currency": "USD"
    },
    "Bali": {
        "attractions": ["Tanah Lot Temple", "Ubud Rice Terraces", "Seminyak Beach", "Sacred Monkey Forest", "Kecak Dance"],
        "type": ["beach", "spiritual", "nature"],
        "best_season": ["summer", "winter"],
        "avg_cost_per_day_usd": 80,
        "language": "Balinese/Indonesian",
        "currency": "IDR"
    },
    "Rome": {
        "attractions": ["Colosseum", "Vatican Museums", "Trevi Fountain", "Roman Forum", "Piazza Navona"],
        "type": ["historic", "cultural", "romantic"],
        "best_season": ["spring", "autumn"],
        "avg_cost_per_day_usd": 180,
        "language": "Italian",
        "currency": "EUR"
    }
}


# ──────────────────────────────────────────────
# KNOWLEDGE BASE 2: Food Recommendations
# ──────────────────────────────────────────────

FOOD_KB = {
    "Paris":    ["Croissant & Café au Lait", "Crêpes Suzette", "Beef Bourguignon", "French Onion Soup", "Macarons"],
    "Tokyo":    ["Ramen", "Sushi at Tsukiji", "Wagyu Beef", "Takoyaki", "Matcha Desserts"],
    "New York": ["New York Pizza", "Bagel with Lox", "Pastrami Sandwich", "Cheesecake", "Halal Cart Food"],
    "Bali":     ["Nasi Goreng", "Babi Guling", "Satay", "Gado-Gado", "Fresh Coconut"],
    "Rome":     ["Cacio e Pepe", "Supplì", "Gelato", "Carbonara", "Tiramisu"]
}


# ──────────────────────────────────────────────
# KNOWLEDGE BASE 3: Wine Ontology (pairing)
# ──────────────────────────────────────────────

WINE_PAIRING_KB = {
    "Beef Bourguignon":   {"wine": "Pinot Noir", "region": "Burgundy", "notes": "Earthy, medium body, pairs with braised beef"},
    "Cacio e Pepe":       {"wine": "Frascati DOC", "region": "Lazio", "notes": "Crisp white, local Roman tradition"},
    "Carbonara":          {"wine": "Greco di Tufo", "region": "Campania", "notes": "Mineral white, cuts through egg richness"},
    "Sushi at Tsukiji":   {"wine": "Junmai Daiginjo Sake", "region": "Japan", "notes": "Delicate, floral – enhances umami"},
    "Wagyu Beef":         {"wine": "Cabernet Sauvignon", "region": "Napa Valley", "notes": "Bold tannins, complements rich fat"},
    "Nasi Goreng":        {"wine": "Gewürztraminer", "region": "Alsace", "notes": "Off-dry, spice-friendly"},
    "New York Pizza":     {"wine": "Chianti Classico", "region": "Tuscany", "notes": "High acidity matches tomato sauce"},
}


# ──────────────────────────────────────────────
# USER PREFERENCES MODEL
# ──────────────────────────────────────────────

@dataclass
class UserPreferences:
    budget_usd: float
    duration_days: int
    interests: List[str]         # e.g. ["cultural", "food", "beach"]
    season: str                  # "spring" | "summer" | "autumn" | "winter"
    name: str = "Traveller"
    dietary: str = "none"        # "vegetarian" | "vegan" | "none"


# ──────────────────────────────────────────────
# INFERENCE ENGINE
# ──────────────────────────────────────────────

class TravelPlannerKB:
    """
    Rule-based inference engine that reuses knowledge bases to:
    1. Match destinations to user interests
    2. Filter by budget and season
    3. Build a personalised daily itinerary
    4. Recommend local food and wine pairings
    5. Provide cost assessment
    """

    def __init__(self):
        self.places = TOURIST_PLACES_KB
        self.food = FOOD_KB
        self.wine = WINE_PAIRING_KB

    # ── Rule 1: Interest Matching ─────────────────
    def _interest_score(self, destination: str, preferences: UserPreferences) -> float:
        place_types = set(self.places[destination]["type"])
        user_interests = set(preferences.interests)
        overlap = place_types & user_interests
        return len(overlap) / max(len(user_interests), 1)

    # ── Rule 2: Budget Feasibility ────────────────
    def _budget_feasible(self, destination: str, preferences: UserPreferences) -> bool:
        cost = self.places[destination]["avg_cost_per_day_usd"] * preferences.duration_days
        return cost <= preferences.budget_usd

    # ── Rule 3: Season Compatibility ─────────────
    def _season_compatible(self, destination: str, preferences: UserPreferences) -> bool:
        return preferences.season in self.places[destination]["best_season"]

    # ── Rule 4: Score & Rank Destinations ─────────
    def recommend_destinations(self, preferences: UserPreferences) -> List[Dict]:
        ranked = []
        for dest, info in self.places.items():
            score = self._interest_score(dest, preferences)
            feasible = self._budget_feasible(dest, preferences)
            season_ok = self._season_compatible(dest, preferences)

            # Weighted score
            final_score = (
                score * 0.5 +
                (0.3 if season_ok else 0.0) +
                (0.2 if feasible else 0.0)
            )
            ranked.append({
                "destination": dest,
                "score": round(final_score, 2),
                "budget_feasible": feasible,
                "season_compatible": season_ok,
                "estimated_cost_usd": info["avg_cost_per_day_usd"] * preferences.duration_days,
                "interest_match": round(score, 2)
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    # ── Rule 5: Build Daily Itinerary ─────────────
    def build_itinerary(self, destination: str, preferences: UserPreferences) -> Dict:
        info = self.places[destination]
        foods = self.food.get(destination, [])
        attractions = info["attractions"][:]

        days = []
        for day in range(1, preferences.duration_days + 1):
            attraction = attractions[(day - 1) % len(attractions)]
            food = foods[(day - 1) % len(foods)] if foods else "Local cuisine"

            wine_rec = None
            if food in self.wine:
                wine_rec = self.wine[food]

            days.append({
                "day": day,
                "activity": attraction,
                "lunch_recommendation": food,
                "wine_pairing": wine_rec,
                "tip": self._generate_tip(destination, attraction)
            })

        return {
            "traveller": preferences.name,
            "destination": destination,
            "duration": f"{preferences.duration_days} days",
            "season": preferences.season,
            "language": info["language"],
            "currency": info["currency"],
            "itinerary": days
        }

    def _generate_tip(self, destination: str, attraction: str) -> str:
        tips = {
            "Eiffel Tower": "Book tickets online to skip the 2hr queue.",
            "Louvre Museum": "Start with Mona Lisa, arrives early to beat crowds.",
            "Senso-ji Temple": "Visit at dawn for peaceful atmosphere.",
            "Shibuya Crossing": "Best viewed from Starbucks 2nd floor window.",
            "Colosseum": "Combined ticket includes Roman Forum – great value.",
            "Tanah Lot Temple": "Best at sunset; wear modest clothing.",
            "Central Park": "Rent a bike to cover all highlights in a day.",
        }
        return tips.get(attraction, f"Enjoy {attraction} – check local opening hours.")

    # ── Rule 6: Cost Assessment ───────────────────
    def cost_assessment(self, destination: str, preferences: UserPreferences) -> Dict:
        info = self.places[destination]
        daily = info["avg_cost_per_day_usd"]
        total_accommodation = daily * 0.4 * preferences.duration_days
        total_food = daily * 0.3 * preferences.duration_days
        total_activities = daily * 0.2 * preferences.duration_days
        total_transport = daily * 0.1 * preferences.duration_days
        total = daily * preferences.duration_days

        return {
            "destination": destination,
            "duration_days": preferences.duration_days,
            "accommodation_usd": round(total_accommodation, 2),
            "food_usd": round(total_food, 2),
            "activities_usd": round(total_activities, 2),
            "local_transport_usd": round(total_transport, 2),
            "total_estimated_usd": round(total, 2),
            "budget_available_usd": preferences.budget_usd,
            "surplus_or_deficit_usd": round(preferences.budget_usd - total, 2),
            "verdict": "Within budget ✓" if total <= preferences.budget_usd else "Over budget ✗"
        }

    # ── Full Plan Generation ───────────────────────
    def generate_plan(self, preferences: UserPreferences) -> Dict:
        destinations = self.recommend_destinations(preferences)
        top = destinations[0]["destination"]
        itinerary = self.build_itinerary(top, preferences)
        cost = self.cost_assessment(top, preferences)
        return {
            "recommended_destinations": destinations[:3],
            "selected_destination": top,
            "itinerary": itinerary,
            "cost_assessment": cost
        }


# ──────────────────────────────────────────────
# PRETTY PRINT
# ──────────────────────────────────────────────

def print_plan(plan: Dict):
    print("\n" + "="*60)
    print("  AI TRAVEL PLANNER – PERSONALISED TOUR PLAN")
    print("="*60)

    print("\n📍 TOP RECOMMENDED DESTINATIONS:")
    for i, d in enumerate(plan["recommended_destinations"], 1):
        b = "✓" if d["budget_feasible"] else "✗"
        s = "✓" if d["season_compatible"] else "✗"
        print(f"  {i}. {d['destination']:<15} score={d['score']}  budget:{b}  season:{s}  est. ${d['estimated_cost_usd']}")

    itin = plan["itinerary"]
    print(f"\n🗺  ITINERARY: {itin['destination']} ({itin['duration']}, {itin['season']})")
    print(f"   Language: {itin['language']}  |  Currency: {itin['currency']}")
    for day in itin["itinerary"]:
        print(f"\n  Day {day['day']}: {day['activity']}")
        print(f"    🍽  Eat: {day['lunch_recommendation']}")
        if day["wine_pairing"]:
            w = day["wine_pairing"]
            print(f"    🍷 Wine: {w['wine']} ({w['region']}) – {w['notes']}")
        print(f"    💡 Tip: {day['tip']}")

    ca = plan["cost_assessment"]
    print(f"\n💰 COST ASSESSMENT ({ca['destination']}, {ca['duration_days']} days):")
    print(f"   Accommodation : ${ca['accommodation_usd']}")
    print(f"   Food          : ${ca['food_usd']}")
    print(f"   Activities    : ${ca['activities_usd']}")
    print(f"   Local Transport: ${ca['local_transport_usd']}")
    print(f"   ─────────────────────────")
    print(f"   Total Estimate : ${ca['total_estimated_usd']}")
    print(f"   Your Budget   : ${ca['budget_available_usd']}")
    print(f"   {ca['verdict']}  (surplus/deficit: ${ca['surplus_or_deficit_usd']})")
    print()


if __name__ == "__main__":
    planner = TravelPlannerKB()

    # Scenario: cultural + food traveller, 5 days, $1200 budget, spring
    prefs = UserPreferences(
        name="Arjun",
        budget_usd=1200,
        duration_days=5,
        interests=["cultural", "food", "historic"],
        season="spring"
    )
    plan = planner.generate_plan(prefs)
    print_plan(plan)

    # Scenario 2: beach lover, tight budget
    prefs2 = UserPreferences(
        name="Priya",
        budget_usd=600,
        duration_days=7,
        interests=["beach", "spiritual", "nature"],
        season="summer"
    )
    plan2 = planner.generate_plan(prefs2)
    print_plan(plan2)
