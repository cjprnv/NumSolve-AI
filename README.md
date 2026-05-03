## Q1: Game Search Algorithms
**File:** `q1_search/search_algorithms.py`  
**Tests:** `q1_search/test_search.py`

### Algorithms Implemented
| Algorithm | Description |
|-----------|-------------|
| Minimax | Full game tree search; MAX maximises, MIN minimises |
| Alpha-Beta Pruning | Minimax with α/β cut-offs; reduces nodes by ~50–90% |
| Heuristic Alpha-Beta | Depth-limited + evaluation function (for large trees) |
| MCTS | Select (UCB1) → Expand → Rollout → Backpropagate |

**Game:** Tic-Tac-Toe (3×3). X=MAX (+1), O=MIN (−1).

### Key Results
- Alpha-Beta explored **18,296 nodes** vs Minimax **549,945** on same empty board
- All algorithms converge to the same optimal move value

### Run
```bash
python q1_search/search_algorithms.py   # demo
python q1_search/test_search.py         # 10 tests
```

---

## Q2: AI Travel Planner
**File:** `q2_travel_planner/travel_planner.py`  
**Tests:** `q2_travel_planner/test_travel_planner.py`

### Knowledge Bases Reused
- **Tourist Places KB** – 5 destinations with type, cost, season data
- **Food Recommendation KB** – Local dishes per destination
- **Wine Ontology KB** – Food–wine pairing rules
- **Cost Assessment KB** – Per-day budget breakdown (accommodation 40%, food 30%, activities 20%, transport 10%)

### Inference Rules
1. **Interest matching** – Score destinations by overlap with user interests
2. **Budget filtering** – Mark infeasible destinations
3. **Season compatibility** – Prefer destinations in best season
4. **Itinerary building** – Assign attractions + food + wine per day
5. **Cost assessment** – Compute surplus/deficit

### Run
```bash
python q2_travel_planner/travel_planner.py   # two demo scenarios
python q2_travel_planner/test_travel_planner.py   # 10 tests
```

---

## Q3: Knowledge Graphs
**File:** `q3_knowledge_graphs/knowledge_graph.py`  
**Tests:** `q3_knowledge_graphs/test_knowledge_graph.py`

### Features
| Feature | Detail |
|---------|--------|
| Triple Store | RDF-like (subject, predicate, object) with 3 indices |
| Pattern Query | `query(subject, predicate, object)` with None wildcards |
| SPARQL-like SELECT | Multi-pattern joins with `?variable` syntax |
| Forward Chaining | Rule-based inference over triples |
| BFS Traversal | Graph reachability from a node |
| Shortest Path | BFS path between two nodes |
| Serialisation | JSON export/import |

**Domain:** Movies & Actors (40 triples; Nolan films, actors, awards)

**Inference Rules:**
- Actor who acts in a genre film → `has_experience_in_genre`
- Director whose film won Oscar → `is_award_winning_director`

### Run
```bash
python q3_knowledge_graphs/knowledge_graph.py   # demo
python q3_knowledge_graphs/test_knowledge_graph.py   # 13 tests
```

---

## Q4: Bayesian Networks
**File:** `q4_bayesian/bayesian_network.py`  
**Tests:** `q4_bayesian/test_bayesian.py`

### Network: Classic Alarm (Russell & Norvig)
```
Burglary   Earthquake
    \       /
     Alarm
    /     \
JohnCalls  MaryCalls
```

### Inference Methods
| Method | Type | Notes |
|--------|------|-------|
| Enumeration | Exact | Sum over full joint; O(2^n) |
| Variable Elimination | Exact | Marginalise hidden vars; more efficient |
| Prior Sampling | Approximate | Sample prior, filter by evidence |
| Likelihood Weighting | Approximate | Weight samples; more sample-efficient |

### Key Result
```
P(Burglary=T | JohnCalls=T, MaryCalls=T) ≈ 0.284  (textbook value)
```
All four methods converge to this result.

### Run
```bash
python q4_bayesian/bayesian_network.py   # demo
python q4_bayesian/test_bayesian.py      # 11 tests
```

---

## Running All Tests
```bash
python q1_search/test_search.py
python q2_travel_planner/test_travel_planner.py
python q3_knowledge_graphs/test_knowledge_graph.py
python q4_bayesian/test_bayesian.py
```

**Total: 44 test cases, all passing.**
