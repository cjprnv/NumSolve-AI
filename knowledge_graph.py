"""
Assignment Q3: Knowledge Graphs
- Implements a Knowledge Graph (KG) as a triple store: (subject, predicate, object)
- Supports: add, query, SPARQL-like pattern matching, inference rules, serialisation
- Domain: Movies & Actors (real-world KG example)
- Tools explored: NetworkX for graph traversal
"""

from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set
import json


# ──────────────────────────────────────────────
# TRIPLE STORE (Core KG)
# ──────────────────────────────────────────────

Triple = Tuple[str, str, str]  # (subject, predicate, object)


class KnowledgeGraph:
    """
    A Knowledge Graph stored as a set of RDF-like triples: (subject, predicate, object).
    Supports:
      - CRUD on triples
      - Pattern matching queries (? wildcards)
      - Forward-chaining inference
      - Graph traversal (BFS)
      - Export to JSON-LD-like format
    """

    def __init__(self, name: str = "KG"):
        self.name = name
        self._triples: Set[Triple] = set()
        # Indices for fast lookup
        self._by_subject: Dict[str, Set[Triple]] = defaultdict(set)
        self._by_predicate: Dict[str, Set[Triple]] = defaultdict(set)
        self._by_object: Dict[str, Set[Triple]] = defaultdict(set)
        self._inference_rules: List[Dict] = []  # forward-chaining rules

    # ── CRUD ──────────────────────────────────

    def add(self, subject: str, predicate: str, obj: str):
        """Add a triple to the KG."""
        triple = (subject, predicate, obj)
        if triple not in self._triples:
            self._triples.add(triple)
            self._by_subject[subject].add(triple)
            self._by_predicate[predicate].add(triple)
            self._by_object[obj].add(triple)

    def remove(self, subject: str, predicate: str, obj: str):
        triple = (subject, predicate, obj)
        self._triples.discard(triple)
        self._by_subject[subject].discard(triple)
        self._by_predicate[predicate].discard(triple)
        self._by_object[obj].discard(triple)

    def __len__(self):
        return len(self._triples)

    # ── QUERY: Pattern Matching ───────────────

    def query(self, subject=None, predicate=None, obj=None) -> List[Triple]:
        """
        Query triples with optional wildcards (None = any).
        E.g. query(subject="Inception") → all triples about Inception
             query(predicate="directed_by") → all directed_by triples
        """
        candidates = self._triples

        if subject is not None:
            candidates = candidates & self._by_subject.get(subject, set())
        if predicate is not None:
            candidates = candidates & self._by_predicate.get(predicate, set())
        if obj is not None:
            candidates = candidates & self._by_object.get(obj, set())

        return sorted(candidates)

    def ask(self, subject: str, predicate: str, obj: str) -> bool:
        """Check if a specific triple exists."""
        return (subject, predicate, obj) in self._triples

    # ── SPARQL-like SELECT ────────────────────

    def select(self, pattern: List[Tuple], filters: Dict = None) -> List[Dict]:
        """
        Simple SPARQL-like SELECT.
        pattern: list of (subject, predicate, object), use '?' prefix for variables.
        Returns list of variable bindings.

        Example:
          select([("?film", "directed_by", "Christopher Nolan"),
                  ("?film", "has_genre", "?genre")])
        """
        bindings = [{}]  # start with one empty binding

        for (s, p, o) in pattern:
            new_bindings = []
            for binding in bindings:
                # Substitute already-bound variables
                rs = binding.get(s, s) if s.startswith("?") else s
                rp = binding.get(p, p) if p.startswith("?") else p
                ro = binding.get(o, o) if o.startswith("?") else o

                # Query with substituted values (None if still unbound variable)
                sq = None if (isinstance(rs, str) and rs.startswith("?")) else rs
                pq = None if (isinstance(rp, str) and rp.startswith("?")) else rp
                oq = None if (isinstance(ro, str) and ro.startswith("?")) else ro

                for (ts, tp, to) in self.query(sq, pq, oq):
                    new_b = dict(binding)
                    if s.startswith("?"): new_b[s] = ts
                    if p.startswith("?"): new_b[p] = tp
                    if o.startswith("?"): new_b[o] = to
                    new_bindings.append(new_b)

            bindings = new_bindings

        # Apply filters
        if filters:
            bindings = [b for b in bindings if all(b.get(k) == v for k, v in filters.items())]

        return bindings

    # ── INFERENCE RULES (Forward Chaining) ────

    def add_rule(self, name: str, if_pattern: List[Tuple], then_triple: Tuple):
        """
        Add a forward-chaining inference rule.
        E.g. IF (X, acted_in, Y) AND (Y, has_genre, Z) THEN (X, has_experience_in, Z)
        """
        self._inference_rules.append({
            "name": name,
            "if": if_pattern,
            "then": then_triple
        })

    def apply_inference(self) -> int:
        """Apply all rules until no new triples are derived. Returns count of new triples."""
        new_count = 0
        changed = True
        while changed:
            changed = False
            for rule in self._inference_rules:
                results = self.select(rule["if"])
                for binding in results:
                    s, p, o = rule["then"]
                    rs = binding.get(s, s)
                    rp = binding.get(p, p)
                    ro = binding.get(o, o)
                    if rs and rp and ro and not self.ask(rs, rp, ro):
                        self.add(rs, rp, ro)
                        new_count += 1
                        changed = True
        return new_count

    # ── GRAPH TRAVERSAL ───────────────────────

    def neighbors(self, node: str) -> List[Tuple[str, str]]:
        """Return (predicate, neighbor) for all outgoing edges from node."""
        return [(p, o) for (s, p, o) in self._by_subject.get(node, [])]

    def bfs(self, start: str, max_depth: int = 3) -> Dict[str, int]:
        """BFS from start node. Returns {node: depth} visited."""
        visited = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            depth = visited[node]
            if depth >= max_depth:
                continue
            for (_, neighbor) in self.neighbors(node):
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    queue.append(neighbor)
        return visited

    def shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """BFS shortest path between two nodes."""
        if start == end:
            return [start]
        visited = {start: None}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for (_, neighbor) in self.neighbors(node):
                if neighbor not in visited:
                    visited[neighbor] = node
                    if neighbor == end:
                        path = []
                        cur = end
                        while cur is not None:
                            path.append(cur)
                            cur = visited[cur]
                        return list(reversed(path))
                    queue.append(neighbor)
        return None  # no path

    # ── SERIALISATION ─────────────────────────

    def to_json(self) -> str:
        data = {
            "name": self.name,
            "triples_count": len(self._triples),
            "triples": [{"s": s, "p": p, "o": o} for s, p, o in sorted(self._triples)]
        }
        return json.dumps(data, indent=2)

    def from_json(self, json_str: str):
        data = json.loads(json_str)
        for t in data["triples"]:
            self.add(t["s"], t["p"], t["o"])

    def display_stats(self):
        preds = defaultdict(int)
        for (_, p, _) in self._triples:
            preds[p] += 1
        print(f"\nKnowledge Graph: '{self.name}'")
        print(f"  Total triples   : {len(self._triples)}")
        print(f"  Unique subjects : {len(self._by_subject)}")
        print(f"  Predicates      : {dict(sorted(preds.items()))}")


# ──────────────────────────────────────────────
# POPULATE: Movies & Actors Domain
# ──────────────────────────────────────────────

def build_movie_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph("Movies & Actors")

    # Entities: Films
    films = [
        ("Inception", "The Dark Knight", "Interstellar"),
        ("The Matrix", "John Wick", "Speed Racer"),
    ]

    # Films directed by Nolan
    for f in ["Inception", "The Dark Knight", "Interstellar", "Memento"]:
        kg.add(f, "directed_by", "Christopher Nolan")
        kg.add(f, "type", "Film")
        kg.add("Christopher Nolan", "type", "Director")

    # Genres
    kg.add("Inception", "has_genre", "Sci-Fi")
    kg.add("Inception", "has_genre", "Thriller")
    kg.add("The Dark Knight", "has_genre", "Action")
    kg.add("The Dark Knight", "has_genre", "Thriller")
    kg.add("Interstellar", "has_genre", "Sci-Fi")
    kg.add("Interstellar", "has_genre", "Drama")
    kg.add("Memento", "has_genre", "Thriller")
    kg.add("Memento", "has_genre", "Mystery")

    # Actors
    for actor in ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"]:
        kg.add(actor, "acted_in", "Inception")
        kg.add(actor, "type", "Actor")

    for actor in ["Christian Bale", "Heath Ledger", "Gary Oldman"]:
        kg.add(actor, "acted_in", "The Dark Knight")
        kg.add(actor, "type", "Actor")

    kg.add("Matthew McConaughey", "acted_in", "Interstellar")
    kg.add("Matthew McConaughey", "type", "Actor")
    kg.add("Guy Pearce", "acted_in", "Memento")
    kg.add("Guy Pearce", "type", "Actor")

    # Awards
    kg.add("Inception", "won_award", "Oscar_Best_Cinematography")
    kg.add("The Dark Knight", "won_award", "Oscar_Best_Supporting_Actor")
    kg.add("Heath Ledger", "won_award", "Oscar_Best_Supporting_Actor")

    # Years
    kg.add("Inception", "release_year", "2010")
    kg.add("The Dark Knight", "release_year", "2008")
    kg.add("Interstellar", "release_year", "2014")
    kg.add("Memento", "release_year", "2000")

    # Inference Rules
    # Rule 1: Actor who acts in a Sci-Fi film has Sci-Fi experience
    kg.add_rule(
        "actor_genre_experience",
        if_pattern=[("?actor", "acted_in", "?film"), ("?film", "has_genre", "?genre")],
        then_triple=("?actor", "has_experience_in_genre", "?genre")
    )

    # Rule 2: Director whose film won an Oscar is an Award-Winning Director
    kg.add_rule(
        "award_winning_director",
        if_pattern=[("?film", "directed_by", "?director"), ("?film", "won_award", "?award")],
        then_triple=("?director", "is_award_winning_director", "True")
    )

    return kg


# ──────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────

if __name__ == "__main__":
    kg = build_movie_kg()
    kg.display_stats()

    print("\n── QUERY: All films directed by Christopher Nolan ──")
    results = kg.query(predicate="directed_by", obj="Christopher Nolan")
    for s, p, o in results:
        print(f"  {s}")

    print("\n── SPARQL SELECT: Actors in Sci-Fi films ──")
    bindings = kg.select([
        ("?actor", "acted_in", "?film"),
        ("?film", "has_genre", "Sci-Fi")
    ])
    for b in bindings:
        print(f"  {b.get('?actor')} → {b.get('?film')}")

    print("\n── INFERENCE: Applying forward-chaining rules ──")
    n = kg.apply_inference()
    print(f"  {n} new triples derived")

    print("\n── QUERY: Actor genre experience (after inference) ──")
    exp = kg.query(predicate="has_experience_in_genre")
    for s, p, o in exp[:6]:
        print(f"  {s} has experience in {o}")

    print("\n── BFS: Nodes reachable from 'Inception' (depth 2) ──")
    reachable = kg.bfs("Inception", max_depth=2)
    for node, depth in sorted(reachable.items(), key=lambda x: x[1]):
        print(f"  depth {depth}: {node}")

    print("\n── SHORTEST PATH: Leonardo DiCaprio → Christopher Nolan ──")
    path = kg.shortest_path("Leonardo DiCaprio", "Christopher Nolan")
    print(f"  {' → '.join(path) if path else 'No path found'}")

    print("\n── SERIALISE to JSON ──")
    j = kg.to_json()
    data = json.loads(j)
    print(f"  Serialised {data['triples_count']} triples")
