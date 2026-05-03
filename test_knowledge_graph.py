"""Test Suite for Q3: Knowledge Graph"""
import sys, json
sys.path.insert(0, '.')
from knowledge_graph import KnowledgeGraph, build_movie_kg

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def test_add_and_query():
    kg = KnowledgeGraph("Test")
    kg.add("Alice", "knows", "Bob")
    kg.add("Alice", "knows", "Carol")
    assert len(kg) == 2
    r = kg.query(subject="Alice", predicate="knows")
    assert len(r) == 2
    print(f"{PASS} add_and_query (2 triples)")


def test_query_wildcards():
    kg = KnowledgeGraph("Test")
    kg.add("X", "type", "Film")
    kg.add("Y", "type", "Actor")
    kg.add("X", "directed_by", "Z")
    films = kg.query(predicate="type", obj="Film")
    assert len(films) == 1 and films[0][0] == "X"
    print(f"{PASS} query_wildcards")


def test_ask():
    kg = KnowledgeGraph("Test")
    kg.add("Sun", "is_a", "Star")
    assert kg.ask("Sun", "is_a", "Star") is True
    assert kg.ask("Moon", "is_a", "Star") is False
    print(f"{PASS} ask_triple")


def test_remove():
    kg = KnowledgeGraph("Test")
    kg.add("A", "B", "C")
    kg.remove("A", "B", "C")
    assert len(kg) == 0
    print(f"{PASS} remove_triple")


def test_select_sparql():
    kg = build_movie_kg()
    results = kg.select([("?film", "directed_by", "Christopher Nolan")])
    films = {b["?film"] for b in results}
    assert "Inception" in films
    assert "Interstellar" in films
    print(f"{PASS} select_sparql_nolan_films ({len(films)} films)")


def test_select_join():
    kg = build_movie_kg()
    results = kg.select([
        ("?actor", "acted_in", "?film"),
        ("?film", "has_genre", "Sci-Fi")
    ])
    actors = {b["?actor"] for b in results}
    assert "Leonardo DiCaprio" in actors
    assert "Matthew McConaughey" in actors
    print(f"{PASS} select_join_actor_scifi ({len(actors)} actors)")


def test_inference_actor_genre():
    kg = build_movie_kg()
    n = kg.apply_inference()
    assert n > 0, f"{FAIL} inference should derive new triples"
    # Leonardo acted in Inception (Sci-Fi) → should have experience in Sci-Fi
    assert kg.ask("Leonardo DiCaprio", "has_experience_in_genre", "Sci-Fi"), \
        f"{FAIL} DiCaprio should have Sci-Fi experience"
    print(f"{PASS} inference_actor_genre ({n} new triples)")


def test_inference_award_winning_director():
    kg = build_movie_kg()
    kg.apply_inference()
    # Nolan directed Inception which won Oscar → award winning director
    assert kg.ask("Christopher Nolan", "is_award_winning_director", "True"), \
        f"{FAIL} Nolan should be award-winning director"
    print(f"{PASS} inference_award_winning_director")


def test_bfs():
    kg = KnowledgeGraph("Test")
    kg.add("A", "rel", "B")
    kg.add("B", "rel", "C")
    kg.add("C", "rel", "D")
    visited = kg.bfs("A", max_depth=3)
    assert "D" in visited
    assert visited["D"] == 3
    print(f"{PASS} bfs_traversal (depth 3)")


def test_shortest_path():
    kg = KnowledgeGraph("Test")
    kg.add("A", "r", "B")
    kg.add("B", "r", "C")
    kg.add("A", "r", "C")  # shortcut
    path = kg.shortest_path("A", "C")
    assert path is not None
    assert path[0] == "A" and path[-1] == "C"
    assert len(path) == 2  # direct A→C
    print(f"{PASS} shortest_path (A→C direct: {path})")


def test_no_path():
    kg = KnowledgeGraph("Test")
    kg.add("A", "r", "B")
    kg.add("C", "r", "D")  # disconnected
    path = kg.shortest_path("A", "D")
    assert path is None
    print(f"{PASS} no_path_returns_none")


def test_serialisation():
    kg = KnowledgeGraph("Test")
    kg.add("X", "p", "Y")
    kg.add("Y", "q", "Z")
    j = kg.to_json()
    data = json.loads(j)
    assert data["triples_count"] == 2
    kg2 = KnowledgeGraph("Test2")
    kg2.from_json(j)
    assert len(kg2) == 2
    print(f"{PASS} serialisation_roundtrip")


def test_movie_kg_size():
    kg = build_movie_kg()
    assert len(kg) >= 20, f"{FAIL} movie KG should have ≥20 triples, has {len(kg)}"
    print(f"{PASS} movie_kg_size ({len(kg)} triples)")


if __name__ == "__main__":
    print("=" * 50)
    print("  Q3 Knowledge Graph – Test Suite")
    print("=" * 50)
    test_add_and_query()
    test_query_wildcards()
    test_ask()
    test_remove()
    test_select_sparql()
    test_select_join()
    test_inference_actor_genre()
    test_inference_award_winning_director()
    test_bfs()
    test_shortest_path()
    test_no_path()
    test_serialisation()
    test_movie_kg_size()
    print("\nAll tests passed!")
