"""
Unit tests for the query router and the CRAG retrieval check.
Run with: pytest tests/ -v
"""

import pytest

from src.rag.router import Confidence, CRAGChecker, QueryRouter, RouteType


@pytest.fixture(scope="module")
def router():
    return QueryRouter()


class TestRouteSelection:
    @pytest.mark.parametrize("query, expected", [
        ("Hello!", RouteType.DIRECT),
        ("thanks", RouteType.DIRECT),
        ("What can you do?", RouteType.DIRECT),
        ("How many awards are in each category?", RouteType.ANALYTICAL),
        ("What is the distribution of subcategories?", RouteType.ANALYTICAL),
        ("What kind of messages do engineering teams write?", RouteType.SEMANTIC),
        ("Describe the tone of recognition for designers", RouteType.SEMANTIC),
        ("Who could replace a Senior Director?", RouteType.SUBSTITUTION),
        ("Which departments recognize each other most?", RouteType.GRAPH),
        ("What is the recognition flow between engineering and product?", RouteType.GRAPH),
        ("Compare the top categories and show me example messages", RouteType.HYBRID),
    ])
    def test_routes_query_to_expected_type(self, router, query, expected):
        assert router.route(query).route == expected

    def test_greeting_inside_a_real_question_is_not_direct(self, router):
        # DIRECT patterns are anchored, so "hi" only counts when it is the whole message.
        assert router.route("hi, how many awards are there?").route == RouteType.ANALYTICAL

    def test_tools_match_route(self, router):
        assert router.route("Hello").tools == []
        assert router.route("How many awards are there?").tools == ["pandas_analytics"]
        assert router.route("Describe the themes").tools == ["vector_search"]
        assert router.route("Who could replace a manager?").tools == ["substitution"]
        assert router.route("Show the recognition network").tools == ["graph_query"]


class TestRoutePrecedence:
    def test_substitution_beats_graph(self, router):
        decision = router.route("Who could replace someone in the engineering department?")
        assert decision.route == RouteType.SUBSTITUTION

    def test_graph_beats_analytical(self, router):
        # "most" is an analytical signal, "department" a graph signal.
        decision = router.route("Which department gets the most awards?")
        assert decision.route == RouteType.GRAPH

    def test_analytical_and_semantic_together_is_hybrid(self, router):
        decision = router.route("How many awards are there and describe the themes")
        assert decision.route == RouteType.HYBRID
        assert decision.tools == ["pandas_analytics", "vector_search"]


class TestFallbackAndConfidence:
    def test_no_signals_falls_back_to_low_confidence_hybrid(self, router):
        decision = router.route("quarterly onboarding")
        assert decision.route == RouteType.HYBRID
        assert decision.confidence == Confidence.LOW

    def test_graph_confidence_grows_with_signals(self, router):
        assert router.route("Show the network").confidence == Confidence.MEDIUM
        assert router.route("Show the network of central hubs").confidence == Confidence.HIGH

    def test_crag_enabled_only_for_retrieval_routes(self, router):
        assert router.route("Describe the themes").use_crag is True
        assert router.route("quarterly onboarding").use_crag is True
        assert router.route("How many awards are there?").use_crag is False
        assert router.route("Hello").use_crag is False


class TestCRAGChecker:
    @pytest.fixture
    def checker(self):
        return CRAGChecker(threshold=0.25)

    def test_high_overlap_proceeds(self, checker):
        result = checker.check("engineering teamwork awards", "Engineering awards for great teamwork")
        assert result["action"] == "proceed"
        assert result["passed"] is True
        assert result["score"] == pytest.approx(1.0)

    def test_score_at_threshold_proceeds(self, checker):
        # 1 of 4 query words found -> 0.25
        result = checker.check("alpha bravo charlie delta", "alpha")
        assert result["score"] == pytest.approx(0.25)
        assert result["action"] == "proceed"

    def test_partial_overlap_reranks(self, checker):
        # 1 of 5 query words found -> 0.2
        result = checker.check("alpha bravo charlie delta echo", "alpha")
        assert result["action"] == "rerank"
        assert result["passed"] is False

    def test_score_at_half_threshold_reranks(self, checker):
        # 1 of 8 query words found -> 0.125
        result = checker.check("alpha bravo charlie delta echo foxtrot golf hotel", "alpha")
        assert result["score"] == pytest.approx(0.125)
        assert result["action"] == "rerank"

    def test_no_overlap_falls_back(self, checker):
        result = checker.check("engineering teamwork", "finance budget review")
        assert result["action"] == "fallback"
        assert result["score"] == 0.0

    def test_empty_context_falls_back(self, checker):
        assert checker.check("engineering teamwork", "")["action"] == "fallback"

    def test_short_words_are_ignored(self, checker):
        # Words under 3 characters are not scored, so this query has no scorable words.
        assert checker.check("is it ok", "is it ok")["score"] == 0.0

    def test_matching_is_case_insensitive(self, checker):
        assert checker.check("ENGINEERING", "engineering")["action"] == "proceed"
