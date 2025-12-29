import pytest

from nl2graph.eval.metrics import Metrics


class TestNormalizeAnswers:

    def test_empty_list(self):
        result = Metrics.normalize_answers([])
        assert result == set()

    def test_single_string(self):
        result = Metrics.normalize_answers(["Paris"])
        assert result == {"paris"}

    def test_multiple_strings(self):
        result = Metrics.normalize_answers(["Paris", "London", "Berlin"])
        assert result == {"paris", "london", "berlin"}

    def test_with_whitespace(self):
        result = Metrics.normalize_answers(["  Paris  ", "London\t"])
        assert result == {"paris", "london"}

    def test_case_insensitive(self):
        result = Metrics.normalize_answers(["PARIS", "Paris", "paris"])
        assert result == {"paris"}

    def test_with_none_values(self):
        result = Metrics.normalize_answers([None, "Paris", None])
        assert result == {"paris"}

    def test_with_numbers(self):
        result = Metrics.normalize_answers([42, 3.14, "100"])
        assert result == {"42", "3.14", "100"}

    def test_mixed_types(self):
        result = Metrics.normalize_answers(["Paris", 42, None, "London"])
        assert result == {"paris", "42", "london"}


class TestExactMatch:

    def test_exact_match_true(self):
        gold = {"paris", "london"}
        pred = {"paris", "london"}
        assert Metrics.exact_match(gold, pred) == 1.0

    def test_exact_match_false_subset(self):
        gold = {"paris", "london"}
        pred = {"paris"}
        assert Metrics.exact_match(gold, pred) == 0.0

    def test_exact_match_false_superset(self):
        gold = {"paris"}
        pred = {"paris", "london"}
        assert Metrics.exact_match(gold, pred) == 0.0

    def test_exact_match_empty(self):
        assert Metrics.exact_match(set(), set()) == 1.0

    def test_exact_match_different(self):
        gold = {"paris"}
        pred = {"london"}
        assert Metrics.exact_match(gold, pred) == 0.0


class TestPrecisionRecallF1:

    def test_perfect_match(self):
        gold = {"paris", "london"}
        pred = {"paris", "london"}
        precision, recall, f1 = Metrics.precision_recall_f1(gold, pred)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_partial_match(self):
        gold = {"paris", "london", "berlin"}
        pred = {"paris", "london"}
        precision, recall, f1 = Metrics.precision_recall_f1(gold, pred)
        assert precision == 1.0
        assert recall == pytest.approx(2/3)
        assert f1 == pytest.approx(0.8)

    def test_extra_predictions(self):
        gold = {"paris"}
        pred = {"paris", "london"}
        precision, recall, f1 = Metrics.precision_recall_f1(gold, pred)
        assert precision == 0.5
        assert recall == 1.0
        assert f1 == pytest.approx(2/3)

    def test_no_overlap(self):
        gold = {"paris"}
        pred = {"london"}
        precision, recall, f1 = Metrics.precision_recall_f1(gold, pred)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_empty_prediction(self):
        gold = {"paris"}
        pred = set()
        precision, recall, f1 = Metrics.precision_recall_f1(gold, pred)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_empty_gold(self):
        gold = set()
        pred = {"paris"}
        precision, recall, f1 = Metrics.precision_recall_f1(gold, pred)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_both_empty(self):
        precision, recall, f1 = Metrics.precision_recall_f1(set(), set())
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0


class TestAccuracy:

    def test_all_correct(self):
        assert Metrics.accuracy(10, 10) == 1.0

    def test_none_correct(self):
        assert Metrics.accuracy(0, 10) == 0.0

    def test_partial(self):
        assert Metrics.accuracy(7, 10) == 0.7

    def test_zero_total(self):
        assert Metrics.accuracy(0, 0) == 0.0

    def test_zero_correct_with_total(self):
        assert Metrics.accuracy(0, 5) == 0.0


class TestStringMatch:

    def test_exact_match(self):
        assert Metrics.string_match("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert Metrics.string_match("PARIS", "paris") is True

    def test_with_whitespace(self):
        assert Metrics.string_match("  Paris  ", "Paris") is True

    def test_no_match(self):
        assert Metrics.string_match("Paris", "London") is False

    def test_empty_strings(self):
        assert Metrics.string_match("", "") is True

    def test_partial_no_match(self):
        assert Metrics.string_match("Par", "Paris") is False
