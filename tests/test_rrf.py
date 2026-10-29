"""Tests for Reciprocal Rank Fusion."""
import pytest
from src.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_single_list():
    result = reciprocal_rank_fusion([[0, 1, 2]])
    assert result == [0, 1, 2]


def test_rrf_two_lists_agree():
    result = reciprocal_rank_fusion([[0, 1, 2], [0, 1, 2]])
    assert result[0] == 0


def test_rrf_two_lists_disagree():
    # Doc 0 ranked 1st by list A, Doc 1 ranked 1st by list B
    result = reciprocal_rank_fusion([[0, 2, 3], [1, 2, 3]])
    # Both doc 0 and 1 are top; doc 2 should be near top too
    assert 2 in result[:3]


def test_rrf_empty_lists():
    result = reciprocal_rank_fusion([[], []])
    assert result == []
