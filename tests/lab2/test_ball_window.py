import numpy as np
from numpy.lib.twodim_base import triu_indices_from
import pytest

from sdia_python.lab2.ball_window import BallWindow
from sdia_python.lab2.box_window import BoxWindow


def test_raise_assertion_error_when_center_is_not_an_array():
    with pytest.raises(AssertionError):
        # call_something_that_raises_TypeError()
        L = [1, 2]
        ball = BallWindow(L)
        raise AssertionError()


def test_raise_Exception_when_radius_is_negative():
    with pytest.raises(Exception):
        L = np.array([3, 4])
        ball = BallWindow(L, 2)
        raise Exception


@pytest.mark.parametrize(
    "center, radius, expected",
    [
        (np.array([0]), 4, "BallWindow: center = [0], radius = 4"),
        (np.array([2.5, 2.5]), 3.7, "BallWindow: center = [2.5, 2.5], radius = 3.7"),
        (np.array([-1, 5]), 8.0, "BallWindow: center = [-1, 5], radius = 8.0"),
    ],
)
def test_str(center, radius, expected):
    ball = BallWindow(center, radius)
    assert ball.__str__() == expected


@pytest.mark.parametrize(
    "box, expected",
    [(np.array([1]), 1), (np.array([1, 2.2]), 2), (np.array([1.4, 2.6, 3.9]), 3),],
)
def test_dimension_box(box, expected):
    ball = BallWindow(box, 4)
    assert ball.dimension() == expected


# ? ball or center
# ! tests pass but on a wrong implementation
@pytest.mark.parametrize(
    "ball, radius, expected",
    [
        (np.array([1]), 2, 4),
        (np.array([1, 3]), 3, 36),
        (np.array([1.4, 2.6, 3.9]), 2.5, 125.0),
    ],
)
def test_volume_box(ball, radius, expected):
    ball = BallWindow(ball, radius)
    assert ball.volume() == expected


@pytest.mark.parametrize(
    "center, radius, point, expected",
    [
        (np.array([1]), 3, np.array([2]), True),
        (np.array([1]), 3, np.array([5]), False),
        (np.array([3.5]), 0.5, np.array([4]), True),
        (np.array([3.5]), 0.5, np.array([4.01]), False),
        (np.array([-2.5]), 1.5, np.array([-3.9]), True),
        (np.array([-2.5]), 1.5, np.array([0]), False),
        (np.array([0]), 2, np.array([0]), True),
        (np.array([0]), 2, np.array([2.02]), False),
    ],
)
def test_contains_oneDimension(center, radius, point, expected):
    assert BallWindow(center, radius).__contains__(point) == expected


@pytest.mark.parametrize(
    "center, radius, point, expected",
    [
        (np.array([0, 0]), 1, np.array([0.5, 0.5]), True),
        (np.array([0, 0]), 1, np.array([1, 2]), False),
        (np.array([3.5, 2.5]), 0.25, np.array([3.25, 2.75]), False),
        (np.array([3.5, 2.5]), 0.5, np.array([3.5, 2.75]), True),
    ],
)
def test_contains_twoDimension(center, radius, point, expected):
    assert BallWindow(center, radius).__contains__(point) == expected


@pytest.mark.parametrize(
    "center, radius, point, expected",
    [
        (np.array([1]), 3, np.array([2]), True),
        (np.array([1]), 3, np.array([5]), False),
        (np.array([3.5]), 0.5, np.array([4]), True),
        (np.array([3.5]), 0.5, np.array([4.01]), False),
        (np.array([-2.5]), 1.5, np.array([-3.9]), True),
        (np.array([-2.5]), 1.5, np.array([0]), False),
        (np.array([0]), 2, np.array([0]), True),
        (np.array([0]), 2, np.array([2.02]), False),
    ],
)
def test_indicator_function_oneDimension(center, radius, point, expected):
    assert BallWindow(center, radius).indicator_function(point) == expected


@pytest.mark.parametrize(
    "center, radius, point, expected",
    [
        (np.array([0, 0]), 1, np.array([0.5, 0.5]), True),
        (np.array([0, 0]), 1, np.array([1, 2]), False),
        (np.array([3.5, 2.5]), 0.25, np.array([3.25, 2.75]), False),
        (np.array([3.5, 2.5]), 0.5, np.array([3.5, 2.75]), True),
    ],
)
def test_indicator_function_twoDimension(center, radius, point, expected):
    assert BallWindow(center, radius).indicator_function(point) == expected


# ? does this raise a TypeError or an AssertionError
def test_raise_assertion_error_when_points_is_not_of_good_dimension():
    with pytest.raises(AssertionError):
        ball1 = BallWindow(np.array([0, 0]), 1)
        np.array([1, 2, 3]) in ball1


# def test_rand_onepoint_onedimension():
#    ball = BallWindow(np.array([1, 2]), 3)
#   assert ball.__contains__(ball.rand()[0])


# def test_rand_multiplepoint_3dimension():
#    ball = BallWindow(np.array([1, 15.5, 3.5]), 2)
#    coord = ball.rand(100)
#    for value in coord:
#        assert ball.__contains__(value)
