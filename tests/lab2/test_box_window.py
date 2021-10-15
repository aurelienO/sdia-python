from os import error
import numpy as np
import pytest

from sdia_python.lab2.box_window import BoxWindow, UnitBoxWindow


def test_raise_assertion_error_when_points_is_not_an_array():
    with pytest.raises(AssertionError):
        # call_something_that_raises_TypeError()
        L = [[1, 2], [3, 4]]
        box = BoxWindow(L)
        raise AssertionError()


def test_raise_Exception_when_bounds_are_incorrect():
    with pytest.raises(Exception):
        L = np.array([[2, 1], [3, 4]])
        box = BoxWindow(L)
        raise Exception


def test_raise_Exception_when_dimension_is_incorrect():
    with pytest.raises(Exception):
        L = np.array([[1, 2, 4], [3, 4, 5]])
        box = BoxWindow(L)
        raise Exception


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (np.array([[2.5, 2.5]]), "BoxWindow: [2.5, 2.5]"),
        (np.array([[0, 5], [0, 5]]), "BoxWindow: [0, 5] x [0, 5]"),
        (
            np.array([[0, 5], [-1.45, 3.14], [-10, 10]]),
            "BoxWindow: [0.0, 5.0] x [-1.45, 3.14] x [-10.0, 10.0]",
        ),
    ],
)
def test_box_string_resentation(bounds, expected):
    assert str(BoxWindow(bounds)) == expected


@pytest.fixture
def box_2d_05():
    return BoxWindow(np.array([[0, 5], [0, 5]]))


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([0, 0]), True),
        (np.array([2.5, 2.5]), True),
        (np.array([-1, 5]), False),
        (np.array([10, 3]), False),
    ],
)
def test_indicator_function_box_2d(box_2d_05, point, expected):
    is_in = box_2d_05.indicator_function(point)
    assert is_in == expected


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([0, 0]), True),
        (np.array([2.5, 2.5]), True),
        (np.array([-1, 5]), False),
        (np.array([10, 3]), False),
    ],
)
def test_contains_function_box_2d(box_2d_05, point, expected):
    is_in = box_2d_05.__contains__(point)
    assert is_in == expected


# ================================
# ==== WRITE YOUR TESTS BELOW ====
# ================================


def test_raise_error_when_dimension_didnot_match_with_point():
    with pytest.raises(AssertionError):
        L = np.array([[1, 2], [3, 4]])
        box = BoxWindow(L)
        box.__contains__(np.array([0.5, 3.5, 2.5]))
        raise AssertionError


@pytest.mark.parametrize(
    "box, expected",
    [
        (np.array([[1, 2]]), 1),
        (np.array([[1, 2], [3, 4]]), 2),
        (np.array([[1, 2], [3, 4], [5, 6]]), 3),
        (np.array([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]]), 6),
    ],
)
def test_len_box(box, expected):
    assert len(BoxWindow(box)) == expected


@pytest.mark.parametrize(
    "box, expected",
    [
        (np.array([[1, 2]]), 1),
        (np.array([[1, 2], [3, 4]]), 2),
        (np.array([[1, 2], [3, 4], [5, 6]]), 3),
        (np.array([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]]), 6),
    ],
)
def test_dimension_box(box, expected):
    assert BoxWindow(box).dimension() == expected


@pytest.mark.parametrize(
    "box, expected",
    [
        (np.array([[1, 2]]), 1),
        (np.array([[1, 3], [3, 5]]), 4),
        (np.array([[1, 2], [3, 4], [5, 7]]), 2),
        (np.array([[1, 2], [3, 5], [5, 9], [1, 2], [3, 5], [5, 6]]), 16),
    ],
)
def test_volume_box(box, expected):
    assert BoxWindow(box).volume() == expected


@pytest.mark.parametrize(
    "box, expected",
    [
        (np.array([[1, 2]]), np.array([1.5])),
        (np.array([[1, 3], [3, 5]]), np.array([2, 4])),
        (np.array([[1, 2], [3, 4], [5, 7]]), np.array([1.5, 3.5, 6.0])),
        (
            np.array([[1, 2], [3, 5], [5, 9], [1, 2], [3, 5], [5, 6]]),
            np.array([1.5, 4.0, 7.0, 1.5, 4.0, 5.5]),
        ),
    ],
)
def test_center_box(box, expected):
    assert np.array_equal(BoxWindow(box).center(), expected)


@pytest.mark.parametrize(
    "bounds",
    [
        (np.array([[1, 2]])),
        (np.array([[1, 3], [3, 5]])),
        (np.array([[1, 2], [3, 4], [5, 7]])),
        (np.array([[1, 2], [3, 5], [5, 9], [1, 2], [3, 5], [5, 6]])),
    ],
)
def test_rand_onepoint(bounds):
    box = BoxWindow(bounds)
    assert box.__contains__(box.rand()[0])


def test_rand_multiplepoint_3dimension():
    box = BoxWindow(np.array([[1, 2], [10, 15.5], [3.5, 7]]))
    coord = box.rand(100)
    for value in coord:
        assert box.__contains__(value)


def test_raise_error_when_center_is_not_an_array():
    with pytest.raises(AssertionError):
        center = [1, 2, 3]
        box = UnitBoxWindow(center)
        raise AssertionError


@pytest.mark.parametrize(
    "center, expected",
    [
        (np.array([0]), "BoxWindow: [-0.5, 0.5]"),
        (np.array([0, 0]), "BoxWindow: [-0.5, 0.5] x [-0.5, 0.5]"),
        (np.array([0, 0, 0]), "BoxWindow: [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]",),
    ],
)
def test_UnitBoxWindow(center, expected):
    unitBox = UnitBoxWindow(center)
    assert unitBox.__str__() == expected


@pytest.mark.parametrize(
    "center, expected",
    [
        (np.array([2.5]), "BoxWindow: [2.0, 3.0]"),
        (np.array([1.5, 4]), "BoxWindow: [1.0, 2.0] x [3.5, 4.5]"),
        (np.array([2.5, 8, -4.5]), "BoxWindow: [2.0, 3.0] x [7.5, 8.5] x [-5.0, -4.0]"),
    ],
)
def test_UnitBoxWindow_with_center_specified(center, expected):
    unitBox = UnitBoxWindow(center)
    assert unitBox.__str__() == expected


@pytest.mark.parametrize(
    "center", [(np.array([2.5])), (np.array([1.5, 4])), (np.array([2.5, 8, -4.5])),],
)
def test_UnitBoxWindow_volume_is_equal_to_one(center):
    unitBox = UnitBoxWindow(center)
    assert unitBox.volume() == 1
