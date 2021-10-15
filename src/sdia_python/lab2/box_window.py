import numpy as np

from sdia_python.lab2.utils import get_random_number_generator


class BoxWindow:
    """Representation of a box defines by [a1,b1] x [a2,b2] x ..."""

    def __init__(self, bounds):
        """Constructor of a BoxWIndow

        Args:
            bounds (numpy.array): The bounds of the box.
                                It must be of dimension N * 2
        """
        assert isinstance(bounds, np.ndarray)
        if bounds.shape[1] != 2:
            raise Exception("The dimension is not correct")
        if not np.all(np.diff(bounds) >= 0):
            raise Exception("The bounds are not correct")
        self.bounds = bounds

    def __str__(self):
        """Returns the representation of a box, for example the following string :
        "BoxWindow: [a_1, b_1] x [a_2, b_2]"

        Returns:
            str: The representation of the box
        """
        s = "BoxWindow: "
        bounds_list = [f"{list(e)}" for e in self.bounds]
        sep = " x "
        return s + sep.join(bounds_list)

    def __len__(self):
        """Returns the len of the box, ie the dimension.

        Returns:
            int: the dimension of the box
        """
        return len(self.bounds)

    def __contains__(self, point):
        """Returns True if the point belongs to the box

        Args:
            point (numpy.array): the point

        Returns:
            Boolean: True if the point belongs to the box
        """
        assert len(point) == self.dimension()
        return all(a <= x <= b for (a, b), x in zip(self.bounds, point))

    def dimension(self):
        """Returns the dimension of the box, ie the number of segment.

        Returns:
            int: the dimension of the box
        """
        return len(self)

    def volume(self):
        """Returns the volume of the box, ie the multiplication of the size of each segment.

        Returns:
            int: the volume of the box
        """
        return np.prod(np.diff(self.bounds))

    def indicator_function(self, points):
        """Returns True if the point belongs to the box

        Args:
            point (numpy.array): the point

        Returns:
            boolean: True if the point belongs to the box
        """
        return self.__contains__(points)

    def center(self):
        """Return the array with the coordinates of the center of the box.

        Returns:
            numpy array: The array with the coordinates of the center of the box.
        """
        return np.sum(self.bounds, axis=1) / 2

    def rand(self, n=1, rng=None):
        """Generate n points uniformly at random inside the BoxWindow.

        Args:
            n (int, optional): the number of points. Defaults to 1.
            rng (numpy.random._generator.Generator, optional): Random number generator. Defaults to None.

        Returns:
            A list of n points generated uniformly at random inside the BoxWindow.
        """
        rng = get_random_number_generator(rng)
        # points = (rng.uniform(a, b, n) for (a, b) in self.bounds)
        points = np.array(
            [[rng.uniform(a, b) for a, b in self.bounds] for i in range(n)]
        )
        return points


class UnitBoxWindow(BoxWindow):
    """Represent a BoxWindow where all the segment over all dimensions have a size of one."""

    def __init__(self, center):
        """Returns a unit box window, with segments of length 1 for each dimension, centered on args if the center is precised, else, it is centered on (0,0,...,0).

        Args:
            dimension (int): The dimension of the box window
            center (numpy.array, optional): The array of the center of each segment of the box window. Defaults to None.
        """

        assert isinstance(center, np.ndarray)
        bounds = np.add.outer(center, [-0.5, 0.5])
        super().__init__(bounds)
