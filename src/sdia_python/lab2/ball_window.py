import numpy as np

from sdia_python.lab2.utils import get_random_number_generator


class BallWindow:
    """This class represents a ball according to the norm 1 of any dimension"""

    def __init__(self, center, radius=1):
        """Constructor of the class : build a ball whose dimension is given by the size of the center array and the radius by the float radius.

        Args:
            center (numpy.array): an array containing the coordinates of the center. It must be of length <= 3.
            radius (float): the radius of the ball.
        """
        assert isinstance(center, np.ndarray)
        if radius < 0:
            raise Exception("radius must be positive")
        if len(center) > 3:
            raise Exception("Dimension is too high")
        self.center = center
        self.radius = radius

    def __str__(self):
        """Returns for example the following string :
        "BallWindow: [a_1, b_1] x [a_2, b_2]"

        Returns:
            str: The representation of the box Window
        """
        return f"BallWindow: center = {list(self.center)}, radius = {self.radius}"

    def __contains__(self, point):
        """Return True if the ball contains the point given in argument.

        Args:
            point (numpy.array): a point represented by a numpy array of same size that the center.

        Returns:
            boolean: True if the ball contains the point given in argument
        """
        assert len(point) == len(self.center)
        return np.linalg.norm(self.center - point) <= self.radius

    def dimension(self):
        """Returns the dimension of the ball.

        Returns:
            int: the dimension of the ball
        """
        return len(self.center)

    def volume(self):
        """Returns the volume of the box

        Returns:
            float: Returns the volume of the box
        """
        n = self.dimension()
        if n == 1:
            return 2 * self.radius
        if n == 2:
            return np.pi * self.radius ** 2
        return (4 / 3) * np.pi * self.radius ** 3

    def indicator_function(self, point):
        """Return True if the ball contains the point given in argument.

        Args:
            point (numpy.array): a point represented by a numpy array of same size that the center.

        Returns:
            boolean: True if the ball contains the point given in argument
        """
        # ? how would you handle multiple points
        return self.__contains__(point)

    def rand(self, n=1, rng=None):
        """Generate n points uniformly at random inside the BallWindow.

        Args:
            n (int, optional): Number of points. Defaults to 1.
            rng ((numpy.random._generator.Generator, optional): Random number generator. Defaults to None.

        Returns:
            list: A list of n points generated uniformly at random inside the BallWindow.
        """
        rng = get_random_number_generator(rng)
        d = self.dimension()
        if d == 1:
            points = rng.uniform(
                self.center[0] - self.radius, self.center[0] + self.radius, size=n
            )
            return points
        if d == 2:
            points = np.zeros((n, 2))
            r = np.sqrt(rng.uniform(0, self.radius, size=n))
            theta = rng.uniform(0, 2 * np.pi, size=n)
            points[:, 0] = r * np.cos(theta) + self.center[0]
            points[:, 1] = r * np.sin(theta) + self.center[1]
            return points

        # ? are you sure points are uniformly distributed


class UnitBallWindow:
    """Represent a BallWindow where the radius has a size of one.
    """

    def __init__(self, center):
        """Return a BallWindow where the radius is equal to one.

        Args:
            center (numpy.array): an array containing the coordinates of the center.
        """
        super().__init__(center, 1)
