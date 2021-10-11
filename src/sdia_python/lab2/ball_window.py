import numpy as np

from sdia_python.lab2.utils import get_random_number_generator


class BallWindow:

    """This class represents a ball according to the norm 1 of any dimension"""

    def __init__(self, center, radius=1):

        """Constructor of the class : build a ball whose dimension is given by the size of the center array and the radius by the float radius.

        Args:
            center (numpy.array): an array containing the coordinates of the center.
            radius (float): the radius of the ball.
        """

        assert isinstance(center, np.ndarray)
        if radius < 0:
            raise Exception("radius must be positive")
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
        # todo rewrite the method
        # ! the volume (area) of a disk = pi r^2
        # ? is this tested
        return (2 * self.radius) ** self.dimension()

    def indicator_function(self, point):
        """Return True if the ball contains the point given in argument.

        Args:
            point (numpy.array): a point represented by a numpy array of same size that the center.

        Returns:
            boolean: True if the ball contains the point given in argument
        """
        # ? how would you handle multiple points
        return self.__contains__(point)

    def rand(self, numberOfPoints=1, rng=None):
        """Generate n points uniformly at random inside the BallWindow.

        Args:
            numberOfPoints (int, optional): Number of points. Defaults to 1.
            rng ((numpy.random._generator.Generator, optional): Random number generator. Defaults to None.

        Returns:
            list: A list of n points generated uniformly at random inside the BallWindow.
        """
        rng = get_random_number_generator(rng)
        points = []
        # ! naming: snake case for variables number_of_points
        # ! readability
        # * exploit numpy, rng.uniform(a, b, size=n)
        for k in range(0, numberOfPoints):
            pointk = np.zeros([self.dimension()])
            # * iterate over self.center
            for i in range(0, self.dimension()):
                c = rng.uniform(
                    self.center[i] - self.radius, self.center[i] + self.radius,
                )
                pointk[i] = c
            points.append(pointk)
        return points
        # ? are you sure points are uniformly distributed
