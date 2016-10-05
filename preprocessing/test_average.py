import unittest
import os
import numpy as np
from numpy import nan
from scipy import ndimage
from preprocessing import raster_average

UNCOVER = os.environ['UNCOVER']


class TestRasterAverage(unittest.TestCase):

    def setUp(self):
        self.data = np.array([[1.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1.4, 1.6, 1.8]])
        self.expected_average_2 = np.array([[1.0, 1.1, 1.3, 1.5, 1.7],
                                            [1.0, 1.1, 1.3, 1.5, 1.7],
                                            [1.0, 1.1, 1.3, 1.5, 1.7],
                                            [1.0, 1.1, 1.3, 1.5, 1.7],
                                            [1.0, 1.1, 1.3, 1.5, 1.7]])

        self.data_rand = np.random.rand(5, 5)

    def test_average_size2(self):
        averaged_data = raster_average.filter_data(self.data,
                                                   size=2)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_2)

    def test_average_size3(self):
        averaged_data = raster_average.filter_data(self.data,
                                                   size=3)
        np.testing.assert_array_almost_equal(averaged_data[:, 1:-1],
                                             self.data[:, 1:-1])


class TestUniformFilterWithNoData(unittest.TestCase):

    def setUp(self):
        self.data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
                              [1000.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1000.0, 1.6, 1.8],
                              [2.0, 1.2, 1.4, 1.6, 1.8],
                              [1000.0, 1.2, 1.4, 1.6, 1.8]])

        self.expected_average_3 = np.array([[nan, 1.2, 1.3, 1.4, 1.6],
                                            [nan, 1.2, 1.3, 1.4, 1.6],
                                            [1.0, 1.15, 1.23333333, 1.4, 1.625],
                                            [1.5, 1.32, 1.34285714, 1.4, 1.625],
                                            [1.5, 1.32, 1.34285714, 1.4,
                                             1.625]])

        self.expected_average_5 = np.array([[nan, 1.2, 1.3, 1.4, 1.5],
                                            [nan, 1.2, 1.3, 1.4, 1.5],
                                            [1.0, 1.15, 1.23333333, 1.35555556,
                                             1.46666667],
                                           [1.5, 1.3, 1.33333333, 1.41538462,
                                            1.50588235],
                                           [1.5, 1.28571429, 1.32727273,
                                            1.4125, 1.5047619]])

    def test_average_size3(self):
        averaged_data = raster_average.filter_uniform_filter(
            self.data, size=3, no_data_val=1000.0, func=np.nanmean)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_3)

    def test_average_size5(self):
        averaged_data = raster_average.filter_uniform_filter(
            self.data, size=5, no_data_val=1000.0, func=np.nanmean)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_5)


class TestFilterCenterWithNoData(unittest.TestCase):

    def setUp(self):
        self.data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
                              [1000.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1000.0, 1.6, 1.8],
                              [2.0, 1.2, 1.4, 1.6, 1.8],
                              [1000.0, 1.2, 1.4, 1.6, 1.8]])

        self.expected_average_3 = np.array([[1.2, 1.3, 1.4, 1.6, 1.7],
                                            [1.15, 1.23333333, 1.4, 1.625, 1.7],
                                            [1.32, 1.34285714, 1.4, 1.625, 1.7],
                                            [1.32, 1.34285714, 1.4, 1.625, 1.7],
                                            [1.46666667, 1.44, 1.4, 1.6, 1.7]])

        self.expected_average_5 = np.array(
            [[1.23333333, 1.35555556, 1.46666667, 1.50909091, 1.625],
             [1.33333333, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
             [1.32727273, 1.4125, 1.5047619, 1.50526316, 1.61428571],
             [1.33333333, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
             [1.34285714, 1.42, 1.50769231, 1.50909091, 1.625]]
            )

        self.data_5x4 = np.array([[1000.0, 1.2, 1.4, 1.6],
                              [1000.0, 1.2, 1.4, 1.6],
                              [1.0, 1.2, 1000.0, 1.6],
                              [2.0, 1.2, 1.4, 1.6],
                              [1000.0, 1.2, 1.4, 1.6]])
        self.expected_average_3_5x4 = np.array([[1.2, 1.3, 1.4, 1.5],
                                            [1.15, 1.23333333, 1.4, 1.52],
                                            [1.32, 1.34285714, 1.4, 1.52],
                                            [1.32, 1.34285714, 1.4, 1.52],
                                            [1.46666667, 1.44, 1.4, 1.5]])

    def test_average_size3(self):
        averaged_data = raster_average.filter_center(
            self.data, size=3, no_data_val=1000.0, func=np.nanmean)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_3)

    def test_average_size3_5x4(self):
        averaged_data = raster_average.filter_center(
            self.data_5x4, size=3, no_data_val=1000.0, func=np.nanmean)

        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_3_5x4)

    def test_average_size5(self):
        averaged_data = raster_average.filter_center(
            self.data, size=5, no_data_val=1000.0, func=np.nanmean)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_5)

if __name__ == '__main__':
    unittest.main()