# cSpell: disable=invalid-name
"""Unit Test for Front End."""
import unittest

import cv2

from SuperPointPretrainedNetwork.feature_extraction.Shicong_02_front_end import FrontEnd


class TestFrontEnd(unittest.TestCase):
    def setUp(self):
        image_directory_path = 'SuperPointPretrainedNetwork/feature_extraction/undistort_images/'
        image_extension = '*.jpg'
        image_size = (640, 480)
        nn_thresh = 0.7
        self.front_end = FrontEnd(
            image_directory_path, image_extension, image_size, nn_thresh)

    def test_leading_zero(self):
        """test leading zero"""
        actual = self.front_end.leading_zero(145)
        expected = '0000145'
        self.assertEqual(actual, expected)

    def test_get_image_paths(self):
        """test get image paths"""
        self.front_end.get_image_paths()
        self.assertEqual(len(self.front_end.img_paths), 6)


if __name__ == "__main__":
    unittest.main()
