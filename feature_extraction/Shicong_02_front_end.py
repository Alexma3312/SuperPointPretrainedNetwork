# cSpell: disable=invalid-name
import glob
import os
import time

import cv2
import numpy as np

from SuperPointPretrainedNetwork.demo_superpoint import (PointTracker,
                                                         SuperPointFrontend,
                                                         SuperPointNet)

# pylint: disable=no-member

# Font parameters for visualizaton.
font = cv2.FONT_HERSHEY_DUPLEX
font_clr = (255, 255, 255)
font_pt = (4, 12)
font_sc = 0.4
# Jet colormap for visualization.
myjet = np.array([[0., 0., 0.5],
                  [0., 0., 0.99910873],
                  [0., 0.37843137, 1.],
                  [0., 0.83333333, 1.],
                  [0.30044276, 1., 0.66729918],
                  [0.66729918, 1., 0.30044276],
                  [1., 0.90123457, 0.],
                  [1., 0.48002905, 0.],
                  [0.99910873, 0.07334786, 0.],
                  [0.5, 0., 0.]])


class FrontEnd(object):
    """Save superpoint extracted features to files."""

    def __init__(self, image_directory_path='SuperPointPretrainedNetwork/feature_extraction/undistort_images/', image_extension='*.jpg', image_size=(640, 480), nn_thresh=0.7):
        self.basedir = image_directory_path
        self.img_extension = image_extension
        self.nn_thresh = nn_thresh
        self.fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                     nms_dist=4,
                                     conf_thresh=0.015,
                                     nn_thresh=0.7,
                                     cuda=False)
        self.img_size = image_size

    def read_image(self, impath):
        """ Read image as grayscale.
        Inputs:
            impath - Path to input image.
            img_size - (W, H) tuple specifying resize size.
        Returns:
            grayim - float32 numpy array sized H x W with values in range [0, 1].
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        grayim = (grayim.astype('float32') / 255.)
        return grayim

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Returns:
            superpoint - Nx2 (gtsam.Point2) numpy array of 2D point observations.
            descriptors - Nx256 numpy array of corresponding unit normalized descriptors.

        Refer to /SuperPointPretrainedNetwork/demo_superpoint for more information about the parameters
        Output of SuperpointFrontend.run():
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
        """

        superpoints, descriptors, heatmap = self.fe.run(image)

        return superpoints[:2, ].T, descriptors.T, heatmap

    def get_image_paths(self):
        """Get all image paths within the directory."""
        print('==> Processing Image Directory Input.')
        search = os.path.join(self.basedir, self.img_extension)
        self.img_paths = glob.glob(search)
        self.img_paths.sort()
        print("Number of Images: ", len(self.img_paths))
        maxlen = len(self.img_paths)
        if maxlen == 0:
            raise IOError(
                'No images were found (maybe wrong \'image extension\' parameter?)')

    def extract_all_image_features(self):
        """Extract features for each image within the image path list"""
        for i, impath in enumerate(self.img_paths):
            grayim = self.read_image(impath)
            keypoints, descriptors, heatmap = self.superpoint_generator(grayim)
            self.draw_features(keypoints, grayim, heatmap, i)
            self.save_to_file(keypoints, descriptors, i)

    def save_to_file(self, kp_data, desc_data, index):
        """This is used to save the features information of each frame into a .key file,[x,y, desc(256)]"""
        nrpoints = kp_data.shape[0]
        descriptor_length = 256

        features = [np.hstack((point, desc_data[i]))
                    for i, point in enumerate(kp_data)]
        features = np.array(features)

        dirName = self.basedir+'features/'
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        np.savetxt(dirName+self.leading_zero(index) +
                   '.key', features, fmt='%.4f')

        first_line = str(nrpoints)+' '+str(descriptor_length)+'\n'
        with open(dirName+self.leading_zero(index)+'.key', 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(first_line+content)

    def leading_zero(self, index):
        """Create leading zero filename"""
        index_string = str(index)
        index_len = len(index_string)
        output = ['0' for i in range(7)]
        for i in range(index_len):
            output[7-index_len+i] = index_string[i]
        return ''.join(output)

    def draw_features(self, keypoints, img, heatmap, index):
        """Draw feature images and heatmap images."""
        # Extra output -- Show current point detections.
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        for pt in keypoints:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out1, pt1, 1, (0, 255, 0), -1, lineType=16)
            cv2.putText(out1, 'Raw Point Detections', font_pt,
                        font, font_sc, font_clr, lineType=16)

        # Extra output -- Show the point confidence heatmap.
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min() + .00001)
            out2 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
            out2 = (out2*255).astype('uint8')
        else:
            out2 = np.zeros_like(out1)
            cv2.putText(out2, 'Raw Point Confidences', font_pt,
                        font, font_sc, font_clr, lineType=16)

        out_dir_1 = self.basedir+'feature_image/'
        if not os.path.exists(out_dir_1):
            os.mkdir(out_dir_1)
        out_file_1 = out_dir_1+'frame_%05d' % index+'.jpg'
        print('Writing image to %s' % out_file_1)
        cv2.imwrite(out_file_1, out1)

        out_dir_2 = self.basedir+'heatmap/'
        if not os.path.exists(out_dir_2):
            os.mkdir(out_dir_2)
        out_file_2 = out_dir_2+'heatmap_%05d' % index+'.jpg'
        print('Writing image to %s' % out_file_2)
        cv2.imwrite(out_file_2, out2)


if __name__ == "__main__":
    image_directory_path = 'SuperPointPretrainedNetwork/feature_extraction/undistort_images/'
    image_extension = '*.jpg'
    image_size = (640, 480)
    nn_thresh = 0.7
    front_end = FrontEnd(image_directory_path,
                         image_extension, image_size, nn_thresh)
    front_end.get_image_paths()
    front_end.extract_all_image_features()
