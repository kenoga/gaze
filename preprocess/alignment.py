import cv2
import numpy as np
from shapely.geometry import Point, Polygon


class Alignment:

    def __init__(self, dim, template, triangles):

        template = np.asarray(template)

        self.dim = dim

        self.template = template

        self.triangles = triangles

        self.masks = [np.array([[[1] * 3 if Point(x, y).intersects(Polygon(dim * template[triangle])) else [0] * 3
                                 for x in range(dim)] for y in range(dim)], dtype=np.uint8) for triangle in triangles]

    def align(self, image, landmarks):

        landmarks = np.asarray(landmarks)

        alignedImage = np.zeros((self.dim, self.dim, 3), image.dtype)

        for triangle, mask in zip(self.triangles, self.masks):

            minPosition = landmarks[triangle].min(0)
            maxPosition = landmarks[triangle].max(0)

            affineTransform = cv2.getAffineTransform(
                src=np.apply_along_axis(
                    func1d=lambda position: position - minPosition,
                    axis=1,
                    arr=landmarks[triangle]
                ).astype(self.template.dtype),
                dst=self.dim * self.template[triangle]
            )

            warpedImage = cv2.warpAffine(
                src=image[minPosition[1]:maxPosition[1]+1, minPosition[0]:maxPosition[0]+1],
                M=affineTransform,
                dsize=(self.dim, self.dim)
            )

            alignedImage += warpedImage * mask

        return alignedImage
