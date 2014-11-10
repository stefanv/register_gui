from skimage import io, transform, color

from viewer import ImageViewer
from viewer.plugins import Plugin
from viewer.canvastools import LineTool

import numpy as np
import matplotlib.pyplot as plt

import sys


def in_interval(interval, v):
    return v >= interval[0] and v < interval[1]


class CorrespondenceSelector(Plugin):
    def __init__(self, image_shapes, maxdist=10, n=4, zoom_radius=10, **kwargs):
        """
        Parameters
        ----------
        image_shapes : tuple of shapes
            The shapes of the two side-by-side images.  E.g.,
            [(100, 50), (200, 100)]
        n : int
            Initial number of correspondences.
        zoom_radius : int
            Size of zoom window.
        """
        super(CorrespondenceSelector, self).__init__(**kwargs)
        self._maxdist = maxdist

        (y0, x0), (y1, x1) = (s[:2] for s in image_shapes)
        self._xlim_start = (0, x0 - 1)
        self._ylim_start = (0, y0 - 1)
        self._xlim_end = (x0, x0 + x1 - 1)
        self._ylim_end = (0, y1)

        self._lines = []
        self._n = n

        self._zoom_radius = zoom_radius
        self._zoom = np.zeros((2 * zoom_radius, 4 * zoom_radius))

        print(self.help())

    def attach(self, image_viewer):
        super(CorrespondenceSelector, self).attach(image_viewer)

        self.image_viewer.canvas.mpl_connect('key_press_event', self._on_key_press)

        try:
            coords = np.load('_reg_coords.npz')
        except IOError:
            n = self._n
            for i in range(n):
                end_points = np.array([
                    [10 + 30 * n, 10 + 30 * i],
                    [self._xlim_end[0] + 30 * n, 20 + 30 * i]])
                self.add_line(end_points=end_points)
        else:
            source = coords['source']
            target = coords['target']
            source[:, 0] += self._xlim_end[0]
            end_points = np.array(zip(target, source))
            for ep in end_points:
                self.add_line(end_points=ep)

    def help(self):
        helpstr = ("Correspondence selector tool",
                   "+: Add new correspondence",
                   "",
                   "Select ends and drag to adjust",
                   "Press 'delete' while dragging to remove a correspondence")
        return '\n'.join(helpstr)

    def add_line(self, end_points=None):
        N = len(self._lines)
        line_changed = lambda end_points: self._line_changed(N, end_points)
        line_released = lambda where: self._line_released(N, where)

        line = LineTool(self.image_viewer.ax,
                        maxdist=self._maxdist,
                        on_move=line_changed,
                        on_release=line_released,
                        line_props=dict(linewidth=2),
                        nograb_draw=False, useblit=False)

        line._handles._markers.set_markeredgecolor('g')
        line._handles._markers.set_markersize(20)
        line._handles._markers.set_markeredgewidth(1)
        line._handles._markers.set_markerfacecolor('r')
        line._handles._markers.set_alpha(0.2)


        if end_points is None:
            end_points = np.array([
                [self._xlim_start[1] / 2,
                  self._ylim_start[1]/2],
                [(self._xlim_end[0] + self._xlim_end[1])/2,
                  self._ylim_end[1]/2]])
        line.valid = False

        self._lines.append(line)
        self._line_changed(N, end_points)
        self.artists.append(line)

    def _on_key_press(self, event):
        if event.key == "+":
            self.add_line()
        elif event.key == "delete":
            badlines = [l for l in self._lines if l._active_pt is not None]
            self._lines = [l for l in self._lines if l._active_pt is None]
            for l in badlines:
                l.remove()
            self.image_viewer.redraw()

    def _line_changed(self, N, end_points):
        x, y = np.transpose(end_points)

        if x[0] > x[1]:
            x = x[::-1]
            y = y[::-1]

        line = self._lines[N]

        if not (in_interval(self._xlim_start, x[0]) and
                in_interval(self._ylim_start, y[0]) and
                in_interval(self._xlim_end, x[1]) and
                in_interval(self._ylim_end, y[1])):

            line._line.set_color('r')
            line.valid = False
            return

        line._line.set_color('b')
        line.valid = True
        line.end_points = end_points

        r = self._zoom_radius
        image = self.image_viewer.original_image

        # Grab a slice out of the original image around each endpoint
        zoom_shape = (2 * r, 2 * r)
        _z0 = image[max(y[0]-r, 0):min(y[0]+r, self._ylim_end[1]),
                    max(x[0]-r, 0):min(x[0]+r, self._xlim_end[1])]
        _z1 = image[max(y[1]-r, 0):min(y[1]+r, self._ylim_end[1]),
                    max(x[1]-r, 0):min(x[1]+r, self._xlim_end[1])]

        if _z0.shape != zoom_shape:
            _z0 = np.zeros(zoom_shape)

        if _z1.shape != zoom_shape:
            _z1 = np.zeros(zoom_shape)

        delta = np.abs(_z0 - _z1)

        self._zoom = np.hstack((_z0, _z1, delta))

    def _line_released(self, N, where):
        self.output_changed.emit(self.output())

    def output(self):
        end_points = np.array([l.end_points for l in self._lines])
        end_points[..., 1::2, 0] -= self._xlim_end[0]
        return self._zoom, (all(l.valid for l in self._lines), end_points)


class WarpPlugin(Plugin):
    def __init__(self, images, correspondence_plugin,
                 **kwargs):
        super(WarpPlugin, self).__init__(image_filter=self.warp,
                                         **kwargs)
        self._images = images
        self._plugin = correspondence_plugin
        self._display_mode = "normal"  # or "diff" or "source" or "target"
        self._active = True
        self._full_extents = False

        self._tf0 = None
        self._tf1 = None

        print(self.help())

    def attach(self, image_viewer):
        super(WarpPlugin, self).attach(image_viewer)
        image_viewer.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        key = event.key.lower()
        if key == "c":
            print("Saving coordinates...")
            np.savez('_reg_coords.npz', source=self._source, target=self._target)
        elif key == "d":
            if self._display_mode == "normal":
                self._display_mode = "diff"
            else:
                self._display_mode = "normal"
            self.filter_image()
        elif key == "1":
            self._display_mode = "source"
            self.filter_image()
        elif key == "2":
            self._display_mode = "target"
            self.filter_image()
        elif key == "e":
            print("Saving warped images...")
            plt.imsave('warped0.png', self._warped0)
            plt.imsave('warped1.png', self._warped1)
        elif key == "q":
            self._active = False
            print("Disabled rendering")
        elif key == "a":
            self._active = True
            print("Activated rendering")
        elif key == "p":
            print("Transformation matrices:")
            print(self._tf0.params)
            print(self._tf1.params)
        elif key == "f":
            self._full_extents = not self._full_extents
            print("Toggling full extents:", self._full_extents)
            self.filter_image()

    def help(self):
        helpstr = ("Focus on the warped view, then:",
                   "C: Save coordinates to file",
                   "D: Show difference instead of average",
                   "1: Show first frame",
                   "2: Show second frame (warped to fit on first)",
                   "E: Export warped images",
                   "F: Toggle [F]ull export extents--no cropping when exporting",
                   "Q: Disable rendering",
                   "A: Enable rendering",
                   "P: Print transformation matrix",
                   "")
        return "\n".join(helpstr)

    def warp(self, image):
        # Estimate the transformation between the two sets of coordinates,
        # assuming it is an affine transform
        _, (valid, coords) = self._plugin.output()

        if valid and self._active:
            target = coords[:, ::2].squeeze()
            source = coords[:, 1::2].squeeze()

            tf = transform.estimate_transform('similarity', source, target)

            r0, c0 = self._images[0].shape[:2]
            r1, c1 = self._images[1].shape[:2]
            corners_first = np.array([[0, 0],
                                      [0, r0],
                                      [c0, 0],
                                      [c0, r0]])
            corners_second = np.array([[0, 0],
                                       [0, r1],
                                       [c1, 0],
                                       [c1, r1]])

            if self._full_extents:
                corners_second_warped = tf(corners_second)
                all_corners = np.vstack((corners_first, corners_second_warped))

                corner_min = np.min(all_corners, axis=0)
                corner_max = np.max(all_corners, axis=0)

                output_shape = corner_max - corner_min
                output_shape = np.ceil(output_shape[::-1])

                offset = transform.SimilarityTransform(translation=-corner_min)
            else:
                output_shape = (r0, c0)
                offset = transform.SimilarityTransform()

            img0_warped = transform.warp(images[0], inverse_map=(offset).inverse,
                                         output_shape=output_shape)

            img1_warped = transform.warp(images[1], inverse_map=(tf + offset).inverse,
                                         output_shape=output_shape)

            self._tf0 = offset
            self._tf1 = tf + offset

            mode = self._display_mode
            if mode == "diff":
                registered = np.abs(img0_warped - img1_warped)
            elif mode == "normal":
                mask = (img0_warped != 0) & (img1_warped != 0)
                registered = img0_warped + img1_warped
                registered[mask] /= 2
            elif mode == "source":
                registered = img0_warped
            elif mode == "target":
                registered = img1_warped

            self._source = source
            self._target = target

            self._warped0 = img0_warped
            self._warped1 = img1_warped

            return registered
        else:
            return image



if len(sys.argv) == 1:
    images = io.ImageCollection('data/webreg_*.jpg')
else:
    images = [io.imread(sys.argv[1]), io.imread(sys.argv[2])]

y0, x0 = images[0].shape[:2]
y1, x1 = images[1].shape[:2]

image = np.zeros((max(y0, y1), x0 + x1))
image[:y0, :x0] = color.rgb2gray(images[0])
image[:y1, x0:] = color.rgb2gray(images[1])
both_shapes = (images[0].shape, images[1].shape)

viewer = ImageViewer(image)
R = 10
selector_plugin = CorrespondenceSelector(both_shapes, n=4, zoom_radius=R)
viewer += selector_plugin

warp_viewer = ImageViewer(selector_plugin)
warp_plugin = WarpPlugin(images, selector_plugin)
warp_viewer += warp_plugin

zoom_viewer = ImageViewer(selector_plugin)

zoom_plugin = CorrespondenceSelector([(2*R, 2*R), (2*R, 2*R)], n=1)
zoom_viewer += zoom_plugin
zoom_plugin.add_line(end_points=[[R, R], [3*R, R]])

viewer.show()
