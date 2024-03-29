import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
LABELS_CMAP = mcolors.ListedColormap(['white', 'red', 'dodgerblue', 'gold',
                                          'greenyellow', 'blueviolet'])

from .base import CanvasToolBase


__all__ = ['PaintTool']


class PaintTool(CanvasToolBase):
    """Widget for painting on top of a plot.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes where tool is displayed.
    overlay_shape : shape tuple
        2D shape tuple used to initialize overlay image.
    alpha : float (between [0, 1])
        Opacity of overlay
    on_move : function
        Function called whenever a control handle is moved.
        This function must accept the end points of line as the only argument.
    on_release : function
        Function called whenever the control handle is released.
    on_enter : function
        Function called whenever the "enter" key is pressed.
    rect_props : dict
        Properties for :class:`matplotlib.patches.Rectangle`. This class
        redefines defaults in :class:`matplotlib.widgets.RectangleSelector`.

    Attributes
    ----------
    overlay : array
        Overlay of painted labels displayed on top of image.
    label : int
        Current paint color.
    """
    def __init__(self, ax, overlay_shape, radius=5, alpha=0.3, on_move=None,
                 on_release=None, on_enter=None, rect_props=None):
        super(PaintTool, self).__init__(ax, on_move=on_move, on_enter=on_enter,
                                        on_release=on_release)

        props = dict(edgecolor='r', facecolor='0.7', alpha=0.5, animated=True)
        props.update(rect_props if rect_props is not None else {})

        self.alpha = alpha
        self.cmap = LABELS_CMAP
        self._overlay_plot = None
        self.shape = overlay_shape

        self._cursor = plt.Rectangle((0, 0), 0, 0, **props)
        self._cursor.set_visible(False)
        self.ax.add_patch(self._cursor)

        # `label` and `radius` can only be set after initializing `_cursor`
        self.label = 1
        self.radius = radius

        # Note that the order is important: Redraw cursor *after* overlay
        self._artists = [self._overlay_plot, self._cursor]

        self.connect_event('button_press_event', self.on_mouse_press)
        self.connect_event('button_release_event', self.on_mouse_release)
        self.connect_event('motion_notify_event', self.on_move)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        if value >= self.cmap.N:
            raise ValueError('Maximum label value = %s' % len(self.cmap - 1))
        self._label = value
        self._cursor.set_edgecolor(self.cmap(value))

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        self._radius = r
        self._width = 2 * r + 1
        self._cursor.set_width(self._width)
        self._cursor.set_height(self._width)
        self.window = CenteredWindow(r, self._shape)

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, image):
        self._overlay = image
        if image is None:
            self.ax.images.remove(self._overlay_plot)
            self._overlay_plot = None
        elif self._overlay_plot is None:
            props = dict(cmap=self.cmap, alpha=self.alpha,
                         norm=mcolors.no_norm(), animated=True)
            self._overlay_plot = self.ax.imshow(image, **props)
        else:
            self._overlay_plot.set_data(image)
        self.redraw()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        if not self._overlay_plot is None:
            self._overlay_plot.set_extent((-0.5, shape[1] + 0.5,
                                           shape[0] + 0.5, -0.5))
            self.radius = self._radius
        self.overlay = np.zeros(shape, dtype='uint8')

    def _on_key_press(self, event):
        if event.key == 'enter':
            self.callback_on_enter(self.geometry)
            self.redraw()

    def on_mouse_press(self, event):
        if event.button != 1 or not self.ax.in_axes(event):
            return
        self.update_cursor(event.xdata, event.ydata)
        self.update_overlay(event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button != 1:
            return
        self.callback_on_release(self.geometry)

    def on_move(self, event):
        if not self.ax.in_axes(event):
            self._cursor.set_visible(False)
            self.redraw() # make sure cursor is not visible
            return
        self._cursor.set_visible(True)

        self.update_cursor(event.xdata, event.ydata)
        if event.button != 1:
            self.redraw() # update cursor position
            return
        self.update_overlay(event.xdata, event.ydata)
        self.callback_on_move(self.geometry)

    def update_overlay(self, x, y):
        overlay = self.overlay
        overlay[self.window.at(y, x)] = self.label
        # Note that overlay calls `redraw`
        self.overlay = overlay

    def update_cursor(self, x, y):
        x = x - self.radius - 1
        y = y - self.radius - 1
        self._cursor.set_xy((x, y))

    @property
    def geometry(self):
        return self.overlay


class CenteredWindow(object):
    """Window that create slices numpy arrays over 2D windows.

    Examples
    --------
    >>> a = np.arange(16).reshape(4, 4)
    >>> w = CenteredWindow(1, a.shape)
    >>> a[w.at(1, 1)]
    array([[ 0,  1,  2],
           [ 4,  5,  6],
           [ 8,  9, 10]])
    >>> a[w.at(0, 0)]
    array([[0, 1],
           [4, 5]])
    >>> a[w.at(4, 3)]
    array([[14, 15]])
    """
    def __init__(self, radius, array_shape):
        self.radius = radius
        self.array_shape = array_shape

    def at(self, row, col):
        h, w = self.array_shape
        r = self.radius
        xmin = max(0, col - r)
        xmax = min(w, col + r + 1)
        ymin = max(0, row - r)
        ymax = min(h, row + r + 1)
        return [slice(ymin, ymax), slice(xmin, xmax)]


if __name__ == '__main__':
    np.testing.rundocs()
    from skimage import data

    image = data.camera()

    f, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest')
    paint_tool = PaintTool(ax, image.shape)
    plt.show()
