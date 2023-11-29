import math
from tkinter import Tk, Canvas
from turtle import RawTurtle, TurtleScreen, Turtle, Screen

import numpy as np
import plato.draw.matplotlib as draw

_root = Tk()
_root.withdraw()
_canvas = Canvas(master=_root)
_screen = TurtleScreen(_canvas)


def get_turtle(with_screen=False):
    if with_screen:
        screen = Screen()
        screen.setup(600, 600)
        screen.setworldcoordinates(-100, -100, 100, 100)
        rp = Turtle()
    else:
        rp = RawTurtle(_screen)
    rp.speed("slow")
    return rp


def regular_polygon(n: int = 6, length: float = 1, with_screen=False):
    rp = get_turtle(with_screen)
    coords = []
    for _ in range(n):
        rp.forward(length)
        rp.left(360 / n)
        coords.append([rp.xcor(), rp.ycor()])
    return coords


def rectangle(x: float, y: float, with_screen):
    rp = get_turtle(with_screen)
    coords = []
    for length in [x, y, x, y]:
        rp.forward(length)
        rp.left(90)
        coords.append([rp.xcor(), rp.ycor()])
    return coords


def tips_pn_polygon(
        r: float = 3.67,
        d=11.75 / 2,
        w=13.83,
        # h=2.84,
        h=3.75,
        with_screen=False,
):
    """
    2d shape of tips pn, values in angstrom

    :param r: the longest distance between a hydrogen and a silicon in the side group
    :param d: a half of the distance between two silicon atoms
    :param w: the width of pn, defined by two hydrogen atoms along the long axis
    :param h: the height of pn (short axis), defined by
    1. two carbon(2.84), or
    2. one carbon and one hydrogen(3.75) atoms
    :return:
    """
    acetylene_width = 0.2
    rp = get_turtle(with_screen)
    coords = []

    for _ in range(2):
        rp.forward(w / 2 - acetylene_width / 2)
        rp.left(90)
        coords.append([rp.xcor(), rp.ycor()])

        assert d - h / 2 - r > 0  # from pn terminal carbon to the periphery of silyl
        rp.forward(d - h / 2 - r)
        coords.append([rp.xcor(), rp.ycor()])

        rp.left(90)
        silyl_periphery_length = 2 * math.pi * r
        n_silyl_polygon_side = 8
        silyl_polygon_side = silyl_periphery_length / n_silyl_polygon_side
        rp.right(360 / n_silyl_polygon_side)
        for __ in range(n_silyl_polygon_side - 1):
            rp.forward(silyl_polygon_side)
            rp.right(360 / n_silyl_polygon_side)
            coords.append([rp.xcor(), rp.ycor()])

        rp.left(90)
        rp.forward(d - h / 2 - r)
        coords.append([rp.xcor(), rp.ycor()])

        rp.left(90)
        rp.forward(w / 2 - acetylene_width / 2)
        coords.append([rp.xcor(), rp.ycor()])

        rp.right(90)
        rp.forward(h)
        coords.append([rp.xcor(), rp.ycor()])
        rp.right(90)
    coords = np.array([*reversed(coords)])
    center = np.sum(coords, axis=0) / coords.shape[0]
    coords = coords - center
    return coords


def render_plato(vertices, position, orientation):
    primitives = draw.Polygons(vertices=vertices, positions=position[:, :2], orientations=orientation)
    scene = draw.Scene(primitives, zoom=0.1)

    primitives.colors = (.25, .7, .2, 1)

    # the primitives within a Scene object can be iterated over
    for primitive in scene:
        primitive.outline = .1
    return scene


if __name__ == '__main__':
    from shapely.geometry import Polygon

    TIPS_PN_VERTICES = tips_pn_polygon(with_screen=True)
    np.save("vertices_tips_pn_2d.npy", TIPS_PN_VERTICES)
    TIPS_PN_AREA = Polygon(TIPS_PN_VERTICES).area
    print(TIPS_PN_AREA)  # 144.05592434385488
