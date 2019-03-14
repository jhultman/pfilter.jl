import numpy as np
from imageio import imread
from skimage import feature

def find_verts(edges):
    is_edge = edges[::-1, ::-1].T > 0
    verts = np.stack(np.where(is_edge), -1)   
    return verts

def _order_vertices(verts, ccw=+1):
    assert ccw in (-1, +1)
    vectors = verts - np.median(verts, 0)
    angles = np.arctan2(*vectors[:, ::-1].T)
    order = np.argsort(ccw * angles)
    verts_ccw = verts[order][::ccw]
    return verts_ccw

def order_vertices(verts):
    "Approximate CCW ordering."
    verts_a = _order_vertices(verts, +1)
    verts_b = _order_vertices(verts, -1)
    agree = (verts_a == verts_b).all(1)
    verts_ccw = verts_a[agree]
    return verts_ccw

def main():
    img = imread('../data/airplane.png').mean(-1)
    edges = feature.canny(img, sigma=4)
    verts = order_vertices(find_verts(edges))
    verts = verts[::20] / verts.max()
    np.savetxt('../data/airplane_verts.txt', verts)

if __name__ == '__main__':
    main()
