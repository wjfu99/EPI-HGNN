from scipy.special import lambertw
import numpy as np
import random
from datetime import datetime


def planar_laplacian(loc_coordinate, loc_id_dict, loc_tree, eps=0.2*100):
    # original_X, original_Y = zip(*loc_coordinate)
    original_X, original_Y = loc_coordinate[0], loc_coordinate[1]

    # planar laplacian using polar system
    theta = np.pi * 2 * np.random.random()
    p = np.random.random()
    r = -1 / eps * (lambertw((p-1) / np.e, -1) + 1)
    r = r.real

    obfuscate_X = original_X + r * np.cos(theta)
    obfuscate_Y = original_Y + r * np.sin(theta)

    query_pts = np.array([obfuscate_X, obfuscate_Y])

    distance, nearest_id = loc_tree.query(query_pts)
    nearest_pts = loc_tree.data[nearest_id]


    # obfuscate_Loc = [loc_id_dict[tuple(pts)] for pts in nearest_pts]
    obfuscate_Loc = loc_id_dict[tuple(nearest_pts)]
    obfuscate_Loc = random.sample(obfuscate_Loc, 1)[0]
    confidence = np.exp(-distance / eps / 25.0)
    return obfuscate_Loc, confidence