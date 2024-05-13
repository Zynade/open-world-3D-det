import torch
import numpy as np
import random


def get_medoid(points: torch.Tensor) -> torch.Tensor:
    """
    Get the medoid of a set of points.
    :param points: torch.Tensor of shape (3, N)
    :return: torch.Tensor of shape (3,)
    """
    dist_matrix = torch.cdist(points.T, points.T, p=2)

    return torch.argmin(dist_matrix.sum(axis=0))

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)