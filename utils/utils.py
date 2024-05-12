import torch


def get_medoid(points: torch.Tensor) -> torch.Tensor:
    """
    Get the medoid of a set of points.
    :param points: torch.Tensor of shape (3, N)
    :return: torch.Tensor of shape (3,)
    """
    dist_matrix = torch.cdist(points.T, points.T, p=2)

    return torch.argmin(dist_matrix.sum(axis=0))