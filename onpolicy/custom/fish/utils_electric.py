import numpy as np

def log_scale_gradients(measured_gradients, min_magnitude=None):
    """
    Apply logarithmic scaling to gradients while preserving directions.

    Parameters:
    -----------
    measured_gradients : np.ndarray shape(..., 2)
        Input gradient vectors
    min_magnitude : float, optional
        Minimum magnitude to use for scaling. If None, uses minimum non-zero value

    Returns:
    --------
    scaled_gradients : np.ndarray same shape as input
        Log-scaled gradient vectors
    """
    # Calculate magnitudes and directions
    magnitudes = np.linalg.norm(measured_gradients, axis=-1)
    directions = measured_gradients / (magnitudes[..., None] + 1e-10)

    # Find scaling reference
    if min_magnitude is None:
        min_magnitude = np.min(magnitudes[magnitudes > 0])

    # Log scale the magnitudes
    scale = 1.0 / min_magnitude
    log_magnitudes = np.log1p(magnitudes * scale)

    # Recombine with directions
    return directions * log_magnitudes[..., None]


def clip_gradients(measured_gradients, percentile=90):
    """
    Clip gradients to specified percentile while preserving directions.

    Parameters:
    -----------
    measured_gradients : np.ndarray shape(..., 2)
        Input gradient vectors
    percentile : float
        Percentile (0-100) to clip at

    Returns:
    --------
    scaled_gradients : np.ndarray same shape as input
        Clipped gradient vectors
    """
    # Calculate magnitudes and directions
    magnitudes = np.linalg.norm(measured_gradients, axis=-1)
    directions = measured_gradients / (magnitudes[..., None] + 1e-10)

    # Clip magnitudes
    max_mag = np.percentile(magnitudes, percentile)
    clipped_magnitudes = np.clip(magnitudes, 0, max_mag)

    # Recombine with directions
    return directions * clipped_magnitudes[..., None]
