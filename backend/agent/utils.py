"""
Utility functions for geospatial calculations and other agent helpers.
"""

from haversine import Unit, haversine_vector


def normalize_location(loc: object) -> tuple[float, float]:
    """
    Normalize a location to a (lat, lng) tuple.

    Accepts:
        - tuple: (lat, lng)
        - dict: {"lat": ..., "lng": ...} or {"latitude": ..., "longitude": ...}
        - object: with lat/lng or latitude/longitude attributes

    Returns:
        tuple[float, float]: (lat, lng)

    Raises:
        ValueError: if location format is not recognized.
    """
    if isinstance(loc, tuple) and len(loc) == 2:
        return loc
    if isinstance(loc, dict):
        lat = loc.get("lat")
        lng = loc.get("lng")
        if lat is None or lng is None:
            lat = loc.get("latitude")
            lng = loc.get("longitude")
        if lat is not None and lng is not None:
            return (lat, lng)
    # Try to get lat/lng or latitude/longitude attributes
    lat = getattr(loc, "lat", None)
    lng = getattr(loc, "lng", None)
    if lat is None or lng is None:
        lat = getattr(loc, "latitude", None)
        lng = getattr(loc, "longitude", None)
    if lat is not None and lng is not None:
        return (lat, lng)
    raise ValueError(f"Invalid location format: {loc}")


def get_line_distances(origin, destinations) -> list[float]:
    """
    Calculate the Haversine distance between an origin and multiple destination points.

    Args:
        origin: The origin location. Can be:
            - tuple: (lat, lng)
            - dict: {"lat": ..., "lng": ...} or {"latitude": ..., "longitude": ...}
            - object: with lat/lng or latitude/longitude attributes
        destinations: A list of locations, accepts the same formats as `origin`.

    Returns:
        list[float]: A list of distances in kilometers from the origin to each destination.
    """
    if not destinations:
        return []

    norm_destinations = [normalize_location(d) for d in destinations]
    norm_origins = [normalize_location(origin)] * len(norm_destinations)
    distances = haversine_vector(
        norm_origins, norm_destinations, unit=Unit.KILOMETERS, check=True
    )
    return distances.tolist()
