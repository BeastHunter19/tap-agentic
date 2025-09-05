import os
from typing import Any, Dict, List, Literal, Optional, Sequence
from urllib.parse import quote

import httpx
from google.maps import places_v1, routing_v2
from google.type import latlng_pb2
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.utils import get_line_distances

"""Tools related to maps and geolocation.

The custom tools use official Google Maps Platform Python clients and are
designed to be clear for ReAct-style agents (explicit params and behaviors).
"""

# -----------------------------
# Google Maps clients & helpers
# -----------------------------

_places_client = None
_routes_client = None


def _get_api_key() -> str:
    """Return the Google Maps API key from env or local secrets.

    Looks for `GOOGLE_MAPS_API_KEY` env var first, then falls back to
    reading from the `MAPS_API_KEY_FILE` file.
    """
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if key:
        return key
    # Fallback to local secrets file
    try:
        key_file = os.getenv("MAPS_API_KEY_FILE")
        with open(key_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as exc:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY must be set (env) or MAPS_API_KEY_FILE must point to a valid file"
        ) from exc


def _get_places_client():
    """Lazily create the Places (New) async client."""
    global _places_client
    if _places_client is None:
        _places_client = places_v1.PlacesAsyncClient(
            client_options={"api_key": _get_api_key()}
        )
    return _places_client


def _get_routes_client():
    """Lazily create the Routes async client."""
    global _routes_client
    if _routes_client is None:
        _routes_client = routing_v2.RoutesAsyncClient(
            client_options={"api_key": _get_api_key()}
        )
    return _routes_client


# No typed client for Geocoding v4 among google client libraries; we use REST with httpx.


def _places_fieldmask_for_nearby() -> str:
    """Field mask for Places SearchNearby response (bill only what you use)."""
    fields = [
        "id",
        "displayName",
        "location",
        # only for debugging:
        "primaryType",
        "types",
    ]
    return ",".join(f"places.{f}" for f in fields)


def _routing_fieldmask_for_matrix() -> str:
    """Minimal field mask for ComputeRouteMatrix to rank by nearest."""
    fields = [
        "originIndex",
        "destinationIndex",
        "distanceMeters",
        "duration",
        "condition",
        "status",
    ]
    return ",".join(fields)


def _places_fieldmask_for_details() -> str:
    """Field mask for Place Details of the selected supermarkets."""
    fields = [
        "formattedAddress",
        "currentOpeningHours.openNow",
        "googleMapsUri",
        "viewport",
    ]
    return ",".join(f"places.{f}" for f in fields)


def _geocode_fieldmask_for_address() -> str:
    """Minimal field mask for GeocodeAddress results."""
    fields = [
        "results.location",
    ]
    return ",".join(fields)


# -----------------------------
# Custom tools
# -----------------------------


# ---- Pydantic schemas for tool arguments ----


class Location(BaseModel):
    """Latitude/longitude coordinates."""

    lat: float = Field(description="Latitude in decimal degrees.")
    lng: float = Field(description="Longitude in decimal degrees.")


class FindNearbySupermarketsInput(BaseModel):
    """Inputs for searching nearby supermarkets using Places API (New)."""

    user_location: Location = Field(description="User location coordinates.")
    radius_meters: int = Field(
        1500,
        description="Search radius in meters (1–50000). Values outside this range are clamped.",
    )
    max_results: int = Field(
        20,
        description="Maximum number of places to return (1–20). Values outside this range are clamped.",
    )
    language_code: Optional[str] = Field(
        default=None,
        description="Optional BCP-47 language code for localized names/addresses.",
    )
    region_code: Optional[str] = Field(
        default=None,
        description="Optional CLDR region code for regional formatting/bias.",
    )


class GetAccurateSupermarketDistancesInput(BaseModel):
    """Inputs for computing distances/durations to supermarkets using Routes API."""

    user_location: Location = Field(description="User location coordinates.")
    destinations: List[Location] = Field(
        description="List of destination coordinates (lat/lng)."
    )
    travel_mode: Literal["driving", "walking", "bicycling"] = Field(
        "driving",
        description='Travel mode to use. One of "driving", "walking", or "bicycling".',
    )
    units: Literal["metric", "imperial"] = Field(
        "metric",
        description='Unit system for distances. "metric" returns meters; "imperial" returns feet/miles.',
    )


class GetSupermarketDetailsInput(BaseModel):
    """Inputs for fetching minimal details of a supermarket place."""

    place_id_or_name: str = Field(
        description="Place ID (e.g., 'ChIJ...') or resource name (e.g., 'places/ChIJ...')."
    )
    language_code: Optional[str] = Field(
        default=None, description="Optional BCP-47 language code for localized data."
    )
    region_code: Optional[str] = Field(
        default=None,
        description="Optional CLDR region code for regional formatting/bias.",
    )


class GeocodeAddressInput(BaseModel):
    """Inputs for forward geocoding an address into coordinates."""

    address: str = Field(description="Indirizzo testuale da geocodificare.")
    language_code: Optional[str] = Field(
        default=None, description="Codice lingua BCP-47 per i risultati."
    )
    region_code: Optional[str] = Field(
        default=None, description="Codice regione CLDR per il bias regionale."
    )


@tool(args_schema=FindNearbySupermarketsInput)
async def find_nearby_supermarkets(
    user_location: Location,
    radius_meters: int = 1500,
    max_results: int = 10,
    language_code: Optional[str] = None,
    region_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Find supermarkets near the user's location using Places API (New).

    This tool searches for nearby places of type "supermarket" around the given
    coordinates within a circular area. It returns key details needed to match
    the supermarket to offers and compute distances. The results are ordered
    by ascending distance from the user's location (the distance is just a
    straight line, not travel distance). You can use this distance ordering
    as a distance estimate and avoid calling get_accurate_supermarket_distances unless
    you need to decide between a few close candidates and need precise travel distances.

    Returns:
        List of supermarkets with fields:
        - id, name, location {lat, lng}.
    """

    client = _get_places_client()

    # Clamp values to API-supported ranges
    radius = max(1, min(int(radius_meters), 50000))
    max_count = max(1, min(int(max_results), 20))

    # Location restriction (circle)
    center = latlng_pb2.LatLng(
        latitude=float(user_location.lat), longitude=float(user_location.lng)
    )
    location_restriction = places_v1.SearchNearbyRequest.LocationRestriction(
        circle=places_v1.Circle(center=center, radius=radius)
    )

    req_kwargs = {
        "location_restriction": location_restriction,
        "included_primary_types": ["supermarket"],
        "max_result_count": max_count,
        # The results are always ranked by ascending (line) distance from the center
        "rank_preference": places_v1.SearchNearbyRequest.RankPreference.DISTANCE,
    }
    if language_code:
        req_kwargs["language_code"] = language_code
    if region_code:
        req_kwargs["region_code"] = region_code
    req = places_v1.SearchNearbyRequest(**req_kwargs)

    # Call API with API key and field mask headers
    fieldmask = _places_fieldmask_for_nearby()
    resp = await client.search_nearby(
        request=req,
        retry=None,
        timeout=30.0,
        metadata=[("x-goog-fieldmask", fieldmask)],
    )

    places = getattr(resp, "places", []) or []
    place_locations = [getattr(p, "location", None) for p in places]
    line_distances = get_line_distances(user_location, place_locations)

    results: List[Dict[str, Any]] = []
    for p, dist in zip(places, line_distances):
        loc = getattr(p, "location", None)
        results.append(
            {
                "id": getattr(p, "id", None),
                "name": getattr(p, "display_name", None),
                "location": {
                    "lat": getattr(loc, "latitude", None),
                    "lng": getattr(loc, "longitude", None),
                }
                if loc
                else None,
                "line_distance_km": dist,
            }
        )

    return results


@tool(args_schema=GetAccurateSupermarketDistancesInput)
async def get_accurate_supermarket_distances(
    user_location: Location,
    destinations: Sequence[Location],
    travel_mode: Literal["driving", "walking", "bicycling"] = "driving",
    units: Literal["metric", "imperial"] = "metric",
) -> List[Dict[str, Any]]:
    """Compute distances and durations from the user to supermarkets.

    Uses Routes API ComputeRouteMatrix with the user location as a single origin
    and a list of destination supermarkets. This is ideal for finding the
    closest supermarket (by distance or duration) based on the chosen travel mode.
    Only use this tool after you've shortlisted candidates and if you need to
    decide between a few candidates with similar offer coverage, otherwise you
    can just rely on the straight-line distance ordering from the nearby search.
    The results are returned in ascending order of (spatial) distance.

    Returns:
        A list of elements, each containing:
        - destination_index: index in the input destinations list
        - distance_meters: integer distance in meters
        - duration: RFC3339 duration string (e.g., "123s")
        - condition: string status (e.g., "ROUTE_EXISTS", "ROUTE_NOT_FOUND")
        - status: element-level status code if present
    """

    client = _get_routes_client()

    # Origin (user)
    origin_latlng = latlng_pb2.LatLng(
        latitude=float(user_location.lat), longitude=float(user_location.lng)
    )
    origins = [
        routing_v2.RouteMatrixOrigin(
            waypoint=routing_v2.Waypoint(
                location=routing_v2.Location(lat_lng=origin_latlng)
            )
        )
    ]

    # Destinations
    route_dests: List[routing_v2.RouteMatrixDestination] = []
    for d in destinations:
        latlng = latlng_pb2.LatLng(latitude=float(d.lat), longitude=float(d.lng))
        waypoint = routing_v2.Waypoint(location=routing_v2.Location(lat_lng=latlng))
        route_dests.append(routing_v2.RouteMatrixDestination(waypoint=waypoint))

    if not route_dests:
        return []

    req = routing_v2.ComputeRouteMatrixRequest(
        origins=origins,
        destinations=route_dests,
        travel_mode=travel_mode,
        units=units,
    )

    # Call API (streaming response)
    fieldmask = _routing_fieldmask_for_matrix()
    stream = client.compute_route_matrix(
        request=req,
        retry=None,
        timeout=60.0,
        metadata=[("x-goog-fieldmask", fieldmask)],
    )

    results: List[Dict[str, Any]] = []
    async for element in stream:
        di = getattr(element, "destination_index", None)
        res = {
            "destination_index": di,
            "distance_meters": getattr(element, "distance_meters", None),
            "duration": getattr(element, "duration", None),
            "condition": getattr(element, "condition", None),
            "status": getattr(element, "status", None),
        }
        results.append(res)

    # Sort by distance (ascending) for convenience
    results.sort(key=lambda x: (x.get("distance_meters") or float("inf")))
    return results


@tool(args_schema=GeocodeAddressInput)
async def geocode_address(
    address: str,
    language_code: Optional[str] = None,
    region_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Geocodes an address into coordinates (latitude and longitude) using Geocoding API.

    Returns:
        List of results: location {lat,lng}.
    """

    # GET https://geocode.googleapis.com/v4beta/geocode/address/{addressQuery}
    base_url = "https://geocode.googleapis.com/v4beta/geocode/address"
    address_query = quote(address, safe="")
    url = f"{base_url}/{address_query}"

    headers = {
        "X-Goog-Api-Key": _get_api_key(),
        "X-Goog-FieldMask": _geocode_fieldmask_for_address(),
    }
    params: Dict[str, str] = {}
    if language_code:
        params["languageCode"] = language_code
    if region_code:
        params["regionCode"] = region_code

    async with httpx.AsyncClient(timeout=30.0) as http:
        r = await http.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()

    out: List[Dict[str, Any]] = []
    for res in data.get("results", []) or []:
        loc = res.get("location") or {}
        out.append(
            {
                "location": {
                    "lat": loc.get("latitude"),
                    "lng": loc.get("longitude"),
                }
                if loc
                else None,
            }
        )

    return out


@tool(args_schema=GetSupermarketDetailsInput)
async def get_supermarket_details(
    place_id_or_name: str,
    language_code: Optional[str] = None,
    region_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Get minimal details for a supermarket using Places API (New).

    Use this after you've shortlisted the best candidate(s) to fetch fields that
    are billed at higher tiers. This avoids requesting these fields for every result in Nearby Search.

    Returns:
        A dictionary with fields:
        - address, open_now, google_maps_uri, viewport {low:{lat,lng}, high:{lat,lng}}.
    """
    client = _get_places_client()

    # Normalize to resource name
    name = place_id_or_name.strip()
    if not name.startswith("places/"):
        name = f"places/{name}"

    details_kwargs = {"name": name}
    if language_code:
        details_kwargs["language_code"] = language_code
    if region_code:
        details_kwargs["region_code"] = region_code
    req = places_v1.GetPlaceRequest(**details_kwargs)

    fieldmask = _places_fieldmask_for_details()
    resp = await client.get_place(
        request=req,
        retry=None,
        timeout=30.0,
        metadata=[("x-goog-fieldmask", fieldmask)],
    )

    p = getattr(resp, "place", None)
    if not p:
        return {}

    current_open = None
    if getattr(p, "current_opening_hours", None) and hasattr(
        p.current_opening_hours, "open_now"
    ):
        current_open = p.current_opening_hours.open_now
    vp = getattr(p, "viewport", None)
    low = getattr(vp, "low", None) if vp else None
    high = getattr(vp, "high", None) if vp else None

    return {
        "address": getattr(p, "formatted_address", None),
        "open_now": current_open,
        "google_maps_uri": getattr(p, "google_maps_uri", None),
        "viewport": {
            "low": {
                "lat": getattr(low, "latitude", None),
                "lng": getattr(low, "longitude", None),
            }
            if low
            else None,
            "high": {
                "lat": getattr(high, "latitude", None),
                "lng": getattr(high, "longitude", None),
            }
            if high
            else None,
        }
        if vp
        else None,
    }
