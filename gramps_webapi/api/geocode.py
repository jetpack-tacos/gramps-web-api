#
# GenAI - Batch Geocoding Module
#
# Uses Nominatim (OpenStreetMap) to geocode place names.
# Rate limited to 1 request per second per Nominatim usage policy.
#

"""Batch geocoding functions using Nominatim."""

import logging
import re
import time
import urllib.parse
import urllib.request
import json
from typing import Optional, Tuple, List, Dict, Any

from celery import Task, shared_task
from flask import current_app
from gramps.gen.db.base import DbReadBase
from gramps.gen.lib import Place

from .tasks import AsyncResult, run_task
from .util import get_db_outside_request, close_db

logger = logging.getLogger(__name__)


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "GenAI-Genealogy/1.0 (batch geocoding for family tree)"
RATE_LIMIT_SECONDS = 1.1  # Slightly over 1 second to be safe


def _nominatim_query(query: str) -> Optional[Tuple[float, float]]:
    """
    Make a single Nominatim API call.

    Returns:
        Tuple of (latitude, longitude) or None if not found.
    """
    if not query or not query.strip():
        return None

    params = urllib.parse.urlencode({
        'q': query,
        'format': 'json',
        'limit': 1,
    })

    url = f"{NOMINATIM_URL}?{params}"

    request = urllib.request.Request(
        url,
        headers={'User-Agent': USER_AGENT}
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data and len(data) > 0:
                return (float(data[0]['lat']), float(data[0]['lon']))
    except Exception as e:
        logger.warning(f"  -> Nominatim request error for '{query}': {e}")

    return None


def _clean_place_name(place_name: str) -> str:
    """
    Clean a place name to improve geocoding success.

    Strips apartment/unit numbers, PO Box prefixes, zip codes,
    suite numbers, and other noise that confuses Nominatim.
    """
    cleaned = place_name

    # Strip apartment/unit/suite designators: "Apt 4", "# 3", "Unit 2B", "Suite 100"
    cleaned = re.sub(r',?\s*(?:Apt|Unit|Suite|Ste)\.?\s*#?\s*\w+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r',?\s*#\s*\d+\w*', '', cleaned)

    # Strip zip codes: "60090-5922" or "53024"
    cleaned = re.sub(r',?\s*\d{5}(-\d{4})?', '', cleaned)

    # Strip "PO Box NNN" — replace entire PO Box reference, keep the city
    cleaned = re.sub(r'^\d*\s*P\.?O\.?\s*Box\s*\d*\s*,?\s*', '', cleaned, flags=re.IGNORECASE)

    # Strip "No. 57" style house numbers in non-address contexts
    cleaned = re.sub(r'\s+No\.?\s*\d+', '', cleaned)

    # Collapse extra commas and whitespace
    cleaned = re.sub(r'\s*,\s*,\s*', ', ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' ,')

    return cleaned


def geocode_place_name(place_name: str) -> Optional[Tuple[float, float]]:
    """
    Geocode a place name using Nominatim with progressive fallback.

    Strategy:
    1. Try the full name as-is.
    2. Try a cleaned version (strip apt numbers, zip codes, PO Box, etc.).
    3. Progressively drop the first comma-separated part and retry,
       so "Business Name, City, State" falls back to "City, State".

    Each attempt respects the Nominatim rate limit.

    Returns:
        Tuple of (latitude, longitude) or None if all attempts fail.
    """
    if not place_name or not place_name.strip():
        return None

    # Attempt 1: exact name as-is
    result = _nominatim_query(place_name)
    if result:
        logger.info(f"  -> Found: ({result[0]}, {result[1]})")
        return result

    # Attempt 2: cleaned name (strip apt, zip, PO Box, etc.)
    cleaned = _clean_place_name(place_name)
    if cleaned and cleaned != place_name:
        time.sleep(RATE_LIMIT_SECONDS)
        result = _nominatim_query(cleaned)
        if result:
            logger.info(f"  -> Found (cleaned '{cleaned}'): ({result[0]}, {result[1]})")
            return result

    # Attempt 3+: progressively drop the first part
    # "Business Name, City, State, Country" -> "City, State, Country" -> "State, Country"
    working = cleaned or place_name
    parts = [p.strip() for p in working.split(',')]

    # Only try fallback if there are multiple parts, and stop when we're
    # down to a single part (don't geocode just "Wisconsin")
    while len(parts) > 2:
        parts = parts[1:]
        fallback = ', '.join(parts)
        time.sleep(RATE_LIMIT_SECONDS)
        result = _nominatim_query(fallback)
        if result:
            logger.info(f"  -> Found (fallback '{fallback}'): ({result[0]}, {result[1]})")
            return result

    logger.warning(f"  -> All attempts failed for '{place_name}'")
    return None


def get_place_display_name(db_handle: DbReadBase, place: Place) -> str:
    """
    Build a display name for a place including hierarchy.
    
    Args:
        db_handle: Database handle
        place: Place object
        
    Returns:
        Comma-separated place name with hierarchy (e.g., "Norwich, Connecticut, USA")
    """
    names = []
    
    # Get the primary name
    if place.name and place.name.value:
        names.append(place.name.value)
    
    # Walk up the place hierarchy
    current_place = place
    seen_handles = {place.handle}
    
    while current_place.placeref_list:
        parent_ref = current_place.placeref_list[0]
        parent_handle = parent_ref.ref
        
        if parent_handle in seen_handles:
            break  # Avoid infinite loops
        seen_handles.add(parent_handle)
        
        try:
            parent_place = db_handle.get_place_from_handle(parent_handle)
            if parent_place and parent_place.name and parent_place.name.value:
                names.append(parent_place.name.value)
            current_place = parent_place
        except Exception:
            break
    
    return ", ".join(names)


@shared_task(bind=True)
def geocode_all_places_task(
    self,
    tree: str,
    user_id: str,
    task: Optional[Task] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Celery task to geocode all places in the database.
    
    Args:
        tree: Tree name
        user_id: User ID
        task: Celery task for progress updates
        skip_existing: If True, skip places that already have coordinates
        
    Returns:
        Dict with geocoded count and error count
    """
    from gramps.gen.db import DbTxn

    logger.info("=" * 60)
    logger.info(f"BATCH GEOCODING STARTED - tree={tree}, user={user_id}, skip_existing={skip_existing}")
    logger.info("=" * 60)

    # Open database with write access (same pattern as all other Celery tasks)
    db_handle = get_db_outside_request(
        tree=tree, view_private=True, readonly=False, user_id=user_id
    )

    try:
        # Get all places
        places = list(db_handle.iter_places())
        total = len(places)
        geocoded = 0
        skipped = 0
        errors = 0
        error_names = []

        logger.info(f"Found {total} places in database")

        for i, place in enumerate(places):
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': i + 1,
                    'total': total,
                    'updated': geocoded,
                    'skipped': skipped,
                    'errors': errors,
                }
            )

            # Skip if already has coordinates
            if skip_existing and place.lat and place.long:
                skipped += 1
                continue

            # Build place name
            place_name = get_place_display_name(db_handle, place)

            if not place_name:
                logger.warning(f"[{i+1}/{total}] Skipping place {place.gramps_id} — no name")
                errors += 1
                continue

            logger.info(f"[{i+1}/{total}] Geocoding: '{place_name}' (ID: {place.gramps_id})")

            # Geocode (includes its own rate limiting for fallback retries)
            coords = geocode_place_name(place_name)

            if coords:
                lat, lon = coords
                # Update place with coordinates
                with DbTxn("Geocode place", db_handle) as trans:
                    place.lat = str(lat)
                    place.long = str(lon)
                    db_handle.commit_place(place, trans)
                geocoded += 1
            else:
                errors += 1
                error_names.append(place_name)

            # Rate limiting between places (the first Nominatim call for the next place)
            time.sleep(RATE_LIMIT_SECONDS)

        logger.info("=" * 60)
        logger.info(f"BATCH GEOCODING COMPLETE")
        logger.info(f"  Total:    {total}")
        logger.info(f"  Updated:  {geocoded}")
        logger.info(f"  Skipped:  {skipped} (already had coordinates)")
        logger.info(f"  Errors:   {errors}")
        if error_names:
            logger.info(f"  Failed places: {error_names[:20]}{'...' if len(error_names) > 20 else ''}")
        logger.info("=" * 60)

        return {
            'status': 'complete',
            'total': total,
            'updated': geocoded,
            'skipped': skipped,
            'errors': errors,
        }

    except Exception as e:
        logger.error(f"BATCH GEOCODING FAILED: {e}", exc_info=True)
        raise

    finally:
        close_db(db_handle)


# Register as Celery task
def geocode_places(
    tree: str,
    user_id: str,
    skip_existing: bool = True,
):
    """Entry point for geocoding task."""
    return run_task(
        geocode_all_places_task,
        tree=tree,
        user_id=user_id,
        skip_existing=skip_existing,
    )
