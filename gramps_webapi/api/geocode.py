#
# GenAI - Batch Geocoding Module
#
# Uses Nominatim (OpenStreetMap) to geocode place names.
# Rate limited to 1 request per second per Nominatim usage policy.
#

"""Batch geocoding functions using Nominatim."""

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
from .util import get_db_handle, close_db


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "GenAI-Genealogy/1.0 (batch geocoding for family tree)"
RATE_LIMIT_SECONDS = 1.1  # Slightly over 1 second to be safe


def geocode_place_name(place_name: str) -> Optional[Tuple[float, float]]:
    """
    Geocode a place name using Nominatim.
    
    Args:
        place_name: The place name to geocode (e.g., "Norwich, Connecticut, USA")
        
    Returns:
        Tuple of (latitude, longitude) or None if not found
    """
    if not place_name or not place_name.strip():
        return None
    
    params = urllib.parse.urlencode({
        'q': place_name,
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
        current_app.logger.warning(f"Geocoding failed for '{place_name}': {e}")
    
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
    from ..dbmanager import WebDbManager
    from gramps.gen.db import DbTxn
    
    # Open database with write access
    dbmgr = WebDbManager(tree)
    db_handle = dbmgr.get_db(readonly=False)
    
    try:
        # Get all places
        places = list(db_handle.iter_places())
        total = len(places)
        geocoded = 0
        skipped = 0
        errors = 0
        
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
                errors += 1
                continue
            
            # Geocode
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
            
            # Rate limiting
            time.sleep(RATE_LIMIT_SECONDS)
        
        return {
            'status': 'complete',
            'total': total,
            'updated': geocoded,
            'skipped': skipped,
            'errors': errors,
        }
        
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
