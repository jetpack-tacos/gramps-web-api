#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      Jer Olson
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""AI person insight endpoints."""

import uuid
from datetime import datetime

from flask import current_app
from flask_jwt_extended import get_jwt_identity
from gramps.gen.errors import HandleError

from ..llm import generate_insight
from ..search.text import obj_strings_from_object
from ..search.text_semantic import (
    date_to_text,
    name_to_text,
    event_to_line,
)
from ..util import (
    abort_with_message,
    check_quota_ai,
    get_db_outside_request,
    get_logger,
    get_tree_from_jwt_or_fail,
    update_usage_ai,
)
from . import ProtectedResource
from ...auth import PersonInsight, user_db
from ...auth.const import PERM_USE_CHAT, PERM_VIEW_PRIVATE
from ..auth import has_permissions, require_permissions


def _person_brief(person, db_handle, include_private=True):
    """Get a brief summary line for a person (name, born, died)."""
    name = name_to_text(person.primary_name)
    gramps_id = person.gramps_id

    birth_str = ""
    try:
        if person.birth_ref_index >= 0:
            birth_ref = person.event_ref_list[person.birth_ref_index]
            birth_event = db_handle.get_event_from_handle(birth_ref.ref)
            if not include_private and (birth_ref.private or birth_event.private):
                pass
            elif birth_event.date and not birth_event.date.is_empty():
                birth_str = date_to_text(birth_event.date)
                if birth_event.place:
                    try:
                        place = db_handle.get_place_from_handle(birth_event.place)
                        place_name = place.get_title()
                        if place_name:
                            birth_str += f", {place_name}"
                    except HandleError:
                        pass
    except (IndexError, HandleError):
        pass

    death_str = ""
    try:
        if person.death_ref_index >= 0:
            death_ref = person.event_ref_list[person.death_ref_index]
            death_event = db_handle.get_event_from_handle(death_ref.ref)
            if not include_private and (death_ref.private or death_event.private):
                pass
            elif death_event.date and not death_event.date.is_empty():
                death_str = date_to_text(death_event.date)
                if death_event.place:
                    try:
                        place = db_handle.get_place_from_handle(death_event.place)
                        place_name = place.get_title()
                        if place_name:
                            death_str += f", {place_name}"
                    except HandleError:
                        pass
    except (IndexError, HandleError):
        pass

    parts = [f"{name} [{gramps_id}]"]
    if birth_str:
        parts.append(f"born {birth_str}")
    if death_str:
        parts.append(f"died {death_str}")

    return ", ".join(parts)


def _get_person_events(person, db_handle, include_private=True):
    """Get all events for a person as text lines."""
    events = []
    for event_ref in person.event_ref_list:
        try:
            event = db_handle.get_event_from_handle(event_ref.ref)
        except HandleError:
            continue
        if not include_private and (event_ref.private or event.private):
            continue
        # Use the semantic event_to_line for consistent formatting
        event_line = event_to_line(event, db_handle)
        # event_to_line returns a PString, get the string_all
        if hasattr(event_line, "string_all"):
            text = event_line.string_all if include_private else event_line.string_public
        else:
            text = str(event_line)
        if text:
            events.append(f"  - {text}")
    return events


def _get_person_notes(person, db_handle, include_private=True):
    """Get notes for a person."""
    notes = []
    for note_handle in person.note_list:
        try:
            note = db_handle.get_note_from_handle(note_handle)
        except HandleError:
            continue
        if not include_private and note.private:
            continue
        text = note.get()
        if text:
            notes.append(text[:500])  # Truncate long notes
    return notes


def _build_person_context(gramps_id, db_handle, include_private=True):
    """Build the full person + family context string for insight generation.

    Returns:
        Tuple of (context_string, person_handle) or raises abort
    """
    logger = get_logger()

    # Get the person
    person = db_handle.get_person_from_gramps_id(gramps_id)
    if not person:
        abort_with_message(404, f"Person {gramps_id} not found")

    if not include_private and person.private:
        abort_with_message(403, "Person is private")

    person_handle = person.handle

    # Build the main person section using obj_strings_from_object for the full record
    obj_dict = obj_strings_from_object(
        db_handle=db_handle,
        class_name="Person",
        obj=person,
        semantic=True,
    )
    if obj_dict:
        person_text = (
            obj_dict["string_all"] if include_private else obj_dict["string_public"]
        )
    else:
        person_text = f"Person: {name_to_text(person.primary_name)} [{gramps_id}]"

    context_parts = [f"PERSON RECORD:\n{person_text}"]

    # Get source count
    citation_count = len(person.citation_list)
    if citation_count:
        context_parts.append(f"Sources: {citation_count} source citations attached")

    # PARENTS
    parents_section = []
    for family_handle in person.parent_family_list:
        try:
            family = db_handle.get_family_from_handle(family_handle)
        except HandleError:
            continue

        if family.father_handle:
            try:
                father = db_handle.get_person_from_handle(family.father_handle)
                if include_private or not father.private:
                    parents_section.append(
                        f"  Father: {_person_brief(father, db_handle, include_private)}"
                    )
            except HandleError:
                pass

        if family.mother_handle:
            try:
                mother = db_handle.get_person_from_handle(family.mother_handle)
                if include_private or not mother.private:
                    parents_section.append(
                        f"  Mother: {_person_brief(mother, db_handle, include_private)}"
                    )
            except HandleError:
                pass

    if parents_section:
        context_parts.append("PARENTS:\n" + "\n".join(parents_section))

    # SIBLINGS (children of parent families, excluding this person)
    siblings_section = []
    for family_handle in person.parent_family_list:
        try:
            family = db_handle.get_family_from_handle(family_handle)
        except HandleError:
            continue

        for child_ref in family.child_ref_list:
            if child_ref.ref == person_handle:
                continue  # Skip the person themselves
            try:
                sibling = db_handle.get_person_from_handle(child_ref.ref)
            except HandleError:
                continue
            if not include_private and (child_ref.private or sibling.private):
                continue
            siblings_section.append(
                f"  - {_person_brief(sibling, db_handle, include_private)}"
            )

    if siblings_section:
        context_parts.append("SIBLINGS:\n" + "\n".join(siblings_section))

    # SPOUSES and CHILDREN (from family_list where this person is a parent)
    spouses_section = []
    children_section = []
    for family_handle in person.family_list:
        try:
            family = db_handle.get_family_from_handle(family_handle)
        except HandleError:
            continue

        # Determine the spouse
        spouse_handle = None
        if family.father_handle == person_handle:
            spouse_handle = family.mother_handle
        elif family.mother_handle == person_handle:
            spouse_handle = family.father_handle

        if spouse_handle:
            try:
                spouse = db_handle.get_person_from_handle(spouse_handle)
                if include_private or not spouse.private:
                    spouse_text = _person_brief(spouse, db_handle, include_private)
                    # Add marriage info from family events
                    for event_ref in family.event_ref_list:
                        try:
                            event = db_handle.get_event_from_handle(event_ref.ref)
                            event_type = str(event.type)
                            if "marriage" in event_type.lower():
                                marriage_info = ""
                                if event.date and not event.date.is_empty():
                                    marriage_info = date_to_text(event.date)
                                if event.place:
                                    try:
                                        place = db_handle.get_place_from_handle(
                                            event.place
                                        )
                                        place_name = place.get_title()
                                        if place_name:
                                            if marriage_info:
                                                marriage_info += f", {place_name}"
                                            else:
                                                marriage_info = place_name
                                    except HandleError:
                                        pass
                                if marriage_info:
                                    spouse_text += f", married {marriage_info}"
                                break
                        except HandleError:
                            continue
                    spouses_section.append(f"  {spouse_text}")
            except HandleError:
                pass

        # Children of this family
        for child_ref in family.child_ref_list:
            try:
                child = db_handle.get_person_from_handle(child_ref.ref)
            except HandleError:
                continue
            if not include_private and (child_ref.private or child.private):
                continue
            children_section.append(
                f"  - {_person_brief(child, db_handle, include_private)}"
            )

    if spouses_section:
        context_parts.append("SPOUSE(S):\n" + "\n".join(spouses_section))
    if children_section:
        context_parts.append("CHILDREN:\n" + "\n".join(children_section))

    context = "\n\n".join(context_parts)
    logger.debug(
        "Built person context for %s: %d chars", gramps_id, len(context)
    )
    return context, person_handle


class InsightResource(ProtectedResource):
    """AI person insight resource."""

    def get(self, gramps_id):
        """Get cached insight for a person."""
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()

        # We need the person handle to look up the insight
        db_handle = None
        person_handle = None
        try:
            db_handle = get_db_outside_request(
                tree=tree, view_private=True, readonly=True, user_id=user_id
            )
            person = db_handle.get_person_from_gramps_id(gramps_id)
            if person:
                person_handle = person.handle
        except Exception as e:
            get_logger().error("Error looking up person %s: %s", gramps_id, e)
        finally:
            if db_handle:
                db_handle.close()

        if not person_handle:
            abort_with_message(404, f"Person {gramps_id} not found")

        insight = PersonInsight.query.filter_by(
            tree=tree, person_handle=person_handle
        ).first()

        if not insight:
            abort_with_message(404, "No insight found for this person")

        return {
            "data": {
                "id": insight.id,
                "person_handle": insight.person_handle,
                "gramps_id": gramps_id,
                "content": insight.content,
                "model": insight.model,
                "created_at": insight.created_at.isoformat() if insight.created_at else None,
            }
        }, 200

    def post(self, gramps_id):
        """Generate (or regenerate) insight for a person."""
        require_permissions({PERM_USE_CHAT})
        check_quota_ai(requested=1)
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})

        config = current_app.config
        model_name = config.get("LLM_MODEL")
        if not model_name:
            abort_with_message(500, "No LLM model configured")

        # Build person context from Gramps DB
        db_handle = None
        try:
            db_handle = get_db_outside_request(
                tree=tree,
                view_private=include_private,
                readonly=True,
                user_id=user_id,
            )
            person_context, person_handle = _build_person_context(
                gramps_id, db_handle, include_private
            )
        except Exception as e:
            if db_handle:
                db_handle.close()
            raise
        finally:
            if db_handle:
                db_handle.close()

        # Generate insight via Gemini (synchronous, no tools)
        insight_text, metadata = generate_insight(
            person_context=person_context,
            tree=tree,
            include_private=include_private,
            user_id=user_id,
        )

        if not insight_text:
            abort_with_message(500, "Failed to generate insight")

        # Upsert into person_insights table
        existing = PersonInsight.query.filter_by(
            tree=tree, person_handle=person_handle
        ).first()

        if existing:
            existing.content = insight_text
            existing.generated_by = user_id
            existing.model = model_name
            existing.created_at = datetime.utcnow()
            insight_id = existing.id
        else:
            insight_id = str(uuid.uuid4())
            new_insight = PersonInsight(
                id=insight_id,
                tree=tree,
                person_handle=person_handle,
                content=insight_text,
                generated_by=user_id,
                model=model_name,
            )
            user_db.session.add(new_insight)

        user_db.session.commit()
        update_usage_ai(new=1)

        return {
            "data": {
                "id": insight_id,
                "person_handle": person_handle,
                "gramps_id": gramps_id,
                "content": insight_text,
                "model": model_name,
                "created_at": datetime.utcnow().isoformat(),
            }
        }, 200
