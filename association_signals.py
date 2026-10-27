"""Fifty evidence-based association *signals*, inspired by associative-memory ideas.

Signals are measurable pairwise cues, not fifty biological neural connections or
independent facts. Each is trained as a weight. No cue on its own proves identity,
causation, co-presence, or truth. All values are finite and in [0, 1].
"""
from __future__ import annotations

import re
from typing import Protocol


class EventLike(Protocol):
    actor: str
    item: str
    action: str
    location: str
    stated_time: str
    kind: str
    source: str
    novelty: bool
    identity: str


_STOP = frozenset('a an the and to in on into at by near of her his she he is was were are it them later usually'.split())
_HEDGE = frozenset('maybe perhaps possibly probably might could allegedly reportedly seems apparently'.split())
_NEGATE = frozenset('not never no neither without cannot didnot'.split())
_DAYS = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
_PARTS = ('dawn', 'sunrise', 'morning', 'noon', 'afternoon', 'dusk', 'evening', 'night', 'midnight')
_MOVE = frozenset(('put', 'placed', 'moved', 'carried', 'left', 'stored', 'kept'))
_HABIT = frozenset(('usually', 'habit'))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _terms(text: str) -> set[str]:
    return _tokens(text) - _STOP


def _bigrams(text: str) -> set[tuple[str, str]]:
    words = [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in _STOP]
    return set(zip(words, words[1:]))


def _overlap(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _has(text: str, word: str) -> bool:
    return word in _tokens(text)


def _clock(text: str) -> str:
    match = re.search(r'\b(\d{1,2})(?::(\d\d))?\s*(am|pm)\b', text.lower())
    if not match:
        return ''
    hour = int(match[1]) % 12 + (12 if match[3] == 'pm' else 0)
    return f'{hour:02d}:{match[2] or "00"}'


def brain_signals(a: EventLike, b: EventLike) -> dict[str, float]:
    """Fifty distinct observable conjunctions/contrasts, not assertions of truth."""
    actor = bool(a.actor and b.actor and a.actor.casefold() == b.actor.casefold())
    item = bool(a.item and b.item and a.item.casefold() == b.item.casefold())
    same = bool(a.identity and a.identity == b.identity)
    act = bool(a.action and b.action and a.action.casefold() == b.action.casefold())
    loc = bool(a.location and b.location and a.location.casefold() == b.location.casefold())
    have_loc = bool(a.location and b.location)
    times = bool(a.stated_time and b.stated_time)
    same_time = times and a.stated_time.casefold() == b.stated_time.casefold()
    habits = a.kind == 'habit' or b.kind == 'habit'
    ta, tb = _tokens(a.stated_time), _tokens(b.stated_time)
    sa, sb = _tokens(a.source), _tokens(b.source)
    va, vb = _terms(a.source), _terms(b.source)
    pa, pb = _tokens(a.location), _tokens(b.location)
    day_a, day_b = {x for x in ta if x in _DAYS}, {x for x in tb if x in _DAYS}
    part_a, part_b = {x for x in ta if x in _PARTS}, {x for x in tb if x in _PARTS}
    ca, cb = _clock(a.stated_time), _clock(b.stated_time)
    move_a, move_b = a.action in _MOVE, b.action in _MOVE
    hedge_a, hedge_b = bool(sa & _HEDGE), bool(sb & _HEDGE)
    neg_a, neg_b = bool(sa & _NEGATE), bool(sb & _NEGATE)
    result = {
        # Identity binding and pattern separation: these do NOT merge entities.
        'actor_item_action': float(same and act),
        'actor_item_place': float(same and loc),
        'actor_item_time': float(same and same_time),
        'actor_item_habit': float(same and habits),
        'actor_shared_action': float(actor and act),
        'item_shared_action': float(item and act),
        'actor_shared_time': float(actor and same_time),
        'item_shared_time': float(item and same_time),
        'actor_item_changed_action': float(same and bool(a.action and b.action) and not act),
        'same_label_different_actor': float(item and bool(a.actor and b.actor) and not actor),
        # Spatial and contextual bindings.
        'actor_at_same_place': float(actor and loc),
        'item_at_same_place': float(item and loc),
        'actor_place_transition': float(actor and have_loc and not loc),
        'item_place_transition': float(item and have_loc and not loc),
        'place_exact': float(loc),
        'place_word_overlap': _overlap(pa, pb),
        'place_same_head': float(have_loc and a.location.split()[-1:] == b.location.split()[-1:]),
        'place_same_modifier': float(have_loc and a.location.split()[:1] == b.location.split()[:1]),
        'place_missing_one': float(bool(a.location) != bool(b.location)),
        'place_known_both': float(have_loc),
        # Temporal context and event-order cues. No fabricated absolute timestamps.
        'time_label_exact': float(same_time),
        'time_word_overlap': _overlap(ta, tb),
        'daypart_shared': float(bool(part_a & part_b)),
        'daypart_changed': float(bool(part_a and part_b and not (part_a & part_b))),
        'weekday_shared': float(bool(day_a & day_b)),
        'weekday_changed': float(bool(day_a and day_b and not (day_a & day_b))),
        'time_known_both': float(times),
        'time_missing_one': float(bool(a.stated_time) != bool(b.stated_time)),
        'clock_same': float(bool(ca and cb and ca == cb)),
        'clock_changed': float(bool(ca and cb and ca != cb)),
        # Event roles and action motifs.
        'action_exact': float(act),
        'action_both_movement': float(move_a and move_b),
        'action_movement_mismatch': float(move_a != move_b),
        'action_known_both': float(bool(a.action and b.action)),
        'habit_mixed': float(habits and (a.kind == 'habit') != (b.kind == 'habit')),
        'habit_both': float(a.kind == 'habit' and b.kind == 'habit'),
        'event_kind_same': float(a.kind == b.kind),
        'actor_event_kind_same': float(actor and a.kind == b.kind),
        'item_event_kind_same': float(item and a.kind == b.kind),
        'habit_location_conflict': float(same and habits and have_loc and not loc),
        # Source evidence, ambiguity and novelty (weights may inhibit links).
        'source_exact': float(bool(a.source and b.source and a.source == b.source)),
        'source_term_overlap': _overlap(va, vb),
        'source_bigram_overlap': _overlap(_bigrams(a.source), _bigrams(b.source)),
        'source_shared_distinctive': float(bool(va & vb)),
        'source_disjoint': float(bool(va and vb and not (va & vb))),
        'hedge_shared': float(hedge_a and hedge_b),
        'hedge_mismatch': float(hedge_a != hedge_b),
        'negation_mismatch': float(neg_a != neg_b),
        'novelty_shared': float(a.novelty and b.novelty),
        'novelty_mismatch': float(a.novelty != b.novelty),
    }
    if len(result) != 50:
        raise AssertionError('Expected exactly fifty additional memory signals')
    return result
