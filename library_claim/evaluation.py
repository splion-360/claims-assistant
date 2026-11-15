"""Score a claim packet against hand-collected ground truth on the brief's pass bars.

Rules that keep the score honest:
- A value the system did not produce counts as a miss, never as a skipped sample.
- A bar whose ground truth was not collected is "not measured", never "pass".
- A wrong title is "confidently wrong" only when the system neither left it blank nor
  sent that line to the review queue.
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from library_claim.identification.matching import text_relation

# Text relations accepted as the same title, strongest first; matching uses them in this order.
TITLE_RELATIONS = ('exact', 'partial', 'fuzzy')
TIME_TO_PACKET_LIMIT_S = 300


@dataclass
class BarResult:
    """Outcome of one pass bar."""

    name: str
    bar: str
    value: str
    status: str  # 'pass', 'fail' or 'not measured'
    details: list[str] = field(default_factory=list)


def load_truth(path: Path) -> dict:
    """Read a ground-truth YAML file.

    Args:
        path: Ground-truth file in the shape of ground_truth/template.yml.

    Returns:
        The parsed ground truth; missing sections are simply absent.
    """
    return yaml.safe_load(Path(path).read_text()) or {}


def _match_titles(lines: list[dict], truths: list[dict]) -> dict[int, int]:
    """Pair packet lines with true titles one to one, strongest text relation first.

    Args:
        lines: Packet lines that carry a title.
        truths: Ground-truth entries with a title.

    Returns:
        Truth index keyed by line index.
    """
    pairs, used = {}, set()
    # Exact pairs are claimed before partial ones, so "Book 1" never steals "Book 10"'s line.
    for relation in TITLE_RELATIONS:
        for li, entry in enumerate(lines):
            if li in pairs:
                continue
            for ti, truth in enumerate(truths):
                if ti not in used and text_relation(entry['title'], truth['title']) == relation:
                    pairs[li] = ti
                    used.add(ti)
                    break
    return pairs


def _relative_error(measured, truth: float) -> float:
    """Return |measured - truth| / truth, treating a missing measurement as a 100% error.

    Args:
        measured: System value or None.
        truth: Ground-truth value, non-zero.

    Returns:
        Relative error.
    """
    if measured is None:
        return 1.0
    return abs(float(measured) - truth) / truth


def _find_line(lines: list[dict], title: str) -> dict | None:
    """Return the packet line whose title best matches a ground-truth title.

    Args:
        lines: Packet book lines.
        title: Ground-truth title.

    Returns:
        The best matching line, or None.
    """
    for relation in TITLE_RELATIONS:
        for entry in lines:
            if entry.get('title') and text_relation(entry['title'], title) == relation:
                return entry
    return None


def _count(packet: dict, truth: dict) -> BarResult:
    """Score the book count against the true count (within 5%)."""
    bar = 'within 5% of true count'
    true = (truth.get('books') or {}).get('count')
    if not true:
        return BarResult('Book count', bar, '', 'not measured')
    found = packet['totals'].get('book_count') or 0
    error = abs(found - true) / true
    return BarResult('Book count', bar, f'{found} vs {true} ({error:.1%})', 'pass' if error <= 0.05 else 'fail')


def _titles(packet: dict, truth: dict) -> BarResult:
    """Score title identification: at least 70% correct, at most 3% confidently wrong."""
    bar = '>= 70% of legible spines correct, <= 3% confidently wrong'
    books = truth.get('books') or {}
    truths = [entry for entry in books.get('titles') or [] if entry.get('title')]
    if not truths:
        return BarResult('Title identification', bar, '', 'not measured')
    lines = [entry for entry in packet['books'] if entry.get('title')]
    pairs = _match_titles(lines, truths)
    reviewed = {entry['ref_id'] for entry in packet.get('review_queue') or []}
    correct = len(set(pairs.values()))
    details = [f'matched "{lines[li]["title"]}" -> "{truths[ti]["title"]}"' for li, ti in sorted(pairs.items())]
    confident, flagged = [], []
    for li, entry in enumerate(lines):
        if li in pairs:
            continue
        # Duplicate readings of an already matched title are a count problem, not a wrong title.
        if any(text_relation(entry['title'], truths[ti]['title']) in TITLE_RELATIONS for ti in pairs.values()):
            details.append(f'duplicate reading "{entry["title"]}" ({entry["id"]})')
            continue
        (flagged if entry['id'] in reviewed else confident).append(entry)
    details += [f'confidently wrong "{entry["title"]}" ({entry["id"]})' for entry in confident]
    details += [f'wrong but sent to review "{entry["title"]}" ({entry["id"]})' for entry in flagged]
    missed = [truths[ti]['title'] for ti in range(len(truths)) if ti not in set(pairs.values())]
    details += [f'not identified "{title}"' for title in missed]
    rate = correct / len(truths)
    # The brief does not name the denominator for "confidently wrong"; the true book count is used.
    denominator = books.get('count') or len(truths)
    wrong_rate = len(confident) / denominator
    status = 'pass' if rate >= 0.70 and wrong_rate <= 0.03 else 'fail'
    return BarResult('Title identification', bar,
                     f'{correct}/{len(truths)} correct ({rate:.0%}), {len(confident)} confidently wrong ({wrong_rate:.1%})',
                     status, details)


def _spines(packet: dict, truth: dict) -> BarResult:
    """Score spine height and thickness: mean error within 15%, unmeasured counting as 100%."""
    bar = 'within 15% on a hand-measured sample'
    sample = ((truth.get('books') or {}).get('spines') or {}).get('sample') or []
    if not sample:
        return BarResult('Spine dimensions', bar, '', 'not measured')
    errors, details = [], []
    for entry in sample:
        found = _find_line(packet['books'], entry['title']) or {}
        for key, label in (('height_cm', 'height'), ('thickness_cm', 'thickness')):
            if entry.get(key):
                error = _relative_error(found.get(f'spine_{key}'), entry[key])
                errors.append(error)
                details.append(f'"{entry["title"]}" {label}: {found.get(f"spine_{key}")} vs {entry[key]} ({error:.0%})')
    mean = sum(errors) / len(errors)
    within = sum(error <= 0.15 for error in errors)
    return BarResult('Spine dimensions', bar,
                     f'mean error {mean:.1%}, {within}/{len(errors)} dimensions within 15%',
                     'pass' if mean <= 0.15 else 'fail', details)


def _prices(packet: dict, truth: dict) -> BarResult:
    """Score replacement prices: mean error within 25%, unpriced counting as 100%."""
    bar = 'within 25% of hand-checked price'
    sample = truth.get('prices') or []
    if not sample:
        return BarResult('Book prices', bar, '', 'not measured')
    errors, details = [], []
    for entry in sample:
        found = _find_line(packet['books'], entry['title']) or {}
        amount = (found.get('replacement_cost') or {}).get('amount')
        error = _relative_error(amount, entry['replacement'])
        errors.append(error)
        details.append(f'"{entry["title"]}": {amount} vs {entry["replacement"]} ({error:.0%})')
    mean = sum(errors) / len(errors)
    within = sum(error <= 0.25 for error in errors)
    return BarResult('Book prices', bar, f'mean error {mean:.1%}, {within}/{len(errors)} within 25%',
                     'pass' if mean <= 0.25 else 'fail', details)


def _items(packet: dict, truth: dict) -> BarResult:
    """Score non-book items: at least 80% found with the right category, matched one to one."""
    bar = '>= 80% found with correct category'
    true_items = truth.get('items') or []
    if not true_items:
        return BarResult('Non-book items', bar, '', 'not measured')
    available: dict[str, int] = {}
    for item in packet.get('items') or []:
        available[item.get('category')] = available.get(item.get('category'), 0) + 1
    found, details = 0, []
    for item in true_items:
        if available.get(item['category'], 0) > 0:
            available[item['category']] -= 1
            found += 1
        else:
            details.append(f'missed {item["category"]}: {item.get("description", "")}'.rstrip(': '))
    details += [f'extra {category} x{count}' for category, count in available.items() if count]
    rate = found / len(true_items)
    return BarResult('Non-book items', bar, f'{found}/{len(true_items)} found ({rate:.0%})',
                     'pass' if rate >= 0.80 else 'fail', details)


def _area(packet: dict, truth: dict, kind: str) -> BarResult:
    """Score floor (10%) or wall (15%) area against tape-measured dimensions."""
    name, tolerance = ('Floor area', 0.10) if kind == 'floor' else ('Wall area', 0.15)
    bar = f'within {tolerance:.0%} of tape-measured area'
    room = truth.get('room') or {}
    true = room.get(f'{kind}_area_m2')
    if true is None and all(room.get(key) for key in ('length_m', 'width_m')):
        # A rectangle's areas follow from its sides; a stated area overrides this for other shapes.
        if kind == 'floor':
            true = room['length_m'] * room['width_m']
        elif room.get('height_m'):
            true = 2 * (room['length_m'] + room['width_m']) * room['height_m']
    if not true:
        return BarResult(name, bar, '', 'not measured')
    found = (packet.get('room') or {}).get(f'{kind}_area_m2')
    if found is None:
        return BarResult(name, bar, f'not measured by the system (truth {true:.1f} m²)', 'fail')
    error = abs(found - true) / true
    return BarResult(name, bar, f'{found:.1f} vs {true:.1f} m² ({error:.1%})', 'pass' if error <= tolerance else 'fail')


def _time(packet: dict) -> BarResult:
    """Score the time from sweep end to packet (under 5 minutes)."""
    bar = 'under 5 minutes after the sweep ends'
    seconds = (packet.get('sweep') or {}).get('time_to_packet_s')
    if seconds is None:
        return BarResult('Time to packet', bar, 'not recorded by the system', 'fail')
    return BarResult('Time to packet', bar, f'{seconds:.0f} s', 'pass' if seconds < TIME_TO_PACKET_LIMIT_S else 'fail')


def score(packet: dict, truth: dict) -> list[BarResult]:
    """Score a claim packet on every pass bar.

    Args:
        packet: Parsed claim_packet.json.
        truth: Parsed ground truth.

    Returns:
        One result per pass bar, in the brief's order.
    """
    return [_count(packet, truth), _titles(packet, truth), _spines(packet, truth), _prices(packet, truth),
            _items(packet, truth), _area(packet, truth, 'floor'), _area(packet, truth, 'wall'), _time(packet)]


def results_markdown(results: list[BarResult]) -> str:
    """Render the results as a Markdown table followed by per-bar details.

    Args:
        results: Scorer output.

    Returns:
        Markdown text.
    """
    rows = ['| Measure | Pass bar | System result | Status |', '| --- | --- | --- | --- |']
    rows += [f'| {r.name} | {r.bar} | {r.value or "–"} | {r.status} |' for r in results]
    sections = []
    for result in results:
        if result.details:
            sections.append(f'\n### {result.name}\n' + '\n'.join(f'- {detail}' for detail in result.details))
    return '\n'.join(rows) + '\n' + '\n'.join(sections) + '\n'
