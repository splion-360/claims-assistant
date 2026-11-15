"""Assemble the claim packet for one sweep and render its readable report."""
import io
import logging
import json
import zipfile
from datetime import datetime
from html import escape
from pathlib import Path

from library_claim.config import CONFIG
from library_claim.domain.models import Frame, Sweep
from library_claim.identification.storage import inventory, inventory_totals
from library_claim.items.pricing import value_items
from library_claim.items.storage import inventory as item_inventory
from library_claim.measurement.room_storage import stored_room
from library_claim.pricing.models import NEW_CONDITION, Valuation
from library_claim.pricing.compare import compare_markets
from library_claim.pricing.storage import CACHE_FILE, value_sweep
from library_claim.storage.repository import Repository

logger = logging.getLogger(__name__)

# Empty money blocks: the keys stay so the contract holds, the amounts stay unknown.
NO_REPLACEMENT_COST = {'amount': None, 'source': '', 'url': '', 'retrieved_at': '', 'converted': False}
NO_USED_VALUE = {'amount': None, 'source': '', 'url': '', 'retrieved_at': '', 'condition_assumed': ''}
TIME_TO_PACKET_LIMIT_S = 300
UNPRICED_REASON = 'No price source; excluded from monetary totals'


def _duration_s(frames: list[Frame]) -> float:
    """Measure the sweep from its first to its last saved frame.

    Args:
        frames: Saved frames of the sweep, in any order.

    Returns:
        Elapsed seconds, or zero when fewer than two frames exist.
    """
    if len(frames) < 2:
        return 0.0
    times = sorted(frame.captured_at for frame in frames)
    return round((times[-1] - times[0]).total_seconds(), 1)


def _time_to_packet(sweep: Sweep) -> float | None:
    """Measure from the camera stopping to the background work finishing.

    Args:
        sweep: The sweep record.

    Returns:
        Seconds, or None when either moment was not recorded.
    """
    if sweep.ended_at is None or sweep.packet_ready_at is None:
        return None
    return round((sweep.packet_ready_at - sweep.ended_at).total_seconds(), 1)


def _book_line(book: dict, evidence: dict[str, str], valuation: Valuation | None = None) -> dict:
    """Convert one inventory entry into a claim packet book line.

    Catalogue fields fill edition and ISBN only where a catalogue answered, so a
    line never carries an identity the sweep did not establish. Prices come from
    the valuation and stay empty when no source answered.

    Args:
        book: Inventory entry from `inventory`.
        evidence: Saved frame path indexed by frame ID.
        valuation: Pricing outcome for this book, or None when pricing did not run.

    Returns:
        A book line holding only values backed by evidence.
    """
    catalogue = book.get('catalogue') or {}
    frame_ids = book.get('frame_ids') or []
    appraisal = book.get('needs_appraisal') or (valuation is not None and valuation.needs_appraisal)
    status = 'needs_appraisal' if appraisal else ('identified' if book.get('title') else 'unidentified')
    # A book withheld for appraisal keeps its figure out of the packet's money lines.
    replacement = None if appraisal or valuation is None else valuation.replacement
    used = None if appraisal or valuation is None else valuation.used
    return {
        'id': book['id'],
        # Shelf and position are not yet derived from the sweep; an empty field is the honest value.
        'shelf': '', 'position': None,
        'frame_ref': evidence.get(frame_ids[0], '') if frame_ids else '',
        'frame_refs': [evidence[frame_id] for frame_id in frame_ids if frame_id in evidence],
        'status': status,
        'title': book.get('title') or '',
        'author': book.get('author') or '',
        'edition': catalogue.get('edition') or '',
        'isbn': catalogue.get('isbn') or '',
        'spine_height_cm': book.get('spine_height_cm'),
        'spine_thickness_cm': book.get('spine_thickness_cm'),
        'scale_method': book.get('scale_method') or '',
        'id_confidence': None,
        'catalogue_url': catalogue.get('url') or '',
        'observation_count': book.get('observation_count', 0),
        'replacement_cost': replacement.as_replacement_block() if replacement else dict(NO_REPLACEMENT_COST),
        'used_value': used.as_used_block() if used else dict(NO_USED_VALUE),
    }


def _item_line(item: dict, evidence: dict[str, str], valuation: Valuation | None) -> dict:
    """Convert one item inventory entry into a claim packet line.

    Args:
        item: Inventory entry from the item store.
        evidence: Saved frame path indexed by frame ID.
        valuation: Pricing outcome for this object, or None when pricing did not run.

    Returns:
        An item line holding only values backed by evidence.
    """
    frame_ids = item.get('frame_ids') or []
    appraisal = item.get('needs_appraisal') or (valuation is not None and valuation.needs_appraisal)
    low = None if appraisal or valuation is None else valuation.replacement
    high = (valuation.replacement_high or low) if low is not None else None
    # One matched product is "priced"; several comparable listings give a "range".
    exact = low is not None and item.get('brand_model') and high.amount == low.amount
    status = 'needs_appraisal' if appraisal else ('priced' if exact else 'range')
    return {
        'id': item['id'], 'category': item.get('category', 'other'),
        'description': item.get('description', ''), 'brand_model': item.get('brand_model') or '',
        'material': item.get('material') or '',
        'frame_ref': evidence.get(frame_ids[0], '') if frame_ids else '',
        'frame_refs': [evidence[frame_id] for frame_id in frame_ids if frame_id in evidence],
        # Dimensions need metric depth, which no item measurement pass supplies yet.
        'dimensions_cm': {'w': None, 'h': None, 'd': None},
        'status': status,
        'replacement_cost': ({'low': round(low.amount, 2), 'high': round(high.amount, 2),
                              'currency': low.currency, 'source': low.source, 'url': low.url,
                              'url_high': high.url, 'retrieved_at': low.retrieved_at.isoformat(),
                              'converted': low.converted, 'converted_from': low.converted_from,
                              'condition_assumed': low.condition_assumed or NEW_CONDITION}
                             if low is not None else
                             {'low': None, 'high': None, 'source': '', 'url': '', 'retrieved_at': '',
                              'converted': False}),
        'observation_count': item.get('observation_count', 0),
        'confidence': None,
    }


def _totals(books: list[dict], lines: list[dict], items: list[dict]) -> dict:
    """Compute every total in code by summing the packet's own lines.

    Args:
        books: Inventory entries, used for the counts and shelf run.
        lines: Book lines of the packet.
        items: Non-book item lines of the packet.

    Returns:
        Counts, shelf run and monetary totals over priced lines only.
    """
    counts = inventory_totals(books)
    priced = [line for line in lines if line['replacement_cost']['amount'] is not None]
    used = [line for line in lines if line['used_value']['amount'] is not None]
    item_amounts = [item['replacement_cost'] for item in items
                    if item['replacement_cost'].get('low') is not None]
    return {
        **counts,
        'books_replacement_cost': round(sum(line['replacement_cost']['amount'] for line in priced), 2),
        'books_used_value': round(sum(line['used_value']['amount'] for line in used), 2),
        'items_replacement_cost_low': round(sum(amount.get('low') or 0 for amount in item_amounts), 2),
        'items_replacement_cost_high': round(sum(amount.get('high') or 0 for amount in item_amounts), 2),
        # Lines carrying no retrievable price contribute nothing to the money totals.
        'excluded_from_totals': len(lines) - len(priced),
    }


def _review_queue(books: list[dict], lines: list[dict], sweep: Sweep,
                  valuations: dict[str, Valuation], objects: list[dict],
                  item_values: dict[str, Valuation], room: dict | None) -> list[dict]:
    """List every line the sweep is not confident in, with its reason.

    Args:
        books: Inventory entries carrying their review reasons.
        lines: Book lines of the packet.
        sweep: The sweep record, checked for the locale pricing needs.
        valuations: Book pricing outcomes indexed by book ID.
        objects: Non-book inventory entries carrying their review reasons.
        item_values: Item pricing outcomes indexed by item ID.
        room: The saved room estimate, or None when the sweep measured none.

    Returns:
        Review entries referencing a book, an object, the room or the sweep.
    """
    queue = []
    for book in books:
        # The inventory already reports unreadable titles, duplicate risk and missing depth.
        for reason in book.get('review_reasons') or []:
            queue.append({'ref_id': book['id'], 'reason': reason})
    for line in lines:
        # Pricing says in its own words why a figure is missing or withheld.
        for reason in (valuations.get(line['id']).reasons if valuations.get(line['id']) else ()):
            queue.append({'ref_id': line['id'], 'reason': reason})
        if line['replacement_cost']['amount'] is None:
            queue.append({'ref_id': line['id'], 'reason': UNPRICED_REASON})
    for entry in objects:
        for reason in entry.get('review_reasons') or []:
            queue.append({'ref_id': entry['id'], 'reason': reason})
        for reason in (item_values[entry['id']].reasons if entry['id'] in item_values else ()):
            queue.append({'ref_id': entry['id'], 'reason': reason})
        # Dimensions for an object need metric depth that no item pass supplies yet.
        queue.append({'ref_id': entry['id'], 'reason': 'Object dimensions were not measured'})
    if room is None:
        queue.append({'ref_id': 'room',
                      'reason': 'Room dimensions and areas were not measured: the sweep captured too '
                                'few frames carrying both depth and a camera pose'})
    else:
        # An estimate measures only what the sweep looked at, so its coverage is a review fact.
        queue.append({'ref_id': 'room', 'reason': room['shape_note']})
        if room['yaw_coverage'] < CONFIG.measurement.room_min_coverage:
            queue.append({'ref_id': 'room',
                          'reason': f'The sweep looked across {round(room["yaw_coverage"] * 100)}% of a '
                                    f'full turn, so the floor outline covers only part of the room'})
        if not room['shelved_wall_area_m2']:
            queue.append({'ref_id': 'room', 'reason': 'No shelving was measured against a wall'})
    if not objects:
        queue.append({'ref_id': 'items', 'reason': 'No non-book contents were recorded during this sweep'})
    seconds = _time_to_packet(sweep)
    if seconds is not None and seconds > TIME_TO_PACKET_LIMIT_S:
        queue.append({'ref_id': 'sweep', 'reason': f'The packet took {seconds:.0f} s after the sweep ended, '
                                                   f'over the 5-minute target'})
    if not sweep.country or not sweep.currency:
        queue.append({'ref_id': 'sweep',
                      'reason': 'Country and currency were not confirmed by the policyholder; '
                                'the configured default market was used for pricing'})
    # One line per distinct concern; the same reason from two sightings is one review item.
    seen, unique = set(), []
    for entry in queue:
        key = (entry['ref_id'], entry['reason'])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def build_packet(database: Path, sweep_id: str, *, price: bool = True,
                 country: str | None = None, client=None, compare: bool = True) -> dict:
    """Build the claim packet for one sweep from its saved evidence.

    Every value comes from a stored record or a retrievable source: identities
    from spine readings and accepted catalogue matches, dimensions from measured
    depth, prices from a named source with its retrieval date, and totals from
    the lines in this packet. Unknowns stay empty and reach the review queue.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        price: Whether to value the books; False builds the packet without lookups.
        country: Market to price in, overriding the sweep's own country.
        client: Optional HTTP client passed to the price sources, used by tests.
        compare: Also price the same books in the configured second market.

    Returns:
        The claim packet as the output contract defines it.

    Raises:
        KeyError: No sweep with that identifier exists.
    """
    with Repository(database) as repo:
        sweep = repo.get(Sweep, sweep_id)
        frames = repo.list_for_sweep(Frame, sweep_id)
        books = inventory(repo, sweep_id)
        contents = item_inventory(repo, sweep_id)
        room = stored_room(repo, sweep_id)
    # frame_ref must resolve to a file an adjuster can open, so it names the saved JPEG.
    evidence = {frame.id: f'frames/{frame.evidence_key}' for frame in frames}
    kept = [book for book in books if not book.get('excluded')]
    chosen = country or sweep.country or CONFIG.pricing.default_country
    # Books already priced keep their saved figure, so a second download fetches nothing.
    valuations = value_sweep(database, sweep_id, kept, chosen, client=client) if price else {}
    if price:
        kept = [{**book, 'valuation': valuations[book['id']].model_dump(mode='json')}
                if book['id'] in valuations else book for book in kept]
    objects = [entry for entry in contents if not entry.get('excluded')]
    # Objects already priced keep their saved figure, so a second download fetches nothing.
    item_values = value_items(database, sweep_id, objects, chosen, client=client) if price else {}
    items = [_item_line(entry, evidence, item_values.get(entry['id'])) for entry in objects]
    lines = [_book_line(book, evidence, valuations.get(book['id'])) for book in kept]
    currency = sweep.currency or next((value.currency for value in valuations.values() if value.currency), '')
    comparison = _comparison(database, kept, chosen, client) if price and compare else None
    packet = {
        'sweep': {'id': sweep.id, 'captured_at': sweep.captured_at.isoformat(), 'device': sweep.device,
                  'duration_s': _duration_s(frames), 'country': (sweep.country or chosen).upper(),
                  'currency': currency,
                  'frames_saved': len(frames), 'generated_at': datetime.now(sweep.captured_at.tzinfo).isoformat(),
                  'ended_at': sweep.ended_at.isoformat() if sweep.ended_at else None,
                  'time_to_packet_s': _time_to_packet(sweep)},
        'room': room or {'length_m': None, 'width_m': None, 'height_m': None, 'floor_area_m2': None,
                         'wall_area_m2': None, 'shelved_wall_area_m2': None,
                         'scale_method': '', 'confidence': 0},
        'books': lines,
        'items': items,
        'totals': _totals(kept, lines, items),
        'review_queue': _review_queue(kept, lines, sweep, valuations, objects, item_values, room),
    }
    if comparison is not None:
        packet['locale_comparison'] = comparison
    return packet


def _comparison(database: Path, books: list[dict], chosen: str, client) -> dict | None:
    """Price the same books in the configured second market, or skip when none is set.

    Args:
        database: Capture SQLite path; the price cache lives beside it.
        books: Kept inventory entries.
        chosen: The policyholder's market.
        client: Optional HTTP client, used by tests.

    Returns:
        The comparison block, or None when no second market is configured or it failed.
    """
    second = (CONFIG.pricing.compare_country or '').upper()
    if not second:
        return None
    # When the sweep is already in the comparison market, compare against the default one.
    if second == chosen.upper():
        second = CONFIG.pricing.default_country.upper()
    if second == chosen.upper():
        return None
    try:
        return compare_markets(books, [chosen.upper(), second], cache_path=Path(database).parent / CACHE_FILE,
                               client=client, limit=CONFIG.pricing.compare_books)
    except Exception as error:  # noqa: BLE001 - the comparison is a demonstration, never a blocker
        logger.warning('locale comparison failed: %s', type(error).__name__)
        return None


def _cell(value) -> str:
    """Render one packet value as table text, showing unknowns as a dash.

    Args:
        value: Any packet value.

    Returns:
        Escaped text, or an em dash when the value is unknown.
    """
    if value is None or value == '':
        return '<span class="unknown">&mdash;</span>'
    return escape(str(value))


def _range(block: dict, currency: str) -> str:
    """Render an item's replacement cost, a single figure or a sourced range.

    Args:
        block: An item replacement_cost block.
        currency: Currency to show when the block names none.

    Returns:
        HTML with the figure(s), source and links, or a note that no price was found.
    """
    if block.get('low') is None:
        return '<span class="unknown">no price found</span>'
    unit = escape(block.get('currency') or currency)
    figure = (f'{unit} {block["low"]:.2f}' if block['low'] == block['high']
              else f'{unit} {block["low"]:.2f} – {block["high"]:.2f}')
    links = f'<a href="{escape(block["url"])}">low</a>'
    if block.get('url_high') and block['url_high'] != block['url']:
        links += f' · <a href="{escape(block["url_high"])}">high</a>'
    converted = ' · converted' if block.get('converted') else ''
    return f'{figure}<br><small>{escape(block["source"])} · {links}{converted}</small>'


def _comparison_html(comparison: dict | None) -> str:
    """Render the second-market comparison table.

    Args:
        comparison: The packet's locale_comparison block, or None.

    Returns:
        An HTML section, or '' when the packet has no comparison.
    """
    if not comparison or not comparison.get('books'):
        return ''
    countries = comparison['countries']

    def cell(entry: dict, kind: str) -> str:
        if entry.get('needs_appraisal'):
            return '<span class="unknown">appraisal</span>'
        figure = entry.get(kind)
        if not figure:
            return '<span class="unknown">&mdash;</span>'
        mark = ' (converted)' if figure['converted'] else ''
        return f'<a href="{escape(figure["url"])}">{escape(figure["currency"])} {figure["amount"]:.2f}</a>{mark}'
    head = ''.join(f'<th>{escape(code)} new</th><th>{escape(code)} used</th>' for code in countries)
    rows = ''.join(f'<tr><td>{escape(row["title"])}</td>'
                   + ''.join(cell(row['prices'][code], 'replacement').join(('<td>', '</td>'))
                             + cell(row['prices'][code], 'used').join(('<td>', '</td>')) for code in countries)
                   + '</tr>' for row in comparison['books'])
    return (f'<h2>Same books in {len(countries)} markets</h2><p class="meta">Each market is looked up in its own '
            f'stores and currency; nothing is hardcoded per country.</p>'
            f'<table><thead><tr><th>Title</th>{head}</tr></thead><tbody>{rows}</tbody></table>')


def _money(block: dict, currency: str) -> str:
    """Render a money block with its source, or state that no price was found.

    Args:
        block: A replacement cost or used value block.
        currency: Sweep currency, blank when it was never confirmed.

    Returns:
        Escaped cell text naming the amount and its source.
    """
    if block.get('amount') is None:
        return '<span class="unknown">no price source</span>'
    source = escape(block.get('source') or 'unsourced')
    url = block.get('url')
    cited = f'<a href="{escape(url)}">{source}</a>' if url else source
    return f'{escape(currency)} {block["amount"]:.2f}<br><small>{cited} · {escape(block.get("retrieved_at") or "")}</small>'


def render_report(packet: dict) -> str:
    """Render the human-readable report for a claim packet.

    Args:
        packet: A packet from `build_packet`.

    Returns:
        A self-contained HTML document built only from the packet.
    """
    sweep, room, totals = packet['sweep'], packet['room'], packet['totals']
    currency = sweep.get('currency') or ''
    books = ''.join(
        f'<tr><td>{_cell(line["status"])}</td><td>{_cell(line["title"])}</td><td>{_cell(line["author"])}</td>'
        f'<td>{_cell(line["isbn"])}</td><td class="n">{_cell(line["spine_height_cm"])}</td>'
        f'<td class="n">{_cell(line["spine_thickness_cm"])}</td><td>{_cell(line["scale_method"])}</td>'
        f'<td>{_money(line["replacement_cost"], currency)}</td><td>{_money(line["used_value"], currency)}</td>'
        f'<td><code>{_cell(line["frame_ref"])}</code></td></tr>'
        for line in packet['books']) or '<tr><td colspan="10" class="unknown">No books recorded.</td></tr>'
    items = ''.join(
        f'<tr><td>{_cell(item.get("category"))}</td><td>{_cell(item.get("description"))}</td>'
        f'<td>{_cell(item.get("brand_model"))}</td><td>{_cell(item.get("status"))}</td>'
        f'<td>{_range(item["replacement_cost"], currency)}</td>'
        f'<td><code>{_cell(item.get("frame_ref"))}</code></td></tr>'
        for item in packet['items']) or '<tr><td colspan="6" class="unknown">No non-book items recorded.</td></tr>'
    review = ''.join(f'<tr><td><code>{_cell(entry["ref_id"])}</code></td><td>{_cell(entry["reason"])}</td></tr>'
                     for entry in packet['review_queue']) or '<tr><td colspan="2">Nothing flagged.</td></tr>'
    # Totals are printed from the packet's own totals block, which code computed from the lines.
    figures = ''.join(f'<div class="figure"><dt>{escape(name)}</dt><dd>{_cell(value)}</dd></div>'
                      for name, value in totals.items())
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Contents claim packet {escape(sweep["id"][:8])}</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 2rem; color: #1a1a1a; background: #fff; }}
 h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; }}
 h2 {{ font-size: 1.05rem; margin: 2rem 0 .5rem; border-bottom: 1px solid #ddd; padding-bottom: .25rem; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
 th, td {{ border-bottom: 1px solid #e5e5e5; padding: .4rem .5rem; text-align: left; vertical-align: top; }}
 th {{ background: #f6f6f6; font-weight: 600; }}
 td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
 code {{ font-size: 11px; color: #555; word-break: break-all; }}
 .unknown {{ color: #999; font-style: italic; }}
 .figures {{ display: flex; flex-wrap: wrap; gap: 1.5rem; margin: .5rem 0; }}
 .figure dt {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #666; }}
 .figure dd {{ margin: .1rem 0 0; font-size: 1.1rem; font-variant-numeric: tabular-nums; }}
 .meta {{ color: #666; margin: 0 0 1rem; }}
</style></head><body>
<h1>Library contents claim packet</h1>
<p class="meta">Sweep <code>{escape(sweep["id"])}</code> · captured {_cell(sweep["captured_at"])} ·
 {_cell(sweep["device"])} · {_cell(sweep["duration_s"])} s · {sweep["frames_saved"]} frames saved ·
 locale {_cell(sweep["country"])} / {_cell(sweep["currency"])}</p>
<h2>Totals</h2><dl class="figures">{figures}</dl>
<h2>Room</h2>
<dl class="figures">
 <div class="figure"><dt>Floor area</dt><dd>{_cell(room["floor_area_m2"])} m&sup2;</dd></div>
 <div class="figure"><dt>Floor area</dt><dd>{_cell(room.get("floor_area_sqft"))} sq ft</dd></div>
 <div class="figure"><dt>Wall area</dt><dd>{_cell(room["wall_area_m2"])} m&sup2;</dd></div>
 <div class="figure"><dt>Wall area</dt><dd>{_cell(room.get("wall_area_sqft"))} sq ft</dd></div>
 <div class="figure"><dt>Shelved wall</dt><dd>{_cell(room["shelved_wall_area_m2"])} m&sup2;</dd></div>
 <div class="figure"><dt>Length</dt><dd>{_cell(room["length_m"])} m</dd></div>
 <div class="figure"><dt>Width</dt><dd>{_cell(room["width_m"])} m</dd></div>
 <div class="figure"><dt>Ceiling</dt><dd>{_cell(room["height_m"])} m</dd></div>
</dl>
<p>Scale from {_cell(room["scale_method"])} · confidence {_cell(room["confidence"])}{
 f' · swept {round(room["yaw_coverage"] * 100)}% of a full turn' if room.get("yaw_coverage") is not None else ''}</p>
{f'<p>{escape(str(room["shape_note"]))}</p>' if room.get("shape_note") else ''}
<h2>Books ({len(packet["books"])})</h2>
<table><thead><tr><th>Status</th><th>Title</th><th>Author</th><th>ISBN</th><th>Height cm</th>
 <th>Thick cm</th><th>Scale</th><th>Replacement</th><th>Used</th><th>Frame</th></tr></thead>
<tbody>{books}</tbody></table>
<h2>Non-book items ({len(packet["items"])})</h2>
<table><thead><tr><th>Category</th><th>Description</th><th>Brand / model</th><th>Status</th><th>Replacement</th>
 <th>Frame</th></tr></thead>
<tbody>{items}</tbody></table>
{_comparison_html(packet.get("locale_comparison"))}
<h2>Review queue ({len(packet["review_queue"])})</h2>
<table><thead><tr><th>Line</th><th>Reason</th></tr></thead><tbody>{review}</tbody></table>
</body></html>'''


def packet_archive(database: Path, frames_root: Path, sweep_id: str, *, price: bool = True,
                   country: str | None = None, client=None) -> bytes:
    """Build the deliverable archive: the packet, its report and every frame cited.

    Shipping the cited frames alongside the packet keeps each `frame_ref`
    resolvable from the archive itself rather than from a running server.

    Args:
        database: Capture SQLite path.
        frames_root: Directory holding saved frame JPEGs.
        sweep_id: Owning sweep.
        price: Whether to value the books; False builds the archive without lookups.
        country: Market to price in, overriding the sweep's own country.
        client: Optional HTTP client passed to the price sources, used by tests.

    Returns:
        Zip archive bytes.

    Raises:
        KeyError: No sweep with that identifier exists.
    """
    from library_claim.storage.evidence import EvidenceStore

    packet = build_packet(database, sweep_id, price=price, country=country, client=client)
    evidence = EvidenceStore(frames_root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('claim_packet.json', json.dumps(packet, indent=2))
        archive.writestr('report.html', render_report(packet))
        for reference in sorted({path for line in packet['books'] for path in line['frame_refs']}):
            # A frame missing from disk is skipped; the packet still names it for the adjuster.
            try:
                archive.writestr(reference, evidence.read(reference.split('/', 1)[1]))
            except (OSError, ValueError, KeyError):
                continue
    return buffer.getvalue()
