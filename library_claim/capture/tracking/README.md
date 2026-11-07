# Frame tracking and perspective mapping

`FrameTracker.update(image, frame_id, timestamp)` returns a visual association,
normalized current-to-reference homography, reference overlap, support count,
reprojection error, and latency. It does not identify books or assign physical
book IDs. Input is decoded uint8 BGR or grayscale; timestamps must increase.

## Association path

1. Reduce the image to a maximum edge of 640 pixels and detect up to 250 corners.
2. Follow corners with pyramidal Lucas–Kanade flow in both directions. Reject
   points with inconsistent forward/backward tracks or positions outside the image.
3. Fit a planar mapping from the surviving correspondences using ordinary least
   squares and verify its error and spatial support. This path does not use RANSAC.
4. On tracking failure, a gap over one second, or every ten successful flow steps,
   extract up to 400 SIFT descriptors and query the saved-view FLANN index.
5. Reject ambiguous descriptor matches, group by reference view, and geometrically
   verify at most three candidates with RANSAC capped at 500 iterations each.

The initial configuration requires at least 12 supporting matches, 65% agreement,
reprojection errors within three assessed pixels, and support over at least 8%
of both image areas. Mapped corners must form a plausible, convex, orientation-
preserving footprint. These are provisional engineering thresholds tested on
synthetic transformations, not calibrated shelf-identification guarantees.

A unique verified recovery gives `recovered`; ordinary continuity gives `tracked`.
An unmatched textured frame creates a provisional `reference_created` view. It
is not proof of a new shelf or new books. Featureless frames, ambiguous recoveries,
and implausible geometry give `uncertain`. Do not treat these as duplicate evidence.

## Coordinate contract

`current_to_reference` maps normalized current-image points to normalized points
in the saved reference view. `map_polygon()` applies that mapping to the actual
corners of a skewed spine. Reference coordinates may fall outside zero to one;
clipping them would hide newly exposed coverage. An image homography is not a
metric shelf or room coordinate system. Scale and measurements remain deferred.

One homography approximates a planar surface. Parallax from protruding books,
multiple walls, occlusion, repeated spines, and camera rolling shutter can defeat
this approximation. Reject uncertain associations rather than merging inventory.
Local shelf-region tracking and physical book association remain later integration
work; the live tracker currently assesses the complete frame.

`BookObservation.polygon` can preserve four normalized image corners alongside
its enclosing box. `FrameAssociation` stores a verified normalized mapping between
two saved Frame records. Migration 002 enforces that both frames belong to the
same sweep. Association records are immutable. The live selector persists a mapping after both
supporting images have been saved; only selected live frame IDs receive persistent records.

## Memory and runtime

The index retains at most 24 grayscale reference views and their descriptors.
Oldest views are evicted and the index is rebuilt on insertion. A larger scan may
therefore lose recovery to an old view; bounded memory does not imply complete
room coverage. References and flow state are session-local and reset when the
camera is toggled. Persisted inventory does not restore optical flow after restart.

The browser now samples at a provisional five frames per second. It skips sending
when its WebSocket buffer exceeds 512 KB. The server permits rate jitter up to ten
video messages per second and runs at most one tracking task per connection,
dropping frames while busy. Work runs in a thread; results from an old camera
activation are discarded. BRISQUE remains capped at once per second and remote
vision remains at one request per eight seconds with one request in flight.

A combined Rich camera summary logs every five seconds by default, controlled by
`capture.summary_log_interval_s`. Full matrices and frame references are emitted
at debug level. Existing visual reasoning is not yet gated by these
associations; relevance and persistent book association are subsequent increments.

## Evidence comparison

`capture.selection.compare_evidence()` compares against the last retained view,
not just the immediately preceding camera frame. It reports candidate new coverage,
closer detail, or improved quality for unresolved targets. Uncertain associations
require review; unchanged observations can be skipped. Caller-owned retained
observations determine the history; this helper does not create another cache.

The initial novelty and quality deltas are heuristic. A candidate does not prove
new readable information. Relevance classification and calibrated quality gating
remain separate. Region-specific quality comparison deliberately requires target
mapping rather than assuming the same crop coordinates depict the same book.

## Live persistence integration

`CameraTracking` now runs tracking, sampled quality assessment, and selective
persistence sequentially in one background worker. This pairs quality measurements
with the exact tracked frame and avoids duplicate BRISQUE computation. Quality
remains capped at once per second; tracking drops incoming frames while busy.
New references save their original JPEG immediately. Later scored frames use
`compare_evidence()` against the last retained view for that reference.

See [storage](../../storage/README.md) for file locations, transactional association
writes, retention, and restart behavior. Persisted frames are candidate evidence;
book relevance, identity resolution, and final packet assembly remain later steps.
