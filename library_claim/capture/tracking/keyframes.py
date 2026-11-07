"""Retrieve bounded candidate views from an in-memory SIFT descriptor index."""
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from library_claim.config import CONFIG

SETTINGS = CONFIG.tracking


@dataclass
class Keyframe:
    """Retain one reference view and its local image descriptors."""

    frame_id: str
    gray: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray


class Keyframes:
    """Own a bounded reference collection and FLANN nearest-neighbor index."""

    def __init__(self, capacity: int = SETTINGS.capacity):
        """Configure the reference budget.

        Args:
            capacity: Maximum number of saved reference views.

        Raises:
            ValueError: Capacity is less than one.
        """
        if capacity < 1:
            raise ValueError('Keyframe capacity must be positive')
        self.capacity = capacity
        self.views: list[Keyframe] = []
        self.sift = cv2.SIFT_create(nfeatures=SETTINGS.sift_features)
        self.matcher = None

    def describe(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract bounded local descriptors from a grayscale image.

        Args:
            gray: Grayscale uint8 image.

        Returns:
            Feature locations and SIFT descriptors, or None for no descriptors.
        """
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return np.float32([point.pt for point in keypoints]).reshape(-1, 2), descriptors

    def add(self, frame_id: str, gray: np.ndarray, points: np.ndarray,
            descriptors: np.ndarray) -> Keyframe:
        """Save a reference and rebuild its bounded retrieval index.

        Args:
            frame_id: Source capture identifier.
            gray: Reference image.
            points: SIFT feature locations.
            descriptors: Corresponding descriptor vectors.

        Returns:
            The saved keyframe.
        """
        view = Keyframe(frame_id, gray.copy(), points.copy(), descriptors.copy())
        self.views.append(view)
        self.views = self.views[-self.capacity:]
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=SETTINGS.flann_trees), dict(checks=SETTINGS.flann_checks))
        self.matcher.add([item.descriptors for item in self.views])
        self.matcher.train()
        return view

    def candidates(self, descriptors: np.ndarray | None) -> list[tuple[Keyframe, list]]:
        """Rank saved views by unambiguous descriptor-match support.

        Args:
            descriptors: Current image's SIFT descriptors.

        Returns:
            At most the configured number of candidate views with unique supporting matches.
        """
        if descriptors is None or self.matcher is None:
            return []
        groups = defaultdict(list)
        matches = self.matcher.knnMatch(descriptors, k=2)
        for pair in matches:
            if len(pair) == 2 and pair[0].distance < SETTINGS.descriptor_ratio * pair[1].distance:
                groups[pair[0].imgIdx].append(pair[0])
        ranked = []
        for index, matches in groups.items():
            # Repeated text must not vote repeatedly for a single reference point.
            unique = {}
            for match in sorted(matches, key=lambda item: item.distance):
                unique.setdefault(match.trainIdx, match)
            if len(unique) >= SETTINGS.min_matches:
                ranked.append((self.views[index], list(unique.values())))
        return sorted(ranked, key=lambda item: len(item[1]), reverse=True)[:SETTINGS.candidate_limit]
