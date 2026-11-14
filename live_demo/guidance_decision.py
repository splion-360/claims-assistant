"""Let Drex choose whether and how the agent guides the sweep (D6)."""
from dataclasses import asdict, dataclass

import httpx

from library_claim.config import CONFIG
from library_claim.decisions.drex import DrexError, choice, decide

# Fixed sentences: instant to speak and unable to invent facts about the shelf.
QUALITY_PROMPTS = {
    'slow_down': 'Slow down a little so I can read the spines.',
    'move_closer': 'Move a little closer to this shelf so I can read the titles.',
    'reduce_glare': 'Shift the camera slightly to the side to avoid the glare.',
    'more_light': "It's too dark here to read the spines.",
    'reposition': 'Something is blocking the books; shift the camera so I can see them.',
    'pan_back': 'Pan back a little; I only saw part of that shelf.',
}
LABELS = ('stay_quiet', *QUALITY_PROMPTS, 'read_title')
CRITERIA = {
    'stay_quiet': 'Capture is fine or a prompt would repeat or interrupt; say nothing',
    'slow_down': 'Frames are blurred by motion',
    'move_closer': 'Books are too small or far away to read',
    'reduce_glare': 'Reflections or glare hide spine text',
    'more_light': 'The scene is too dark to read',
    'reposition': 'Something blocks the books',
    'pan_back': 'Part of a shelf was cut off at the frame edge and never seen whole',
    'read_title': 'A visible book still has no readable title; ask the user to hold steady on it',
}
INSTRUCTIONS = ('A user is sweeping a phone camera across home bookshelves while an inventory agent listens. '
                'Choose the single most useful spoken prompt now, or stay quiet.')


@dataclass
class GuidanceSignals:
    """What the agent knows about recent capture quality and inventory gaps."""
    recent_issues: list[tuple[str, float]]
    gate_failure_streak: int
    missing_title_books: int
    edge_cut_books: int
    last_prompt: str | None
    seconds_since_prompt: float | None
    phase: str


def choose_prompt(signals: GuidanceSignals, *, client: httpx.Client | None = None) -> str:
    """Ask Drex for the next guidance label.

    Args:
        signals: Current capture and inventory signals.
        client: Optional HTTP client, used by tests.

    Returns:
        One of LABELS; 'stay_quiet' when Drex fails or answers outside LABELS.
    """
    try:
        result = decide({'prompt': choice(INSTRUCTIONS, CRITERIA)}, state=asdict(signals), client=client)
    except DrexError:
        return 'stay_quiet'
    label = result.answers['prompt'].choice
    return label if label in LABELS else 'stay_quiet'


class QualityPromptLimiter:
    """Space capture-quality prompts so guidance helps without nagging."""

    def __init__(self):
        """Start with no prompts spoken."""
        self.last_any = float('-inf')
        self.last_by_label: dict[str, float] = {}

    def allow(self, label: str, now: float) -> bool:
        """Report whether a quality prompt may be spoken now.

        Args:
            label: Quality prompt label.
            now: Monotonic time in seconds.

        Returns:
            True when both the kind and same-label cooldowns have passed.
        """
        if now - self.last_any < CONFIG.drex.quality_prompt_cooldown_s:
            return False
        return now - self.last_by_label.get(label, float('-inf')) >= CONFIG.drex.same_prompt_cooldown_s

    def record(self, label: str, now: float) -> None:
        """Remember that a quality prompt was spoken.

        Args:
            label: Quality prompt label.
            now: Monotonic time in seconds.
        """
        self.last_any = now
        self.last_by_label[label] = now


def edge_cut_count(inventory: list[dict]) -> int:
    """Count books only ever seen cut off at the frame edge (a likely missed shelf area).

    Args:
        inventory: Inventory entries with per-observation issue lists.

    Returns:
        Number of books whose every observation is cut off.
    """
    return sum(1 for book in inventory
               if not book.get('excluded') and book.get('observations') and all('cut_off' in row.get('issues', []) for row in book['observations']))


# Instant local gate failures map to the prompt that fixes them.
GATE_PROMPTS = {'too_dark': 'more_light', 'too_bright': 'reduce_glare',
                'insufficient_texture_or_sharpness': 'slow_down'}


def gate_prompt(reason: str) -> str:
    """Pick the quality prompt for a local gate failure.

    Args:
        reason: Failure reason from `local_gate`.

    Returns:
        A QUALITY_PROMPTS label.
    """
    return GATE_PROMPTS.get(reason, 'slow_down')


def should_speak(*, user_speaking: bool, agent_speaking: bool) -> bool:
    """Apply the rule that guidance never talks over anyone.

    Args:
        user_speaking: The user is talking or a reply to them is being prepared.
        agent_speaking: The agent is already speaking.

    Returns:
        Whether a prompt may be spoken.
    """
    return not user_speaking and not agent_speaking
