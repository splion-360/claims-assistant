"""Choose evidence-focused instructions for a selected library book."""


def capture_instruction(book: dict) -> str:
    """Choose one concrete action from unresolved visual evidence.

    Args:
        book: Inventory card with observation issues and identity candidates.

    Returns:
        A brief instruction suitable for speech and the camera panel.
    """
    observations = book.get('observations', [])
    issues = observations[-1].get('issues', []) if observations else []
    title = book.get('title') or 'the selected book'
    if 'blur' in issues:
        action = 'Hold the camera still for a moment with the book in view.'
    elif 'glare' in issues:
        action = 'Shift the camera slightly to the side to reduce the glare.'
    elif 'occlusion' in issues or 'cut_off' in issues:
        action = 'Adjust your viewing angle to show the whole visible spine without moving the books.'
    elif book.get('status') == 'unresolved' or book.get('candidates'):
        action = 'Show that book together with the books beside it so I can associate the views.'
    else:
        action = 'Move the camera a little closer to the shelf so the title is readable.'
    return f'For {title}: {action} Tap Ready to capture when positioned.'
