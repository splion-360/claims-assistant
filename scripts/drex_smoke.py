"""Send one JPEG to the live Drex frame check and print the decision."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library_claim.decisions.frame_check import check_frame  # noqa: E402


def main() -> None:
    """Check the JPEG named on the command line and print the answers as JSON."""
    if len(sys.argv) != 2:
        raise SystemExit('usage: python scripts/drex_smoke.py path/to/frame.jpg')
    print(json.dumps(check_frame(Path(sys.argv[1]).read_bytes()).model_dump(), indent=2))


if __name__ == '__main__':
    main()
