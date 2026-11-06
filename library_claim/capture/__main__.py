"""Print local quality measurements for a saved image."""
import argparse
from pathlib import Path

import cv2

from .quality import QualityAssessor


def main() -> None:
    """Assess an image supplied on the command line and print JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='JPEG or PNG image to assess')
    parser.add_argument('--previous', type=Path, help='Previous image for temporal change')
    parser.add_argument('--max-dimension', type=int, default=640)
    args = parser.parse_args()

    # Decode both inputs consistently so channel order matches OpenCV's contract.
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error('Could not decode the input image')
    previous = None
    if args.previous is not None:
        previous = cv2.imread(str(args.previous), cv2.IMREAD_COLOR)
        if previous is None:
            parser.error('Could not decode the previous image')
    try:
        result = QualityAssessor(max_dimension=args.max_dimension).assess(image, previous=previous)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(result.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
