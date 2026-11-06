"""Save and retrieve immutable image evidence using content hashes."""
import hashlib
import os
from pathlib import Path
import re
import tempfile


class EvidenceStore:
    """Manage immutable image files in an application-owned directory."""
    def __init__(self, root: str | Path):
        """Create the directory used for saved evidence.

        Args:
            root: Application-owned directory for image evidence.

        Raises:
            OSError: The storage directory cannot be created.
        """
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, media_type: str) -> str:
        # Signature checks are not image decoding. Dimensions come from capture.
        """Save image bytes atomically under a content-derived key.

        Args:
            data: Captured image bytes with a JPEG or PNG signature.
            media_type: Declared image/jpeg or image/png media type.

        Returns:
            The SHA-256 digest and image extension used as the evidence key.

        Raises:
            ValueError: The type, signature, path, or existing evidence is invalid.
            OSError: Writing or linking the evidence file fails.
        """
        signatures = {'image/jpeg': (b'\xff\xd8\xff', 'jpg'),
                      'image/png': (b'\x89PNG\r\n\x1a\n', 'png')}
        if media_type not in signatures:
            raise ValueError('Only JPEG and PNG evidence is supported')
        signature, extension = signatures[media_type]
        if not data.startswith(signature):
            raise ValueError('Evidence does not match its declared image type')
        key = hashlib.sha256(data).hexdigest() + '.' + extension
        destination = self.resolve(key)
        descriptor, temporary = tempfile.mkstemp(dir=self.root)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != data:
                    raise ValueError('Existing evidence failed integrity verification')
        finally:
            Path(temporary).unlink(missing_ok=True)
        return key

    def resolve(self, key: str) -> Path:
        """Resolve an evidence key within the configured storage directory.

        Args:
            key: SHA-256 digest followed by a supported image extension.

        Returns:
            The evidence path, which may not exist yet.

        Raises:
            ValueError: The key is malformed or the path is a symbolic link.
        """
        if not re.fullmatch(r'[a-f0-9]{64}\.(jpg|png)', key):
            raise ValueError('Invalid evidence key')
        path = self.root / key
        if path.is_symlink():
            raise ValueError('Evidence cannot be a symbolic link')
        return path

    def read(self, key: str) -> bytes:
        """Read image evidence and verify its content hash.

        Args:
            key: Content-derived key returned by save.

        Returns:
            The verified image bytes.

        Raises:
            ValueError: The key, path, or content hash is invalid.
            OSError: The evidence file cannot be read.
        """
        data = self.resolve(key).read_bytes()
        if hashlib.sha256(data).hexdigest() != key.split('.')[0]:
            raise ValueError('Evidence integrity check failed')
        return data
