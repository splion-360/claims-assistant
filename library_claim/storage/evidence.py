"""Immutable, content-addressed evidence files; no user-supplied paths."""
import hashlib
import os
from pathlib import Path
import re
import tempfile


class EvidenceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, media_type: str) -> str:
        # Signature checks are not image decoding. Dimensions come from capture.
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
        if not re.fullmatch(r'[a-f0-9]{64}\.(jpg|png)', key):
            raise ValueError('Invalid evidence key')
        path = self.root / key
        if path.is_symlink():
            raise ValueError('Evidence cannot be a symbolic link')
        return path

    def read(self, key: str) -> bytes:
        data = self.resolve(key).read_bytes()
        if hashlib.sha256(data).hexdigest() != key.split('.')[0]:
            raise ValueError('Evidence integrity check failed')
        return data
