"""Immutable JSON records and process-owned locks for independent SLURM workers."""
from contextlib import contextmanager
import errno
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import tempfile
import time

from config import HOME, original_source


@contextmanager
def exclusive(path):
    with path.open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno != errno.EACCES:
                        raise
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.lockf(handle, fcntl.LOCK_EX)
        yield


def read_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def publish(path, result):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=f'.{path.name}.', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.close()
            os.link(temporary, path)
        finally:
            handle.close()
            temporary.unlink(missing_ok=True)


def checked(path, key, validate):
    record = read_json(path)
    if not isinstance(record, dict) or record.get('key') != key:
        raise ValueError(f'{path}: incompatible record')
    try:
        validate(record['result'])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'{path}: incomplete or invalid result') from exc
    return record['result']


def cached(path, key, compute, validate):
    if path.exists():
        return checked(path, key, validate)
    with exclusive(path.with_suffix('.lock')):
        if path.exists():
            return checked(path, key, validate)
        result = compute()
        validate(result)
        publish(path, {'key': key, 'result': result})
        return result


def finite_fields(record, fields):
    for field in fields:
        value = record[field]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Invalid numeric field: {field}')


def experiment_identity(X, Y, D, patient_hash):
    code = hashlib.sha256()
    for path in [original_source(), *sorted(HOME.glob('*.py'))]:
        code.update(path.name.encode())
        code.update(path.read_bytes().replace(b'\\r\\n', b'\\n'))
    data = hashlib.sha256(patient_hash.encode())
    for a in (X, Y, D):
        data.update(a.tobytes())
    return dict(code=code.hexdigest(), data=data.hexdigest(), n=len(Y),
                versions={p: version(p) for p in
                          ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pennylane',
                           'pennylane-lightning')})
