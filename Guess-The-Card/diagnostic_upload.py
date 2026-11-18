"""Create and explicitly send a bounded, reviewable log snapshot.

The upload transport is intentionally separate from RemoteLoggingHandler: a
snapshot upload is a one-shot, synchronous operation and must not start the
background diagnostics thread or share its queue.
"""

from collections import namedtuple
import http.client
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid

ENDPOINT = 'https://zx9kgh0wp5.execute-api.us-east-2.amazonaws.com/logs'


MAX_BYTES = 32768           # Snapshot window read from the end of macro.log
MAX_REQUEST_BYTES = 24576   # Receiver body limit
PART_SIZE = 5               # Receiver batches up to 32; keep the upload small
RATE_DELAY = 1.1            # Receiver rate limit between accepted requests
MAX_ATTEMPTS = 3            # Total attempts for a temporary failure
RETRY_DELAYS = (1.1, 2.2)

OK = 'ok'
AUTH_FAILED = 'auth_failed'
REJECTED = 'rejected'
TEMPORARY = 'temporary'
CANCELLED = 'cancelled'

PostResult = namedtuple('PostResult', 'ok status detail')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def open_request(request, timeout=5):
    """Open one HTTPS request; module-level so tests can mock the network."""
    return urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout)


class UploadClient:
    """One-shot upload transport with no worker thread and no queue."""

    def __init__(self, endpoint, token=None, client_id=None):
        if endpoint != ENDPOINT:
            raise ValueError('Invalid diagnostics endpoint')
        if token is not None and not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', token):
            raise ValueError('Invalid diagnostics token')
        if client_id is not None and not re.fullmatch(r'[A-Za-z0-9_-]{8,64}', client_id):
            raise ValueError('Invalid diagnostics client ID')
        self.endpoint, self.token = endpoint, token
        self.client_id = client_id or uuid.uuid4().hex

    def post(self, lines):
        """Send one receiver request and classify the outcome."""
        try:
            body = json.dumps({'client_id': self.client_id, 'lines': lines}).encode('utf-8')
        except (TypeError, ValueError, UnicodeError):
            return PostResult(False, REJECTED, 'The log snapshot could not be encoded.')
        if len(body) > MAX_REQUEST_BYTES:
            return PostResult(False, REJECTED, 'A log part was too large to send.')

        headers = {'Content-Type': 'application/json'}
        if self.token is not None:
            headers['Authorization'] = 'Bearer ' + self.token
        request = urllib.request.Request(self.endpoint, body, headers, method='POST')
        try:
            with open_request(request, timeout=5) as response:
                status = getattr(response, 'status', None)
                raw = response.read(1024)
        except urllib.error.HTTPError as exc:
            code = exc.code
            try:
                exc.close()
            except Exception:
                pass
            if code in (401, 403):
                return PostResult(False, AUTH_FAILED,
                                  'Diagnostic server denied the request (HTTP %d).' % code)
            if code in (408, 429) or 500 <= code <= 599:
                return PostResult(False, TEMPORARY,
                                  'Server is temporarily unavailable (HTTP %d).' % code)
            return PostResult(False, REJECTED,
                              'Server rejected the log (HTTP %d).' % code)
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            return PostResult(False, TEMPORARY,
                              'Could not reach the diagnostic server: %s' % exc)
        except Exception as exc:  # Never crash the upload UI on a transport surprise.
            return PostResult(False, TEMPORARY, 'Upload transport error: %s' % exc)

        if status != 200:
            return PostResult(False, REJECTED,
                              'Unexpected server response (HTTP %s).' % status)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return PostResult(False, REJECTED, 'Unexpected server response.')
        if isinstance(payload, dict) and payload.get('ok') is True:
            return PostResult(True, OK, '')
        return PostResult(False, REJECTED, 'Server rejected the log.')


def snapshot_lines(root, token=''):
    path = Path(root) / 'macro.log'
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - MAX_BYTES))
        data = stream.read(MAX_BYTES).decode('utf-8', errors='replace')
    if not data.strip():
        raise ValueError('The log is empty')
    if token:
        data = data.replace(token, '[redacted]')
    data = re.sub(r'(?i)\b[a-z]:[\\/][^\r\n"\']+', '[path]', data)
    data = re.sub(r'(?i)(token|authorization|api[_-]?key|password|secret)(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+', r'\1\2[redacted]', data)
    data = re.sub(r'(?i)\bbearer\s+\S+', 'Bearer [redacted]', data)
    data = ''.join(c if c >= ' ' or c in '\n\t' else ' ' for c in data)
    return [data[i:i+400] for i in range(0, len(data), 400)]


def _notify(progress, message):
    if progress is not None:
        try:
            progress(message)
        except Exception:
            pass


def _wait(delay, cancelled):
    if cancelled is not None and cancelled():
        return False
    time.sleep(delay)
    return cancelled is None or not cancelled()


def _post_with_retry(client, payload, progress, cancelled):
    attempt = 0
    while True:
        if cancelled is not None and cancelled():
            return PostResult(False, CANCELLED, 'Upload cancelled.')
        attempt += 1
        result = client.post(payload)
        if result.ok or result.status != TEMPORARY or attempt >= MAX_ATTEMPTS:
            return result
        _notify(progress, 'Temporary problem; retrying (%d/%d)...' % (attempt + 1, MAX_ATTEMPTS))
        delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
        if not _wait(delay, cancelled):
            return PostResult(False, CANCELLED, 'Upload cancelled.')


def send_snapshot(root, token=None, lines=None, progress=None, cancelled=None):
    """Upload exactly ``lines`` after the caller has reviewed them.

    ``progress`` is called with short status strings from the worker thread;
    callers must marshal those onto their UI thread. ``cancelled`` is polled
    between parts and retries so a closed window can stop further sends.
    """
    try:
        lines = snapshot_lines(root, token) if lines is None else list(lines)
        if not lines:
            return False, 'Log is empty.'
        client = UploadClient(ENDPOINT, token)
    except (OSError, ValueError) as exc:
        return False, 'Could not prepare log: ' + str(exc)

    total = len(lines)
    upload_id = uuid.uuid4().hex[:12]
    _notify(progress, 'Uploading %d log parts...' % total)

    result = _post_with_retry(
        client, ['log_upload_begin %s parts=%d' % (upload_id, total)],
        progress, cancelled)
    if result.status == CANCELLED:
        return False, result.detail
    if not result.ok:
        return False, 'Upload failed: ' + result.detail

    for offset in range(0, total, PART_SIZE):
        if not _wait(RATE_DELAY, cancelled):
            return False, 'Upload cancelled.'
        batch = ['log %s %d/%d %s' % (upload_id, index + 1, total, line)
                 for index, line in enumerate(lines[offset:offset + PART_SIZE], offset)]
        result = _post_with_retry(client, batch, progress, cancelled)
        if result.status == CANCELLED:
            return False, result.detail
        if not result.ok:
            return False, ('Upload failed after %d of %d parts: %s'
                           % (offset, total, result.detail))
        _notify(progress, 'Uploaded %d of %d parts...'
                % (min(offset + PART_SIZE, total), total))

    if not _wait(RATE_DELAY, cancelled):
        return False, 'Upload cancelled.'
    result = _post_with_retry(
        client, ['log_upload_end %s' % upload_id], progress, cancelled)
    if result.status == CANCELLED:
        return False, result.detail
    if not result.ok:
        return False, 'Upload failed before completion was confirmed: ' + result.detail
    return True, 'Log sent successfully (ID %s).' % upload_id
