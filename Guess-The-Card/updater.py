from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile
import zlib

from package_layout import REQUIRED, allowed

class UpdateError(RuntimeError):
    pass


REPOSITORY = 'senkoisbroken/Guess-The-Card'
RELEASE_API = 'https://api.github.com/repos/' + REPOSITORY + '/releases/latest'
ZIP_NAME = 'Guess-The-Card-Setup.zip'


class GitHubRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != 'https' or parsed.netloc not in (
                'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'):
            raise UpdateError('GitHub redirected to an unexpected download host.')
        return super().redirect_request(request, fp, code, message, headers, newurl)


def read_https(url, limit):
    if urlsplit(url).scheme != 'https':
        raise UpdateError('Update downloads must use HTTPS.')
    request = Request(url, headers={'User-Agent': 'Guess-The-Card-Updater',
                                    'Accept': 'application/vnd.github+json',
                                    'X-GitHub-Api-Version': '2026-03-10'})
    try:
        with build_opener(GitHubRedirect()).open(request, timeout=20) as response:
            if urlsplit(response.geturl()).scheme != 'https':
                raise UpdateError('The download redirected to an insecure connection.')
            data = response.read(limit+1)
            if len(data) > limit:
                raise UpdateError('The update response is larger than allowed.')
            return data
    except HTTPError as exc:
        message = ('No published update was found.' if exc.code == 404
                   else 'GitHub could not serve the update (HTTP '+str(exc.code)+'). Try again later.')
        raise UpdateError(message) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise UpdateError('Could not download the update. Check your internet connection.') from exc


def download_latest(folder):
    try:
        release = json.loads(read_https(RELEASE_API, 1024*1024))
        if release.get('draft') or release.get('prerelease'):
            raise UpdateError('No stable published update was found.')
        assets = {}
        for asset in release['assets']:
            if asset.get('name') in (ZIP_NAME, 'checksums.sha256'):
                if asset['name'] in assets:
                    raise UpdateError('The release contains duplicate update assets.')
                url = asset['browser_download_url']
                parsed = urlsplit(url)
                if (parsed.scheme != 'https' or parsed.netloc != 'github.com'
                        or not parsed.path.startswith('/'+REPOSITORY+'/releases/download/')
                        or parsed.query or parsed.fragment):
                    raise UpdateError('The release contains an unexpected download URL.')
                assets[asset['name']] = url
        if set(assets) != {ZIP_NAME, 'checksums.sha256'}:
            raise UpdateError('This release is missing its update ZIP or checksums.')
        checksum_text = read_https(assets['checksums.sha256'], 16*1024).decode('ascii')
        digests = []
        for line in checksum_text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip('*') == ZIP_NAME:
                digests.append(parts[0].lower())
        if len(digests) != 1 or re.fullmatch(r'[0-9a-f]{64}', digests[0]) is None:
            raise UpdateError('The release checksum is missing or invalid.')
        data = read_https(assets[ZIP_NAME], 96*1024*1024)
        if hashlib.sha256(data).hexdigest() != digests[0]:
            raise UpdateError('The download failed its SHA-256 check. No update was applied.')
        path = Path(folder) / ZIP_NAME
        path.write_bytes(data)
        return path
    except UpdateError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        raise UpdateError('Cannot read the published update: '+str(exc)) from exc


def update_latest(install_dir):
    ensure_macro_closed(Path(install_dir).resolve())
    with tempfile.TemporaryDirectory(prefix='card-update-') as folder:
        return update_zip(download_latest(folder), install_dir)


def safe_name(name):
    return allowed(name)


def read_package(path):
    try:
        files = {}
        seen = set()
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 10000 or sum(entry.file_size for entry in entries) > 96*1024*1024:
                raise UpdateError('Update ZIP is too large.')
            setup_layout = any(entry.filename == 'Guess-The-Card/App Files/package_info.json'
                               for entry in entries)
            for entry in entries:
                name = entry.filename
                if name.casefold() in seen:
                    raise UpdateError('Duplicate ZIP entry: ' + name)
                seen.add(name.casefold())
                if (not name.startswith('Guess-The-Card/') or '\\' in name
                        or '..' in PurePosixPath(name).parts
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise UpdateError('Unsafe ZIP entry: ' + name)
                relative = name[len('Guess-The-Card/'):]
                if setup_layout:
                    if relative in ('Start.bat', 'Update.bat', 'Read Me.txt') and not entry.is_dir():
                        continue
                    if not relative.startswith('App Files/'):
                        raise UpdateError('Unexpected ZIP file: ' + name)
                    relative = relative[len('App Files/'):]
                if entry.is_dir():
                    if relative.rstrip('/') not in ('', 'assets', 'assets/wiki', 'assets/observed'):
                        raise UpdateError('Unexpected ZIP directory: ' + name)
                    continue
                if not safe_name(relative):
                    raise UpdateError('Unexpected ZIP file: ' + name)
                files[relative] = archive.read(entry)
        raw_manifest = files.get('package_info.json')
        if raw_manifest is None:
            raise UpdateError('This ZIP has no package_info.json. Use a current distribution ZIP.')
        manifest = json.loads(raw_manifest)
        hashes = manifest.get('sha256')
        build = manifest.get('package_version', manifest.get('source_build'))
        if (not isinstance(hashes, dict) or not isinstance(build, str) or not build
                or set(hashes) != set(files)-{'package_info.json'}
                or not REQUIRED <= set(files)):
            raise UpdateError('The update manifest is incomplete.')
        for name, digest in hashes.items():
            if not isinstance(digest, str) or hashlib.sha256(files[name]).hexdigest() != digest:
                raise UpdateError('Update file failed its hash check: ' + name)
        return files, build
    except UpdateError:
        raise
    except (OSError, ValueError, TypeError, AttributeError, zipfile.BadZipFile,
            RuntimeError, KeyError, EOFError, zlib.error) as exc:
        raise UpdateError('Cannot read update ZIP: ' + str(exc)) from exc


def ensure_macro_closed(install_dir):
    """Read process command lines only; never open Roblox or send input."""
    if os.name != 'nt':
        return
    env = os.environ.copy()
    env['CARD_UPDATE_APP'] = str(Path(install_dir) / 'app.py')
    script = """$ErrorActionPreference='Stop'
$target=$env:CARD_UPDATE_APP
$found=Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
 Where-Object { $_.CommandLine -and $_.CommandLine.IndexOf($target,[StringComparison]::OrdinalIgnoreCase) -ge 0 }
if ($found) { exit 2 }
"""
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script],
                                env=env, capture_output=True, timeout=15,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UpdateError('Could not check whether the helper is closed. Try again.') from exc
    if result.returncode == 2:
        raise UpdateError('Close Guess the Card first, then run update.bat again.')
    if result.returncode:
        raise UpdateError('Windows could not check running helpers. No update was applied.')


def destination(root, name):
    target = root / name
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise UpdateError('Update path points outside the installation: ' + name)
    return target


@contextmanager
def update_lock(root):
    if os.name != 'nt':
        yield
        return
    import msvcrt
    try:
        destination(root, '.updates').mkdir(exist_ok=True)
        handle = destination(root, '.updates/update.lock').open('a+b')
    except OSError as exc:
        raise UpdateError('Cannot create the update lock: '+str(exc)) from exc
    try:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise UpdateError('Another update is running. Close it and try again.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def update_zip(zip_path, install_dir):
    files, build = read_package(zip_path)
    root = Path(install_dir).resolve()
    if not root.is_dir():
        raise UpdateError('The installation folder does not exist.')
    ensure_macro_closed(root)
    with update_lock(root):
        ensure_macro_closed(root)
        cleanup_updates(root)
        return apply_files(files, build, root)


def _plain_directory(path):
    """Do not traverse symlinks or Windows reparse points (including junctions)."""
    info = path.lstat()
    return (stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            and not getattr(info, 'st_file_attributes', 0)
            & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))


def _remove_update_tree(path, updates):
    """Remove only direct update children whose entire tree stays in .updates."""
    if path.parent != updates or not _plain_directory(path):
        return
    pending = [path]
    while pending:
        directory = pending.pop()
        if not directory.resolve().is_relative_to(updates):
            return
        for child in directory.iterdir():
            info = child.lstat()
            if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0)
                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
                return
            if stat.S_ISDIR(info.st_mode):
                pending.append(child)
    shutil.rmtree(path)


def cleanup_updates(root, current_backup=None):
    """Best-effort housekeeping while holding update_lock; prune only on success.

    Without a current backup, clear crashed staging directories only. Supplying
    the successfully created backup also keeps the newest five backups, always
    reserving one slot for that current backup regardless of its timestamp.
    """
    try:
        root = Path(root).resolve()
        updates = root / '.updates'
        if not _plain_directory(updates) or updates.resolve() != updates:
            return
        backups = []
        for path in updates.iterdir():
            try:
                if not _plain_directory(path):
                    continue
                if path.name.startswith('staging-'):
                    _remove_update_tree(path, updates)
                elif path.name.startswith('backup-'):
                    backups.append(path)
            except OSError:
                continue
        if current_backup is None:
            return
        current_backup = Path(current_backup).resolve()
        backups.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
        retained = {current_backup}
        for path in backups:
            if len(retained) < 5:
                retained.add(path)
        for path in backups:
            if path not in retained:
                try:
                    _remove_update_tree(path, updates)
                except OSError:
                    pass
    except (OSError, RuntimeError):
        # Housekeeping never invalidates an otherwise successful update.
        pass


def apply_files(files, build, root):
    changes = []
    try:
        for name in sorted(files, key=lambda name: (name == 'package_info.json', name)):
            target = destination(root, name)
            if not target.is_file() or target.read_bytes() != files[name]:
                changes.append(name)
    except OSError as exc:
        raise UpdateError('Cannot inspect installed files: ' + str(exc)) from exc
    if not changes:
        return {'updated': False, 'build': build, 'backup': None}
    tag = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    updates = destination(root, '.updates')
    stage = updates / ('staging-'+tag)
    backup = updates / ('backup-'+tag)
    existed = set()
    applied = []
    try:
        stage.mkdir(parents=True)
        backup.mkdir()
        for index, name in enumerate(changes):
            target = destination(root, name)
            (stage / str(index)).write_bytes(files[name])
            if target.exists():
                existed.add(name)
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
        for index, name in enumerate(changes):
            target = destination(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / str(index), target)
            applied.append(name)
    except (OSError, UpdateError) as exc:
        failed = []
        for name in reversed(applied):
            try:
                target = destination(root, name)
                if name in existed:
                    shutil.copy2(backup / name, target)
                else:
                    target.unlink(missing_ok=True)
            except (OSError, UpdateError):
                failed.append(name)
        message = ('Update failed; replaced files were restored.' if not failed
                   else 'Update failed; some files could not be restored: '+', '.join(failed))
        raise UpdateError(message+' Backup: '+str(backup)+' Details: '+str(exc)) from exc
    finally:
        if stage.is_dir():
            for path in stage.iterdir():
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                stage.rmdir()
            except OSError:
                pass
    cleanup_updates(root, current_backup=backup)
    return {'updated': True, 'build': build, 'backup': str(backup)}


def main():
    import sys
    import tkinter as tk
    from tkinter import filedialog, messagebox
    root = tk.Tk()
    root.withdraw()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path == '--local':
        path = filedialog.askopenfilename(title='Select the new Guess the Card setup ZIP',
                                          filetypes=[('Setup ZIP', '*.zip')])
        if not path:
            root.destroy()
            return
    root.title('Guess the Card update')
    tk.Label(root, text='Applying local update...' if path else 'Checking GitHub for updates...',
             padx=25, pady=20).pack()
    root.resizable(False, False)
    root.protocol('WM_DELETE_WINDOW', lambda: None)
    root.deiconify()
    results = queue.Queue()

    def worker():
        try:
            install = Path(__file__).resolve().parent
            results.put((True, update_zip(path, install) if path else update_latest(install)))
        except Exception as exc:
            results.put((False, str(exc)))

    def poll():
        try:
            success, result = results.get_nowait()
        except queue.Empty:
            root.after(75, poll)
            return
        root.withdraw()
        if success:
            text = ('Update complete.' if result['updated'] else 'This update is already installed.')
            text += '\n\nYour Python environment and logs were kept.\n'
            text += 'Open start.bat to run the helper.\nBuild: '+result['build']
            if result['backup']:
                text += '\nBackup: '+result['backup']
            messagebox.showinfo('Guess the Card update', text)
        else:
            messagebox.showerror('Guess the Card update', result)
        root.destroy()

    threading.Thread(target=worker, name='card-update', daemon=False).start()
    root.after(75, poll)
    root.mainloop()


if __name__ == '__main__':
    main()
