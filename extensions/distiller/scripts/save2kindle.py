#!/usr/bin/env python3
"""Convert a URL, HTML, PDF or EPUB to KFX and send it to your Kindle."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def run(command, timeout=14400, **kwargs):
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            if process.wait(timeout=timeout):
                raise ValueError(command[0] + ' failed; see diagnostics above')
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise


class PDFRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        url = urlsplit(newurl)
        if url.scheme not in ('http', 'https') or url.username is not None or url.password is not None:
            raise ValueError('PDF redirect must use HTTP(S) without credentials')
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def download_pdf(url, destination):
    limit = 200 * 1024 * 1024
    deadline = time.monotonic() + 60
    request = Request(url, headers={'User-Agent': 'save2kindle/1.0'})
    with build_opener(PDFRedirects()).open(request, timeout=15) as response, destination.open('wb') as output:
        size = 0
        signature = b''
        while chunk := response.read1(64 * 1024):
            size += len(chunk)
            if size > limit or time.monotonic() > deadline:
                raise ValueError('PDF download exceeded 200 MiB or 60 seconds')
            signature += chunk[:5 - len(signature)]
            if not b'%PDF-'.startswith(signature):
                raise ValueError('PDF link did not return a PDF document')
            output.write(chunk)
        if signature != b'%PDF-':
            raise ValueError('PDF download was empty or incomplete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='HTTP(S) URL or local HTML, PDF, EPUB or KFX file')
    parser.add_argument('-o', '--output', type=Path, help='New local KFX path; existing files are never overwritten')
    parser.add_argument('--cover', type=Path, help='Use a finished PNG/JPEG cover for URL/HTML/PDF conversion')
    args = parser.parse_args()
    try:
        source = args.source
        url = urlsplit(source)
        is_url = url.scheme.lower() in ('http', 'https')
        if is_url:
            if not url.hostname or url.username is not None or url.password is not None or len(source) > 8192 or any(c.isspace() or ord(c) < 32 for c in source):
                raise ValueError('URL must be HTTP(S), at most 8192 characters, without credentials or whitespace')
            stem = re.sub(r'[^A-Za-z0-9_-]', '-', unquote(url.path.rstrip('/').rsplit('/', 1)[-1] or url.hostname))[:70] or 'article'
            default = Path(stem + '-' + hashlib.sha256(source.encode()).hexdigest()[:12] + '.kfx')
            kind = 'url'
        else:
            if url.scheme and not Path(source).exists():
                raise ValueError('Only HTTP(S) links and local HTML, PDF, EPUB or KFX files are supported')
            source = str(Path(source).expanduser().resolve(strict=True))
            if not Path(source).is_file():
                raise ValueError('Source must be a regular file')
            kind = Path(source).suffix.lower()
            if kind not in ('.html', '.htm', '.pdf', '.epub', '.kfx'):
                raise ValueError('Unsupported local file; use HTML, PDF, EPUB or KFX')
            default = Path(source) if kind == '.kfx' else Path(source).with_suffix('.kfx')
        if kind in ('.epub', '.kfx') and args.cover:
            raise ValueError('EPUB/KFX inputs keep their existing cover; --cover is for URL/HTML/PDF inputs')
        if kind == '.kfx' and args.output:
            raise ValueError('KFX is already converted; omit --output to send it directly')
        target = (args.output or default).expanduser().absolute()
        commands = ['kindle-send']
        if kind != '.kfx':
            if target.suffix.lower() != '.kfx':
                raise ValueError('Output must have a .kfx extension')
            if os.path.lexists(target):
                raise ValueError('Output already exists. Send it with ' + shlex.join(['save2kindle', str(target)]) + ' or choose a new --output path')
            if not target.parent.is_dir():
                raise ValueError('Output directory does not exist')
            if kind == '.epub':
                conversion = ['epub-to-kfx', source, '-o', str(target)]
                commands.append('epub-to-kfx')
            else:
                conversion = ['distiller', source, '--mode', 'kfx', '-o', str(target)]
                commands += ['distiller', 'epub-to-kfx']
                if args.cover:
                    conversion += ['--cover', str(args.cover.expanduser().resolve(strict=True))]
                required = []
                if not args.cover and not os.environ.get('MODEL_API_KEY'):
                    required.append('MODEL_API_KEY')
                if (kind == '.pdf' or (is_url and unquote(url.path).lower().endswith('.pdf'))) and not os.environ.get('OPENROUTER_API_KEY'):
                    required.append('OPENROUTER_API_KEY')
                if required:
                    conversion = ['doppler', 'run', '--project', 'llm-experiments', '--config', 'dev_personal', '--only-secrets', ','.join(required), '--no-fallback', '--'] + conversion
                    commands.append('doppler')
            for command in commands:
                if not shutil.which(command):
                    raise ValueError('Required command is not installed: ' + command)
            # Probe before a potentially paid cover request, without creating the destination.
            with tempfile.TemporaryFile(dir=target.parent):
                pass
            with tempfile.TemporaryDirectory(prefix='save2kindle-') as work:
                # shortcut: recognize remote PDFs by URL path; download extensionless PDFs locally first.
                if is_url and unquote(url.path).lower().endswith('.pdf'):
                    downloaded = Path(work) / 'source.pdf'
                    download_pdf(source, downloaded)
                    conversion[conversion.index(source)] = str(downloaded)
                run(conversion, stdout=subprocess.DEVNULL)
            print('Saved KFX: ' + str(target), file=sys.stderr)
        elif not shutil.which('kindle-send'):
            raise ValueError('Required command is not installed: kindle-send')
        try:
            delivery_env = {key: value for key, value in os.environ.items() if key not in ('MODEL_API_KEY','OPENROUTER_API_KEY')}
            run(['kindle-send', str(target)], timeout=100, env=delivery_env)
        except (ValueError, subprocess.TimeoutExpired, OSError):
            print('KFX retained. Wake the Kindle, enable Wi-Fi, then retry: ' + shlex.join(['save2kindle', str(target)]), file=sys.stderr)
            return 1
        return 0
    except subprocess.TimeoutExpired:
        print('save2kindle: conversion timed out; delivery was not attempted', file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print('save2kindle: ' + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('save2kindle: interrupted; no delivery success was confirmed', file=sys.stderr)
        return 130


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
