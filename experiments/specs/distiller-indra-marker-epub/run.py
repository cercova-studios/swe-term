"""Execute the preregistered three-pipeline PDF benchmark in fresh run directories."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    spec = Path(__file__).resolve().parent
    repo = spec.parents[2]
    manifest = json.loads((spec / 'manifest.json').read_text())
    subprocess.run(['just', 'experiment-ready', manifest['experiment_id']], cwd=repo, check=True)
    fixture = json.loads((spec / 'fixtures/source.json').read_text())
    source = Path(fixture['path'])
    assert digest(source) == fixture['sha256'], 'Source changed after preregistration'
    run = repo / manifest['evidence']['raw_runs'] / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, run / 'source.pdf')
    shutil.copy2(spec / 'fixtures/cover.png', run / 'cover.png')
    shutil.copy2(spec / 'fixtures/oracle.json', run / 'oracle.json')
    shutil.copy2(spec / 'manifest.json', run / 'manifest.json')
    venv = Path('/tmp/distiller-pdf-benchmark-venv/bin')
    pandoc = shutil.which('pandoc')
    assert pandoc
    binary = Path('/Users/rohit/.local/bin/distiller')
    identity = {'manifest_sha256': digest(spec / 'manifest.json'), 'source_sha256': digest(source),
                'distiller_sha256': digest(binary), 'started_utc': time.strftime('%FT%TZ', time.gmtime()),
                'packages': subprocess.check_output([str(venv / 'python'), '-m', 'pip', 'freeze'], text=True)
                if (venv / 'pip').exists() else 'See isolated installation log and package metadata',
                'pandoc': subprocess.check_output([pandoc, '--version'], text=True).splitlines()[0],
                'resource_limit': 'Wall timeout enforced per process group; memory and CPU limits not enforced.'}
    (run / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    events = run / 'events.jsonl'

    def execute(command, folder, env, phase):
        started = time.monotonic()
        with (folder / f'{phase}.stdout').open('wb') as out, (folder / f'{phase}.stderr').open('wb') as err:
            process = subprocess.Popen(['/usr/bin/time', '-l', *command], cwd=folder, env=env,
                                       stdout=out, stderr=err, start_new_session=True)
            try:
                status = process.wait(timeout=manifest['environment']['resources']['timeout_seconds'])
                terminal = 'success' if status == 0 else 'failed'
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                status, terminal = process.returncode, 'timeout' if isinstance(error, subprocess.TimeoutExpired) else 'interrupted'
        record = {'pipeline': folder.parent.name, 'repetition': folder.name, 'phase': phase,
                  'command': command, 'environment_names': sorted(env), 'terminal': terminal,
                  'exit_code': status, 'wall_seconds': time.monotonic() - started,
                  'completed_utc': time.strftime('%FT%TZ', time.gmtime())}
        with events.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if terminal == 'interrupted':
            raise KeyboardInterrupt
        return terminal == 'success'

    for repetition in range(1, manifest['repetitions'] + 1):
        for variant in ('distiller', 'indra', 'marker'):
            assert digest(binary) == identity['distiller_sha256'], 'Distiller changed during frozen benchmark'
            folder = run / variant / str(repetition)
            folder.mkdir(parents=True)
            common = {'PATH': str(venv) + os.pathsep + os.environ['PATH'], 'TMPDIR': '/tmp',
                      'HOME': '/tmp/distiller-pdf-benchmark-home', 'HF_HOME': '/tmp/distiller-pdf-benchmark-home/huggingface',
                      'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
                      'HF_HUB_DISABLE_TELEMETRY': '1'}
            output = folder / 'book.epub'
            if variant == 'distiller':
                key = os.environ.get('OPENROUTER_API_KEY')
                assert key, 'Inject only OPENROUTER_API_KEY with Doppler'
                env = dict(common, HOME=str(Path.home()), OPENROUTER_API_KEY=key,
                           DISTILLER_PDF_CACHE=str(folder / 'cache'))
                command = [str(binary), str(run / 'source.pdf'), '--mode', 'epub', '--cover',
                           str(run / 'cover.png'), '-o', str(output)]
            else:
                env = common
                command = ([str(venv / 'indra-ocr'), str(run / 'source.pdf'), '--math', '--cores', '4', '--out', str(folder)]
                           if variant == 'indra' else
                           [str(venv / 'marker_single'), str(run / 'source.pdf'), '--mode', 'fast', '--output_format',
                            'markdown', '--disable_multiprocessing', '--output_dir', str(folder)])
            success = execute(command, folder, env, 'extract')
            if success and variant != 'distiller':
                markdown = list(folder.rglob('*.md'))
                assert len(markdown) == 1, ('Expected one native Markdown output', markdown)
                success = execute([pandoc, str(markdown[0]), '--from=markdown+raw_html+tex_math_dollars', '--to=epub3',
                         '--mathml', '--toc', '--toc-depth=6', '--epub-cover-image=' + str(run / 'cover.png'),
                         '--metadata=title:' + fixture['title'], '--metadata=lang:en',
                         '--resource-path=' + str(markdown[0].parent), '-o', str(output)], folder, env, 'package')
            if success and output.exists():
                (folder / 'output.json').write_text(json.dumps({'epub_sha256': digest(output)}, indent=2) + '\n')
    print('Run artifacts: ' + str(run), flush=True)


if __name__ == '__main__':
    main()
