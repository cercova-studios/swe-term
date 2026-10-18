#!/usr/bin/env python3
"""Public CLI journeys with fake conversion/delivery tools; no credentials or device."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import signal
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'save2kindle.py'
FAKE = '''#!/usr/bin/env python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['JOURNEY_LOG'], 'a') as log:
    log.write(json.dumps({'tool':name,'args':sys.argv[1:],'key_present':bool(os.environ.get('MODEL_API_KEY')),'router_present':bool(os.environ.get('OPENROUTER_API_KEY'))})+'\\n')
if name == 'doppler':
    args = sys.argv[sys.argv.index('--')+1:]
    os.execvp(args[0], args)
if os.environ.get('FAIL_TOOL') == name:
    sys.exit(9)
if name in ('distiller', 'epub-to-kfx'):
    source = sys.argv[1]
    if source.endswith('.pdf'):
        assert pathlib.Path(source).read_bytes().startswith(b'%PDF-')
    output = pathlib.Path(sys.argv[sys.argv.index('-o')+1])
    with output.open('xb') as book:
        book.write(b'CONT\\x02\\x00fake')
else:
    assert pathlib.Path(sys.argv[1]).read_bytes().startswith(b'CONT\\x02\\x00')
    print(json.dumps({'schema':'kindle-transfer/v1','verified':True}))
'''


class Journeys(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.bin = self.work / 'bin'
        self.bin.mkdir()
        for name in ('distiller', 'epub-to-kfx', 'kindle-send', 'doppler'):
            tool = self.bin / name
            tool.write_text(FAKE)
            tool.chmod(0o755)
        self.log = self.work / 'calls.jsonl'
        self.env = {k:v for k,v in os.environ.items() if k not in ('MODEL_API_KEY','OPENROUTER_API_KEY')}
        self.env.update(PATH=str(self.bin)+os.pathsep+os.environ['PATH'], JOURNEY_LOG=str(self.log))

    def invoke(self, *args, ok=True):
        result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=self.work,
                                env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def source(self, name, data=b'input'):
        path = self.work / name
        path.write_bytes(data)
        return path

    def test_url_doppler_conversion_and_verified_delivery(self):
        url = 'https://example.org/article?query=$(touch%20oops)&x=1'
        result = self.invoke(url)
        calls = self.calls()
        self.assertEqual([c['tool'] for c in calls], ['doppler','distiller','kindle-send'])
        self.assertEqual(calls[1]['args'][0], url)
        self.assertIn('--only-secrets', calls[0]['args'])
        self.assertTrue(json.loads(result.stdout)['verified'])
        self.assertFalse((self.work/'oops').exists())

    def test_local_inputs_and_uppercase_kfx(self):
        for name in ('a file.HTML', 'book.pdf', 'existing.epub', 'native.KFX'):
            with self.subTest(name=name):
                self.log.unlink(missing_ok=True)
                source = self.source(name, b'%PDF-input' if name.endswith('pdf') else b'CONT\x02\x00input')
                self.invoke(source)
                calls = self.calls()
                expected = ['kindle-send'] if name.endswith('KFX') else ['epub-to-kfx','kindle-send'] if name.endswith('epub') else ['doppler','distiller','kindle-send']
                self.assertEqual([c['tool'] for c in calls], expected)
                self.assertEqual(calls[-1]['args'][0], str(source if name.endswith('KFX') else source.with_suffix('.kfx')))

    def test_cover_and_injected_key_bypass_doppler(self):
        cover = self.source('cover.png')
        self.invoke('https://example.org/covered', '--cover', cover)
        self.assertEqual([c['tool'] for c in self.calls()], ['distiller','kindle-send'])
        self.log.unlink()
        self.env['MODEL_API_KEY'] = 'test-placeholder'
        self.env['OPENROUTER_API_KEY'] = 'test-router-placeholder'
        self.invoke('https://example.org/injected')
        calls = self.calls()
        self.assertEqual([c['tool'] for c in calls], ['distiller','kindle-send'])
        self.assertTrue(calls[0]['key_present'])
        self.assertFalse(calls[1]['key_present'])
        self.assertFalse(calls[1]['router_present'])

    def test_pdf_with_existing_cover_still_injects_reconstruction_key(self):
        source=self.source('covered.pdf',b'%PDF-test')
        self.invoke(source,'--cover',self.source('cover.png'))
        calls=self.calls()
        self.assertEqual(calls[0]['tool'],'doppler')
        self.assertEqual(calls[0]['args'][calls[0]['args'].index('--only-secrets')+1],'OPENROUTER_API_KEY')

    def test_existing_and_broken_symlink_outputs_fail_before_spending(self):
        for name, symlink in (('taken.kfx',False),('broken.kfx',True)):
            path = self.work / name
            path.symlink_to(self.work/'missing') if symlink else path.write_bytes(b'existing')
            self.invoke('https://example.org/a','-o',path,ok=False)
            self.assertEqual(self.calls(), [])

    def test_invalid_inputs_fail_before_spending(self):
        for args in [('ftp://example.org/a',),('https://user:pass@example.org/a',),
                     (self.source('file.txt'),),('missing.pdf',),
                     ('https://example.org/a','-o',self.work/'missing'/'a.kfx'),
                     (self.source('native.kfx'),'-o','other.kfx'),
                     (self.source('old.epub'),'--cover',self.source('cover.png'))]:
            with self.subTest(args=args):
                self.invoke(*args,ok=False)
                self.assertEqual(self.calls(), [])

    def test_conversion_failure_never_sends(self):
        self.env['FAIL_TOOL'] = 'distiller'
        self.invoke('https://example.org/failure',ok=False)
        self.assertNotIn('kindle-send',[c['tool'] for c in self.calls()])

    def test_failed_delivery_retains_book_and_retry_only_sends(self):
        self.env['FAIL_TOOL'] = 'kindle-send'
        target = self.work/'retained.kfx'
        result = self.invoke('https://example.org/a','-o',target,ok=False)
        self.assertTrue(target.is_file())
        self.assertIn('save2kindle',result.stderr)
        del self.env['FAIL_TOOL']
        self.log.unlink()
        self.invoke(target)
        self.assertEqual([c['tool'] for c in self.calls()],['kindle-send'])

    def test_missing_sender_fails_before_conversion(self):
        # Isolate PATH so a globally installed sender cannot satisfy this check.
        (self.bin/'kindle-send').unlink()
        self.env['PATH'] = str(self.bin)
        self.invoke('https://example.org/a',ok=False)
        self.assertEqual(self.calls(),[])

    def test_long_url_hash_survives_remote_filename_limit(self):
        self.invoke('https://example.org/'+'a'*300)
        target = Path(self.calls()[-1]['args'][0])
        self.assertLessEqual(len(target.stem),100)
        self.assertRegex(target.stem,r'-[0-9a-f]{12}$')

    def test_sigterm_stops_detached_conversion_without_sending(self):
        pidfile = self.work/'converter.pid'
        (self.bin/'distiller').write_text(
            '#!/usr/bin/env python3\nimport os,time\nfrom pathlib import Path\n'
            f'Path({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n')
        self.env['MODEL_API_KEY'] = 'test-placeholder'
        process = subprocess.Popen([sys.executable,str(SCRIPT),'https://example.org/cancel'],
                                   cwd=self.work,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic()+5
            while not pidfile.exists() and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue(pidfile.exists())
            child = int(pidfile.read_text())
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=5)
            self.assertEqual(process.returncode,130)
            with self.assertRaises(ProcessLookupError):
                os.kill(child,0)
            self.assertEqual(self.calls(),[])
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


class DownloadBoundary(unittest.TestCase):
    def test_pdf_download_validation_and_redirect_boundary(self):
        spec = importlib.util.spec_from_file_location('save2kindle',SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import io
        from urllib.request import Request
        with tempfile.TemporaryDirectory() as work:
            path = Path(work)/'source.pdf'
            for data, valid in ((b'%PDF-test',True),(b'<html>login</html>',False),(b'',False)):
                with patch.object(module,'build_opener') as opener:
                    opener.return_value.open.return_value = io.BytesIO(data)
                    if valid:
                        module.download_pdf('https://example.org/book.pdf',path)
                        self.assertEqual(path.read_bytes(),data)
                    else:
                        with self.assertRaises(ValueError):
                            module.download_pdf('https://example.org/book.pdf',path)
            with self.assertRaises(ValueError):
                module.PDFRedirects().redirect_request(Request('https://example.org/a'),None,302,'',{},'file:///etc/passwd')

    def test_fragmented_pdf_signature_and_deadline(self):
        spec = importlib.util.spec_from_file_location('save2kindle',SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as work, patch.object(module,'build_opener') as opener:
            path = Path(work)/'source.pdf'
            response = opener.return_value.open.return_value.__enter__.return_value
            response.read1.side_effect = [b'%',b'P',b'D',b'F',b'-body',b'']
            module.download_pdf('https://example.org/book.pdf',path)
            self.assertEqual(path.read_bytes(),b'%PDF-body')
            response.read1.side_effect = [b'%PDF-body']
            with patch.object(module.time,'monotonic',side_effect=[0,61]):
                with self.assertRaisesRegex(ValueError,'60 seconds'):
                    module.download_pdf('https://example.org/book.pdf',path)


if __name__ == '__main__':
    unittest.main()
