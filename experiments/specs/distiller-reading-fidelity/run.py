#!/usr/bin/env python3
"""Two-source EPUB case study. Requires test-only lxml, pypdf and Pillow."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
import posixpath
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unicodedata
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zipfile import ZipFile

from lxml import html, etree
from PIL import Image
from pypdf import PdfReader

SPEC = Path(__file__).resolve().parent
ROOT = SPEC.parents[2]
NS = {'h': 'http://www.w3.org/1999/xhtml'}
EPUBCHECK = Path('/tmp/distiller-epubcheck/epubcheck-5.4.0/epubcheck.jar')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def pixels(data):
    with Image.open(BytesIO(data)) as image:
        image = image.convert('RGBA')
        return image.size, hashlib.sha256(image.tobytes()).hexdigest()


def source_pixels(image):
    raw = image.indirect_reference.get_object().get_data()
    # pypdf's exported JPEG data is re-encoded; compare the original DCT stream.
    return pixels(raw if raw.startswith(b'\xff\xd8\xff') else image.data)


def inspect(path, source):
    with ZipFile(path) as book:
        container = etree.fromstring(book.read('META-INF/container.xml'))
        opf_path = container.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile').get('full-path')
        opf = etree.fromstring(book.read(opf_path))
        entries=opf.findall('{http://www.idpf.org/2007/opf}manifest/{http://www.idpf.org/2007/opf}item')
        items = {e.get('id'):e.get('href') for e in entries}
        nav_href=next(e.get('href') for e in entries if 'nav' in e.get('properties','').split())
        nav=etree.fromstring(book.read(posixpath.normpath(posixpath.join(posixpath.dirname(opf_path),nav_href))))
        nav_titles=[norm(''.join(e.itertext())) for e in nav.xpath('.//h:nav[@role="doc-toc"]//h:a',namespaces=NS)]
        paths = [posixpath.normpath(posixpath.join(posixpath.dirname(opf_path), items[e.get('idref')])) for e in opf.findall('{http://www.idpf.org/2007/opf}spine/{http://www.idpf.org/2007/opf}itemref')]
        docs = [etree.fromstring(book.read(n)).find('h:body',NS) for n in paths]
        text = norm(' '.join(''.join(d.itertext()) for d in docs))
        headings = [(int(e.tag[-1]), norm(''.join(e.itertext()))) for d in docs
                    for e in d.xpath('.//h:h1|.//h:h2|.//h:h3|.//h:h4|.//h:h5|.//h:h6', namespaces=NS)]
        images = [e for d in docs for e in d.xpath('.//h:img', namespaces=NS)]
        result = {'image_elements': len(images), 'headings': len(headings),
                  'embedded_image_files': sum('/media/' in n for n in book.namelist())}
        all_media = []
        for name,d in zip(paths,docs):
            for e in d.xpath('.//h:img',namespaces=NS):
                asset=posixpath.normpath(posixpath.join(posixpath.dirname(name),e.get('src')))
                if not asset.endswith('.svg'): all_media.append(pixels(book.read(asset)))
        matched = Counter(source['pixels']) & Counter(all_media)
        result['source_images_with_exact_decoded_pixels'] = sum(matched.values())
        if source['kind'] == 'article':
            result.update({
                'source_headings_retained': sum(h in headings for h in source['headings']),
                'source_headings_total': len(source['headings']),
                'source_headings_in_navigation':sum(h[1] in nav_titles for h in source['headings']),
                'source_text_blocks_retained': sum(t in text for t in source['blocks']),
                'source_text_blocks_total': len(source['blocks']),
                'article_image_total': source['images'],
                'chrome_markers_present': [t for t in ['Privacy choices', 'Responsible disclosure', 'Status', 'Cookie policy'] if norm(t) in text],
            })
        else:
            anchors = [e.get('id') for d in docs for e in d.iter() if e.get('id','').startswith('page-')]
            outline_links = [(norm(''.join(a.itertext())), a.get('href','').split('#')[-1]) for d in docs for a in d.xpath('.//h:div[@id="distiller-bookmarks"]//h:a', namespaces=NS)]
            result.update({'source_pages': source['pages'], 'page_anchors': len(set(anchors)),
                           'source_raster_images': len(source['pixels']),
                           'source_images_with_exact_decoded_pixels': sum(matched.values()),
                           'selected_text_presence': {t: norm(t) in text for t in source['checks']},
                           'source_outline_titles_retained_as_headings': sum(any(norm(t)==h[1] for h in headings) for t in source['outline']),
                           'source_outline_titles_in_navigation':sum(norm(t) in nav_titles for t in source['outline']),
                           'source_outline_titles_total': len(source['outline']),
                           'original_outline_links_with_correct_page':sum((norm(t), f'page-{p+1}') in outline_links for t,p in source['outline_destinations'])})
        return result


def main():
    subprocess.run(['just','experiment-ready',SPEC.name], cwd=ROOT, check=True)
    manifest = json.loads((SPEC/'manifest.json').read_text())
    sources = json.loads((SPEC/'fixtures/sources.json').read_text())
    pdf = Path(sources['pdf']['path'])
    assert digest(pdf) == sources['pdf']['sha256']
    run = ROOT/manifest['evidence']['raw_runs']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run.mkdir(parents=True,exist_ok=False)
    tools = {name: Path(shutil.which(name)).resolve() for name in ['distiller','xberg','pandoc']}
    java_properties = subprocess.run(['java','-XshowSettings:properties','-version'],capture_output=True,text=True,check=True).stderr
    java_home = next(line.split('=',1)[1].strip() for line in java_properties.splitlines() if line.strip().startswith('java.home ='))
    tools['java'] = Path(java_home)/'bin/java'
    library = Path.home()/'.local/share/distiller/lib/libpdfium.dylib'
    reader = PdfReader(pdf)
    outline = []
    outline_destinations = []
    def titles(items):
        for item in items:
            if isinstance(item,list): titles(item)
            else:
                outline.append(item.title)
                outline_destinations.append((item.title, reader.get_destination_page_number(item)))
    titles(reader.outline)
    oracle_pdf = {'kind':'pdf','pages':len(reader.pages), 'outline':outline, 'outline_destinations':outline_destinations,
                  'pixels':[source_pixels(i) for p in reader.pages for i in p.images],
                  'checks':['We have machines now','Ev- ery','Amer- ican','com- pute','GPT- 3']}
    document = html.fromstring((SPEC/'fixtures/anthropic-agents.html').read_bytes())
    article = document.xpath('//article')[0]
    oracle_article = {'kind':'article',
                      'headings':[(int(e.tag[-1]),norm(e.text_content())) for e in article.xpath('.//h2|.//h3')],
                      'blocks':[norm(e.text_content()) for e in article.xpath('.//p|.//li[not(.//p)]') if norm(e.text_content())],
                      'images':len(article.xpath('.//img'))}
    # Shared input adapter caches resources; it never selects or removes content.
    assets=run/'article-assets'; assets.mkdir()
    cached={}
    for e in document.iter():
        for attr in ['href','src']:
            value=e.get(attr)
            if value and not value.startswith(('#','data:','mailto:','javascript:')):
                e.set(attr,urljoin(sources['article']['url'],value))
        if e.tag=='img' and e.get('src'):
            remote=e.get('src')
            if remote not in cached:
                with urlopen(Request(remote,headers={'User-Agent':'Mozilla/5.0 (compatible; distiller/1.0; +https://example.com)'}),timeout=20) as response:
                    data=response.read(4*1024*1024+1)
                    assert len(data)<=4*1024*1024, 'Shared image input exceeds cap'
                    (assets/f'{len(cached)}.original').write_bytes(data)
                    media=response.headers.get_content_type()
                extension='svg' if media=='image/svg+xml' else Image.open(BytesIO(data)).format.lower()
                if extension=='webp':
                    buffer=BytesIO(); Image.open(BytesIO(data)).save(buffer,format='PNG'); data=buffer.getvalue(); extension='png'
                name=f'{len(cached)}.{extension}'
                (assets/name).write_bytes(data)
                cached[remote]=f'article-assets/{name}'
            e.set('src',cached[remote]); e.attrib.pop('srcset',None)
    (run/'asset-urls.json').write_text(json.dumps(cached,indent=2))
    oracle_article['pixels']=[pixels((run/e.get('src')).read_bytes()) for e in article.xpath('.//img')]
    adapted = html.tostring(document,encoding='utf-8',method='html')
    (run/'article-adapted.html').write_bytes(adapted)
    identity = {'manifest_sha256':digest(SPEC/'manifest.json'), 'source_digest':manifest['source']['content_digest'],
                'runner_sha256':digest(Path(__file__)), 'pdf_sha256':digest(pdf),
                'adapted_article_sha256':digest(run/'article-adapted.html'),
                'tools':{n:{'path':str(p),'sha256':digest(p)} for n,p in tools.items()},
                'pdfium_sha256':digest(library),'epubcheck_sha256':digest(EPUBCHECK),
                'environment_names':['PATH','HOME','TMPDIR'], 'resource_limits':manifest['environment']['resources']}
    (run/'identity.json').write_text(json.dumps(identity,indent=2))
    results = {}
    with tempfile.TemporaryDirectory(prefix='distiller-reading-') as temporary, (run/'terminal.jsonl').open('a') as events:
        work=Path(temporary); home=work/'home'; lib=home/'.local/share/distiller/lib'; lib.mkdir(parents=True)
        shutil.copy2(library,lib/library.name)
        env={'PATH':os.environ['PATH'],'HOME':str(home),'TMPDIR':str(work)}
        identity['temporary_directory']=temporary
        (run/'identity.json').write_text(json.dumps(identity,indent=2))
        def execute(label,command,cwd):
            before=time.perf_counter(); started=datetime.now(timezone.utc).isoformat()
            with (run/f'{label}.stdout').open('wb') as out,(run/f'{label}.stderr').open('wb') as err:
                process=subprocess.Popen(command,cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True)
                try: status=process.wait(timeout=manifest['environment']['resources']['timeout_seconds'])
                except BaseException as failure:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
                    if not isinstance(failure,subprocess.TimeoutExpired): raise
                    status='timeout'
            record={'label':label,'command':list(map(str,command)),'started':started,
                    'ended':datetime.now(timezone.utc).isoformat(),'exit':status,
                    'wall_seconds':time.perf_counter()-before,'peak_rss_bytes':None}
            events.write(json.dumps(record)+'\n'); events.flush()
            return record
        for case,oracle in [('article',oracle_article),('pdf',oracle_pdf)]:
            for variant in ['distiller','xberg-pandoc','pandoc']:
                label=f'{case}-{variant}'; folder=work/label; folder.mkdir()
                source=folder/('article.html' if case=='article' else 'book.pdf')
                if case=='article':
                    source.write_bytes(adapted)
                    shutil.copytree(assets,folder/'article-assets')
                else: shutil.copy2(pdf,source)
                output=folder/'book.epub'
                extraction=None
                if variant=='distiller':
                    command=[str(tools['distiller']),str(source),'--mode','epub','-o',str(output)]
                    if case=='article': command += ['--base-url',sources['article']['url']]
                    record=execute(label,command,folder)
                else:
                    packaging_source=source
                    if variant=='xberg-pandoc':
                        packaging_source=folder/'extracted.html'
                        command=[str(tools['xberg']),'extract',str(source),'--no-config-discovery','--no-cache','true','--ocr','false','--extract-images','true','--output-dir',str(folder),'--content-format','html','--format','text','-o',str(packaging_source)]
                        extraction=execute(label+'-extract',command,folder)
                        shutil.copytree(folder,run/(label+'-extraction'),ignore=shutil.ignore_patterns('book.pdf','article.html'))
                        if extraction['exit'] != 0:
                            results[label]={'extraction':extraction}; continue
                    title='Building effective agents' if case=='article' else 'Situational Awareness'
                    command=[str(tools['pandoc']),'-f','html' if packaging_source.suffix=='.html' else 'pdf','-t','epub3','--toc','--toc-depth=6','--preserve-tabs','--math-method=mathml','-M',f'title={title}','-M','lang=en',str(packaging_source),'-o',str(output)]
                    record=execute(label,command,folder)
                results[label]={'conversion':record}
                if extraction is not None: results[label]['extraction']=extraction
                results[label]['pipeline_wall_seconds']=record['wall_seconds'] + (extraction['wall_seconds'] if extraction else 0)
                if output.exists():
                    assert output.stat().st_size <= manifest['environment']['resources']['max_output_bytes']
                    saved=run/f'{label}.epub'; shutil.copy2(output,saved)
                    results[label]['artifact_sha256']=digest(saved)
                    try: results[label]['fidelity']=inspect(saved,oracle)
                    except Exception as failure: results[label]['inspection_error']=str(failure)
                    results[label]['epubcheck']=execute(label+'-epubcheck',[str(tools['java']),'-jar',str(EPUBCHECK),str(saved)],folder)
        (run/'summary.json').write_text(json.dumps(results,indent=2))
        artifacts={str(p.relative_to(run)):digest(p) for p in run.rglob('*') if p.is_file()}
        (run/'artifacts.json').write_text(json.dumps(artifacts,indent=2))
    print(run)
    print(json.dumps({k:{'exit':v.get('conversion',v.get('extraction',{})).get('exit'),'fidelity':v.get('fidelity'),'epubcheck_exit':v.get('epubcheck',{}).get('exit')} for k,v in results.items()},indent=2))


if __name__=='__main__':
    main()
