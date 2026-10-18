"""PDF -> EPUB: native mathematical regions, colored vectors and crop closure."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from epub_journey import inspect, NS, PNG
from pdf_model_stub import ModelStub


def prose_journey(binary, root, model):
    source, book, cover = root/'prose.pdf', root/'prose.epub', root/'cover.png'
    cover.write_bytes(PNG)
    lines=['First paragraph uses 30+ minutes and a $100B+ estimate.',
           'This physical line continues the same paragraph,',
           'Another line retains its place in that paragraph.',
           'The paragraph finishes here.',
           'Second paragraph begins after a source blank gap.',
           'Its next line should remain in this paragraph.',
           'A further line provides normal leading evidence.',
           'The second paragraph finishes here.']
    c=canvas.Canvas(str(source),pagesize=(612,792));c.setTitle('Paragraph fidelity')
    c.setFont('Times-Bold',20);c.drawString(72,750,'Chapter One')
    c.setFont('Times-Roman',12)
    for y,text in zip([700,684,668,652,624,608,592,576],lines):c.drawString(72,y,text)
    c.save()
    for mode in ['prose_geometry','prose_continuation']:
        model.mode=mode;book=root/f'{mode}.epub'
        env=dict(os.environ,OPENROUTER_API_KEY='fixture-placeholder',OPENROUTER_BASE_URL=model.url,
                 DISTILLER_PDF_CACHE=str(root/f'{mode}-cache'))
        result=subprocess.run([binary,str(source),'--mode','epub','--cover',str(cover),'-o',str(book)],
                              env=env,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stderr
        _,docs,_,_=inspect(book)
        paragraphs=[''.join(p.itertext()) for d in docs.values() for p in d.findall('.//h:p',NS)]
        assert ' '.join(lines[:4]) in paragraphs,'Source paragraph was falsely split'
        assert ' '.join(lines[4:]) in paragraphs,'Source blank-gap paragraph was merged'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--epubcheck', type=Path)
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    with tempfile.TemporaryDirectory(prefix='distiller-technical-') as directory, ModelStub() as model:
        root = Path(directory)
        prose_journey(binary,root,model)
        ink = root/'tight-ink'; ink.mkdir()
        c=canvas.Canvas(str(ink/'source.pdf'),pagesize=(612,792))
        c.setFont('Times-Bold',20); c.drawString(50,700,'VISIBLE TITLE')
        c.line(50,696.5,220,696.5); c.save()
        result=subprocess.run([binary,str(ink/'source.pdf'),'--mode','epub','--pdf-worker',
                               '-o',str(ink/'source.json')],capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stderr
        source_ink=json.loads((ink/'source.json').read_text())
        title=next(s for s in source_ink['spans'] if s['text']=='VISIBLE TITLE')
        assert all(title['id'] not in r['spans'] for r in source_ink['regions']), 'Loose font bounds swallowed a separate heading'
        source, book, cover = root/'technical.pdf', root/'technical.epub', root/'cover.png'
        cover.write_bytes(PNG)
        c = canvas.Canvas(str(source), pagesize=(612,792))
        c.setTitle('Native technical fidelity')
        c.beginForm('clipped-title',0,0,180,50)
        c.setFont('Times-Roman',12)
        c.drawString(0,70,'Clipped Form title')
        c.drawString(-stringWidth('XX','Times-Roman',12),20,'XXVisible mixed label')
        c.endForm()
        c.saveState(); c.translate(350,700); c.doForm('clipped-title'); c.restoreState()
        c.beginForm('disjoint-inner',0,0,20,20)
        c.setFont('Times-Roman',12); c.drawString(0,5,'Disjoint hidden')
        c.endForm()
        c.beginForm('disjoint-outer',0,0,20,20)
        c.saveState(); c.translate(50,0); c.doForm('disjoint-inner'); c.restoreState()
        c.endForm()
        c.saveState(); c.translate(350,500); c.doForm('disjoint-outer'); c.restoreState()
        c.beginForm('empty-clip',0,0,0,20)
        c.setFont('Times-Roman',12); c.drawString(0,5,'Zero area hidden')
        c.endForm()
        c.saveState(); c.translate(350,450); c.doForm('empty-clip'); c.restoreState()
        c.setFont('Times-Bold',20); c.drawString(72,750,'Chapter One')
        c.setFont('Times-Roman',12); c.drawString(72,710,'Ordinary prose remains selectable before the equation.')
        c.drawString(72,650,'E = x')
        c.setFont('Times-Roman',8); c.drawString(104,656,'2')
        c.setFont('Symbol',12); c.drawString(150,650,'θ')
        c.setFont('Times-Roman',12); c.drawString(210,655,'a + b'); c.drawString(222,625,'c')
        c.drawString(490,650,'(12)')
        c.line(208,645,240,645)
        c.drawString(72,590,'Ordinary prose remains selectable after the equation.')
        for y in [565,535,485,455]: c.line(72,y,495,y)
        c.drawString(80,545,'Model'); c.drawString(255,545,'Complexity')
        c.drawString(80,510,'Recurrent'); c.drawString(255,510,'O(n d')
        c.setFont('Times-Roman',8); c.drawString(286,516,'2')
        c.setFont('Times-Roman',12); c.drawString(292,510,')')
        c.drawString(80,470,'Attention'); c.setFont('Symbol',12); c.drawString(255,470,'θ')
        c.setFont('Times-Roman',12); c.drawString(72,410,'The vector diagram retains both source colors and rotated labels.')
        c.beginForm('network',-5,-5,205,85)
        c.setStrokeColorRGB(1,0,0); c.setFillColorRGB(1,0,0); c.rect(0,0,80,80,fill=0)
        c.setStrokeColorRGB(0,0,1); c.setFillColorRGB(0,0,1); c.rect(120,0,80,80,fill=0)
        c.line(80,40,120,40)
        c.setFont('Times-Roman',10); c.drawString(8,35,'red head'); c.drawString(128,35,'blue head')
        c.saveState(); c.translate(190,5); c.rotate(90); c.drawString(0,0,'rotated label'); c.restoreState()
        c.endForm()
        c.saveState(); c.translate(72,240); c.scale(1.25,1.25); c.doForm('network'); c.restoreState()
        c.beginForm('clipped-raster',0,0,20,20)
        data=BytesIO(); Image.new('RGB',(50,20),'red').save(data,format='PNG')
        c.drawImage(ImageReader(BytesIO(data.getvalue())),0,0,width=40,height=20)
        c.endForm()
        c.saveState(); c.translate(350,600); c.doForm('clipped-raster'); c.restoreState()
        for x, color in [(72,'red'),(220,'blue')]:
            data=BytesIO(); Image.new('RGB',(40,20),color).save(data,format='PNG')
            c.drawImage(ImageReader(BytesIO(data.getvalue())),x,110,width=100,height=60)
        c.setFont('Times-Roman',12); c.drawString(72,90,'Figure 1: Two source panels share this caption.')
        c.save()
        env = dict(os.environ, OPENROUTER_API_KEY='fixture-placeholder', OPENROUTER_BASE_URL=model.url,
                   DISTILLER_PDF_CACHE=str(root/'cache'))
        model.mode = 'technical'
        result = subprocess.run([binary,str(source),'--mode','epub','--cover',str(cover),'-o',str(book)],
                                env=env,capture_output=True,text=True,timeout=180)
        assert result.returncode == 0, result.stderr
        _, documents, _, files = inspect(book)
        text=' '.join(''.join(d.itertext()) for d in documents.values())
        assert 'Ordinary prose remains selectable before the equation.' in text
        assert 'Ordinary prose remains selectable after the equation.' in text
        assert any(p.find('h:img',NS) is not None and 'before the equation' in ''.join(p.itertext())
                   and 'after the equation' in ''.join(p.itertext())
                   for d in documents.values() for p in d.findall('.//h:p',NS)), 'Native passage left its paragraph'
        assert 'O(n d' not in text, 'Flattened mathematical table leaked into reading text'
        assert 'E = x' not in text, 'Ordinary-font equation lost native protection'
        assert 'Visible mixed label' in text and 'XX' not in text
        assert 'Clipped Form title' not in text and 'Disjoint hidden' not in text
        assert 'Zero area hidden' not in text
        assert text.count('Figure 1: Two source panels share this caption.') == 1
        assert any(len(f.findall('h:img',NS))==2 and f.find('h:figcaption',NS) is not None
                   for d in documents.values() for f in d.findall('.//h:figure',NS)), 'Shared panel caption split'
        cache = next((root/'cache').glob('*/receipt.json')).parent
        layout = json.loads((cache/'source.json').read_text())
        assert any(s.get('clipped') and 'Clipped Form title' in s['text'] for s in layout['spans'])
        assert any(s.get('clipped') and s['text']=='XX' for s in layout['spans'])
        assert all(not s.get('clipped') for s in layout['spans'] if 'Visible mixed label' in s['text'])
        assert any(s.get('math') for s in layout['spans'])
        graph = next(g for g in layout['graphics'] if 200 < g['bounds'][1] < 300)
        # Native path bounds include stroke extents; the transformed center and interior are exact.
        assert abs((graph['bounds'][0]+graph['bounds'][2])/2-197)<0.01, graph
        assert graph['bounds'][0]<=72 and graph['bounds'][2]>=322 and graph['bounds'][2]-graph['bounds'][0]<=258, graph
        rendered = json.loads((cache/'region-receipt.json').read_text())
        assert len(rendered['regions']) == 4
        assert any(i.get('clip') for i in layout['images']), 'Partial raster clipping was not protected'
        # Full native page is an independent source rectangle, not model-selected geometry.
        oracle=root/'oracle'; oracle.mkdir()
        manifest=oracle/'manifest.json'
        manifest.write_text(json.dumps(dict(source_sha256=sha256(source.read_bytes()).hexdigest(),
            regions=[dict(page=1,bounds=[0,0,612,792],file='source-region-0.png')])))
        result=subprocess.run([binary,str(source),'--mode','epub','--pdf-worker','--pdf-regions',str(manifest),
                               '-o',str(oracle/'receipt.json')],capture_output=True,text=True,timeout=60)
        assert result.returncode == 0, result.stderr
        full=Image.open(oracle/'source-region-0.png').convert('RGBA')
        valid_manifest=json.loads(manifest.read_text())
        for invalid in [dict(valid_manifest,source_sha256='0'*64),
                        dict(valid_manifest,regions=[dict(page=1,bounds=[-1,0,612,792],file='source-region-1.png')]),
                        dict(valid_manifest,regions=valid_manifest['regions']*2)]:
            manifest.write_text(json.dumps(invalid))
            rejected=oracle/'rejected.json'
            result=subprocess.run([binary,str(source),'--mode','epub','--pdf-worker','--pdf-regions',str(manifest),
                                   '-o',str(rejected)],capture_output=True,text=True,timeout=60)
            assert result.returncode!=0 and not rejected.exists(),result.stderr
        published={sha256(data).hexdigest():data for name,data in files.items() if '/media/' in name}
        for region in rendered['regions']:
            data=published[region['png_sha256']]
            actual=Image.open(BytesIO(data)).convert('RGBA')
            expected=full.crop(region['pixel_rect'])
            assert actual.size==expected.size and actual.tobytes()==expected.tobytes(), region
        diagram=next(r for r in rendered['regions'] if 200 < r['bounds'][1] < 300)
        pixels=Image.open(BytesIO(published[diagram['png_sha256']])).convert('RGB').tobytes()
        colors=Counter(tuple(pixels[i:i+3]) for i in range(0,len(pixels),3))
        assert sum(n for (r,g,b),n in colors.items() if r>180 and g<80 and b<80)>100
        assert sum(n for (r,g,b),n in colors.items() if b>180 and r<80 and g<80)>100
        equation=next(r for r in rendered['regions'] if r['bounds'][1]>600)
        assert equation['bounds'][0]<=72 and equation['bounds'][2]>=240 and equation['bounds'][1]<625 and equation['bounds'][3]>663, 'Equation, fraction or exponent clipped'
        for mode in ['technical_clip','technical_missing_graphic','technical_clipped_body']:
            model.mode=mode; model.calls=model.reviews=0
            failed=root/(mode+'.epub'); env['DISTILLER_PDF_CACHE']=str(root/mode)
            result=subprocess.run([binary,str(source),'--mode','epub','--cover',str(cover),'-o',str(failed)],
                                  env=env,capture_output=True,text=True,timeout=180)
            assert result.returncode!=0 and not failed.exists() and model.calls==3, (mode,result.stderr)
            expected={'technical_clip':'atomic native passage','technical_missing_graphic':'Missing source',
                      'technical_clipped_body':'excluded by native clipping'}[mode]
            assert expected in result.stderr,result.stderr
        rejected=root/'clipped-raster-rejection'; rejected.mkdir()
        c=canvas.Canvas(str(rejected/'source.pdf'),pagesize=(612,792))
        c.drawString(72,700,'Visible body text')
        c.beginForm('excluded-image',0,0,20,20)
        c.drawImage(ImageReader(BytesIO(PNG)),30,0,width=20,height=20); c.endForm()
        c.doForm('excluded-image'); c.save()
        result=subprocess.run([binary,str(rejected/'source.pdf'),'--mode','epub','--pdf-worker',
                               '-o',str(rejected/'source.json')],capture_output=True,text=True,timeout=60)
        assert result.returncode!=0 and 'Wholly clipped PDF rasters' in result.stderr and not (rejected/'source.json').exists(),result.stderr
        if args.epubcheck:
            subprocess.run(['java','-jar',str(args.epubcheck.resolve()),str(book)],check=True,timeout=60)
        print('Technical PDF journey passed: equation, math table, source colors, rotated labels, Form coordinates, shared caption, native clipping and crop rejection.')


if __name__=='__main__':
    main()
