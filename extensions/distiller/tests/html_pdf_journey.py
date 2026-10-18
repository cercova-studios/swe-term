"""HTML-first EPUB preserves structured content and recovers PDF-only evidence."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
import base64
import sys

from reportlab.pdfgen import canvas
from PIL import Image
from io import BytesIO
from epub_journey import inspect, NS, PNG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--epubcheck', type=Path)
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    with tempfile.TemporaryDirectory(prefix='distiller-html-pdf-') as directory:
        root = Path(directory)
        pdf, html, cover = root/'source.pdf', root/'source.html', root/'cover.png'
        cover.write_bytes(PNG)
        lines = ['The first source paragraph retains its complete original wording.',
                 'The second physical line supplies an independent source witness.',
                 'The third line includes training data that prose pruning must retain.',
                 'The fourth source line completes the original paragraph correctly.']
        vector = 'Vector z belongs to the model and remains visible in the source.'
        missing = 'This source-only publication notice must survive in the EPUB.'
        caption = 'Figure 1: Colored boxes and a connecting arrow remain visible.'
        def draw_pdf(extra=(), missing_y=600, caption_text=caption, rotated_page=False, bottom_notice=False):
            c = canvas.Canvas(str(pdf), pagesize=(612,792))
            c.setFont('Times-Bold',20); c.drawString(72,750,'Structured source')
            c.setFont('Times-Roman',12)
            for i, line in enumerate(lines): c.drawString(72,700-i*16,line)
            c.drawString(72,636,vector)
            c.drawString(72,missing_y,missing)
            c.setStrokeColorRGB(1,0,0); c.rect(72,350,100,80)
            c.setStrokeColorRGB(0,0,1); c.rect(220,350,100,80)
            c.line(172,390,220,390)
            c.setStrokeColorRGB(0,0,0); c.drawString(72,320,caption_text)
            for x,y,text in extra:c.drawString(x,y,text)
            if bottom_notice:
                c.setFont('Times-Roman',10.8)
                c.drawString(72,50,'A small publication notice belongs in a retained source note.')
            if rotated_page:
                c.showPage();c.setFont('Times-Roman',12)
                c.saveState();c.translate(500,70);c.rotate(90)
                c.drawString(0,0,'A rotated-only page has no ordinary font baseline for recovery.')
                c.restoreState()
            c.save()
        draw_pdf()
        body = f'''<html><head><title>Structured source</title></head><body><article>
        <p>Publication permission remains part of the original article.</p>
        <h1>Structured source</h1><h6 class="ltx_title_abstract">Abstract</h6><p>{' '.join(lines)}</p>
        <section><h2>Methods</h2><section><h3>Encoder stack</h3><h5 class="ltx_title_paragraph" id="auxiliary">Encoder details</h5><p>Auxiliary heading keeps its structural ancestry.</p></section></section>
        <div><span class="ltx_personname">Test Writer<span class="ltx_note"><sup class="ltx_note_mark">1</sup><span class="ltx_note_content"><sup class="ltx_note_mark">1</sup><span class="ltx_note_type">footnotemark:</span>1</span></span></span></div>
        <p>Repeated marker<span class="ltx_note"><sup class="ltx_note_mark">1</sup><span class="ltx_note_content"><sup class="ltx_note_mark">1</sup><span class="ltx_note_type">footnotemark:</span><span class="ltx_tag_note">1</span></span></span> remains a marker.</p>
        <p>A sentence before the note<span class="ltx_note" id="short-note"><sup class="ltx_note_mark">2</sup><span class="ltx_note_content"><sup class="ltx_note_mark">2</sup><span class="ltx_tag ltx_tag_note">2</span><span class="ltx_note_type">thanks:</span>OK.</span></span> continues after its marker.</p>
        <p>Vector <math><semantics><mi>𝐳</mi><annotation encoding="application/x-tex">z</annotation></semantics></math> belongs to the model and remains visible in the source.</p>
        <figure id="figure-one"><object type="image/svg+xml" data="not-fetched.svg"></object>
        <figcaption>{caption}</figcaption></figure><p><a href="#figure-one">See figure</a></p>
        </article></body></html>'''
        html.write_text(body)
        env = {k:v for k,v in os.environ.items() if k not in ('OPENROUTER_API_KEY','MODEL_API_KEY')}
        def run(name, source=body):
            html.write_text(source)
            book = root/f'{name}.epub'
            result = subprocess.run([binary,str(html),'--pdf-source',str(pdf),'--mode','epub',
                '--cover',str(cover),'-o',str(book)],env=env,capture_output=True,text=True,timeout=180)
            return result, book
        result,book = run('complete')
        assert result.returncode == 0,result.stderr
        _,docs,opf,files = inspect(book)
        assert opf.find('o:metadata/dc:creator',NS).text=='Test Writer'
        titles=[(name,''.join(h.itertext())) for name,d in docs.items() if '/text/' in name for h in d.findall('.//h:h1',NS)]
        assert sum(t=='Structured source' for _,t in titles)==2,('Spurious title chapter',titles)
        assert any(''.join(h.itertext())=='Abstract' for d in docs.values() for h in d.findall('.//h:h2',NS))
        assert any(''.join(h.itertext())=='Encoder details' for d in docs.values() for h in d.findall('.//h:h4',NS)), 'Auxiliary heading skipped its parent level'
        text = ' '.join(''.join(d.itertext()) for d in docs.values())
        assert missing in text and all(line in text for line in lines)
        assert text.count(caption) == 1,'Caption repeated during PDF recovery'
        assert 'footnotemark:' not in text and 'thanks:' not in text
        assert not any('ltx_tag_note' in n.get('class','').split() for d in docs.values() for n in d.iter()), 'Source note label duplicates Pandoc numbering'
        assert sum(n.get('role')=='doc-footnote' for d in docs.values() for n in d.iter()) == 1, 'Repeated markers became empty footnotes'
        assert any(n.get('role')=='doc-footnote' and ''.join(n.itertext()).strip() == 'OK.' for d in docs.values() for n in d.iter()),'Short genuine note lost content or retained a duplicate label'
        assert any('A sentence before the note' in ''.join(p.itertext()) and 'continues after its marker.' in ''.join(p.itertext()) and 'OK.' not in ''.join(p.itertext()) for d in docs.values() for p in d.findall('.//h:p',NS))
        figures = [f for d in docs.values() for f in d.findall('.//h:figure',NS)]
        assert len(figures)==1 and len(figures[0].findall('h:img',NS))==1
        receipt = json.loads(book.with_suffix('.completeness.json').read_text())
        assert receipt['pdf_sha256']==sha256(pdf.read_bytes()).hexdigest()
        assert receipt['html_sha256']==sha256(body.encode()).hexdigest()
        assert receipt['epub_sha256']==sha256(book.read_bytes()).hexdigest()
        assert len(receipt['recovered_text_spans'])==1
        assert len(receipt['recovered_figures'])==1
        crop=receipt['recovered_figures'][0]
        published={sha256(data).hexdigest():data for name,data in files.items() if '/media/' in name}
        recovered=Image.open(BytesIO(published[crop['png_sha256']])).convert('RGBA')
        assert sha256(recovered.tobytes()).hexdigest()==crop['rgba_sha256']
        assert len(receipt['matched_prose_spans'])>=4
        partial=body.replace('<object',f'<img src="data:image/png;base64,{base64.b64encode(PNG).decode()}" alt="partial panel"><object')
        result,partial_book=run('partial-panel',partial)
        assert result.returncode==0,result.stderr
        _,partial_docs,_,_=inspect(partial_book)
        assert sum(len(f.findall('h:img',NS)) for d in partial_docs.values() for f in d.findall('.//h:figure',NS))==1
        before=book.read_bytes()
        result,_=run('complete')
        assert result.returncode!=0 and book.read_bytes()==before,'Overwrite protection failed'
        tools=root/'bin';tools.mkdir()
        encoder=tools/'epub-to-kfx'
        encoder.write_text(f'''#!{sys.executable}
import pathlib,sys
if '--check' not in sys.argv:
    pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())
''')
        encoder.chmod(0o755)
        kfx=root/'paper.kfx';html.write_text(body)
        kfx_env=dict(env,PATH=str(tools)+os.pathsep+env['PATH'])
        result=subprocess.run([binary,str(html),'--pdf-source',str(pdf),'--mode','kfx',
            '--cover',str(cover),'-o',str(kfx)],env=kfx_env,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stderr
        kfx_record=json.loads(kfx.with_suffix('.completeness.json').read_text())
        assert kfx_record['kfx_sha256']==sha256(kfx.read_bytes()).hexdigest()
        assert kfx_record['pdf_sha256']==receipt['pdf_sha256']
        for name, changed, error in [
            ('wrong-title',body.replace('Structured source','Another source'),'title does not match'),
            ('missing-prose',body.replace(lines[1],'').replace(lines[2],''),'Substantial missing'),
            ('missing-caption',body.replace(caption,'A picture'),'numbered caption'),
            ('interior-line',body.replace(lines[1],''),'inside an HTML paragraph'),
            ('orphan-artwork',body.replace('</article>','<object data="orphan.svg"></object></article>'),'no figure/caption'),
            ('ambiguous-caption',body.replace('</article>',f'<figure><figcaption>{caption}</figcaption></figure></article>'),'no unique HTML')]:
            result,failed=run(name,changed)
            assert result.returncode!=0 and error in result.stderr,(name,result.stderr)
            assert not failed.exists() and not failed.with_suffix('.completeness.json').exists()
        right='An unrelated right column paragraph has its own reading order.'
        draw_pdf(extra=[(330,620,right)])
        result,cross=run('columns',body.replace('</article>',f'<p>{right}</p></article>'))
        assert result.returncode==0,result.stderr
        _,cross_docs,_,_=inspect(cross)
        cross_text=' '.join(''.join(d.itertext()) for d in cross_docs.values())
        assert cross_text.index(missing)<cross_text.index(right),'Recovery crossed PDF columns'
        for name, extra, missing_y, caption_text, changed, error in [
            ('shared-boundary',[(72,570,'A second isolated missing source line must not reverse its order.')],610,caption,body,'share an insertion'),
            ('native-prose',[(80,390,'Native prose inside a figure must remain accounted for.')],600,caption,
             body.replace('<object type="image/svg+xml" data="not-fetched.svg"></object>',f'<img src="data:image/png;base64,{base64.b64encode(PNG).decode()}" alt="placeholder">'),'Missing native PDF text'),
            ('table-artwork',[],600,caption.replace('Figure 1:','Table 1:'),
             body.replace('Figure 1:','Table 1:').replace('<figcaption>','<table><tr><td>data</td></tr></table><figcaption>'),'Unsupported artwork inside table')]:
            draw_pdf(extra=extra,missing_y=missing_y,caption_text=caption_text)
            result,failed=run(name,changed)
            assert result.returncode!=0 and error in result.stderr,(name,result.stderr)
            assert not failed.exists()
        draw_pdf(rotated_page=True)
        result,failed=run('rotated-baseline')
        assert result.returncode!=0 and 'Unresolved rotated PDF text' in result.stderr,result.stderr
        assert 'panicked' not in result.stderr and not failed.exists()
        draw_pdf(bottom_notice=True)
        result,notice=run('bottom-notice')
        assert result.returncode==0,result.stderr
        _,notice_docs,_,_=inspect(notice)
        notice_paragraphs=[''.join(p.itertext()) for d in notice_docs.values() for p in d.findall('.//h:p',NS)]
        assert 'A small publication notice belongs in a retained source note.' in notice_paragraphs
        assert notice_paragraphs.index('A small publication notice belongs in a retained source note.')>notice_paragraphs.index(missing)
        assert any(n.get('role')=='note' and 'A small publication notice' in ''.join(n.itertext()) for d in notice_docs.values() for n in d.iter())
        if args.epubcheck:
            subprocess.run(['java','-jar',str(args.epubcheck),str(book)],check=True,timeout=60)
        print('HTML/PDF journey passed: source notice, vector recovery, caption/link fidelity, identity, omission rejection and no-clobber publication.')


if __name__=='__main__':
    main()
