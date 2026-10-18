"""Installed PDF journey: bounded dense input, source compounds and semantic headers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from reportlab.pdfgen import canvas
from epub_journey import inspect, NS, PNG
from pdf_model_stub import ModelStub


def convert(binary, source, output, root, model, offline=False):
    env=dict(os.environ,OPENROUTER_BASE_URL=model.url,OPENROUTER_API_KEY='fixture-placeholder',
             DISTILLER_PDF_CACHE=str(root/'cache'))
    if offline: env.pop('OPENROUTER_API_KEY')
    return subprocess.run([str(binary),str(source),'--mode','epub','--cover',str(root/'cover.png'),'-o',str(output)],
                          env=env,capture_output=True,text=True,timeout=240)


def dense(binary, root, model):
    root.mkdir();(root/'cover.png').write_bytes(PNG);source=root/'dense.pdf'
    c=canvas.Canvas(str(source),pagesize=(612,792));c.setTitle('Dense source ownership')
    expected=[]
    for page in range(2):
        c.setFont('Times-Roman',4)
        for row in range(110):
            for column in range(7):
                text=f'p{page}r{row:03}c{column} evidence';expected.append(text)
                c.drawString(30+column*80,750-row*6,text)
        c.showPage()
    c.save();model.mode='dense';model.page_ranges.clear()
    output=root/'dense.epub';result=convert(binary,source,output,root,model)
    assert result.returncode==0,result.stderr
    assert model.page_ranges==[(1,1),(1,1),(2,2),(2,2)],model.page_ranges
    _,docs,_,_=inspect(output)
    text=' '.join(''.join(d.find('h:body',NS).itertext()) for k,d in docs.items() if '/text/ch' in k)
    assert all(text.count(t)==1 for t in expected),'Owned dense-source text lost or duplicated'
    calls=model.calls;result=convert(binary,source,root/'offline.epub',root,model,offline=True)
    assert result.returncode==0 and model.calls==calls,result.stderr


def repairs(binary, root, model):
    root.mkdir();(root/'cover.png').write_bytes(PNG);source=root/'repairs.pdf'
    c=canvas.Canvas(str(source),pagesize=(612,1000));c.setTitle('General reconstruction repairs')
    c.setFont('Times-Roman',12)
    lines=['Source: policy-guided, fault-tolerant and days-equivalent.',
           'Both re-cover and recover are printed.', 'Earlier re-entry and ported reports.',
           'Measured months of work.', 'policy-', 'guided behavior.', 'fault-', 'tolerant behavior.',
           'months-', 'equivalent work.', 're-', 'ported failures.', 'Ev-', 'ery ordinary word survives.',
           're-', 'cover the package.']
    for i,line in enumerate(lines):c.drawString(72,960-i*14,line)
    for y in [700,660,420]:c.line(72,y,540,y)
    c.setFont('Times-Bold',10)
    for x,text in [(80,'System'),(240,'Error-'),(410,'Latency')]:c.drawString(x,685,text)
    c.drawString(240,671,'tolerant')
    c.setFont('Times-Roman',10)
    for y,values in [(645,('Node A','Low','1 ms')),(625,('Node B','High','2 ms'))]:
        for x,text in zip([80,240,410],values):c.drawString(x,y,text)
    c.save();model.mode='repair';result=convert(binary,source,root/'repairs.epub',root,model)
    assert result.returncode==0,result.stderr
    _,docs,_,_=inspect(root/'repairs.epub')
    text=' '.join(''.join(d.find('h:body',NS).itertext()) for k,d in docs.items() if '/text/ch' in k)
    for expected in ['policy-guided behavior.','fault-tolerant behavior.','months-equivalent work.',
                     'reported failures.','Every ordinary word survives.','recover the package.']:
        assert expected in text,expected
    heads=[[''.join(th.itertext()) for th in head.findall('.//h:th',NS)] for d in docs.values() for head in d.findall('.//h:thead',NS)]
    assert ['System','Error-tolerant','Latency'] in heads,heads
    assert all(th.get('scope')=='col' for d in docs.values() for th in d.findall('.//h:thead/h:tr/h:th',NS))
    assert model.join_calls==1,model.join_calls
    cache=next((root/'cache').glob('*/receipt.json')).parent
    plans={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.glob('pages-*.json')}
    before=model.join_calls;result=convert(binary,source,root/'offline.epub',root,model,offline=True)
    assert result.returncode==0 and model.join_calls==before,result.stderr
    assert all(hashlib.sha256((cache/name).read_bytes()).hexdigest()==value for name,value in plans.items())
    for p in cache.glob('joins-*.json'):
        if not p.name.endswith(('-envelope.json','-input.json')):p.unlink()
    before=model.join_calls;result=convert(binary,source,root/'missing-key.epub',root,model,offline=True)
    assert result.returncode!=0 and 'Unresolved compound pairs' in result.stderr and model.join_calls==before and not (root/'missing-key.epub').exists(),result.stderr
    model.mode='repair_invalid_joins';result=convert(binary,source,root/'invalid.epub',root,model)
    assert result.returncode!=0 and 'changed source IDs' in result.stderr and not (root/'invalid.epub').exists(),result.stderr


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binary',type=Path,required=True);args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='distiller-repair-') as directory,ModelStub() as model:
        root=Path(directory);dense(args.binary.resolve(),root/'dense',model);repairs(args.binary.resolve(),root/'repairs',model)
    print('PDF repair journey passed: dense ownership, adaptive replay, compounds, header cells and rejected join IDs.')


if __name__=='__main__':main()
