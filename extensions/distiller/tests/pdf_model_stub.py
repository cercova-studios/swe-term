"""Fixture-only OpenRouter server: explicit layout oracle, never a production fallback."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading


def technical_blocks(data, mode):
    """Explicit page-layout oracle for pdf_fidelity_journey.py, not an extractor."""
    blocks, used = [], set()
    for left, bottom, right, top in [(65,615,500,680), (65,440,510,575), (65,230,350,360)]:
        selected = [s for s in data['spans'] if s.get('native_region') and
                    bottom <= sum(s['native_region']['bounds'][1::2])/2 <= top and
                    left <= sum(s['native_region']['bounds'][::2])/2 <= right]
        assert len(selected)==1,selected
        used.update(s['id'] for s in selected)
        blocks.append(dict(kind='visual', parts=[dict(id=s['id'],join='none') for s in selected]))
    before=next(s for s in data['spans'] if s['text'].startswith('Ordinary prose remains selectable before'))
    after=next(s for s in data['spans'] if s['text'].startswith('Ordinary prose remains selectable after'))
    used.update([before['id'],after['id']])
    blocks[0]=dict(kind='p',parts=[dict(id=before['id'],join='space'),*blocks[0]['parts'],dict(id=after['id'],join='none')])
    caption = [s for s in data['spans'] if s['text'].startswith('Figure 1:')]
    used.update(s['id'] for s in caption)
    panels = [i['id'] for i in data['images']]
    for span in data['spans']:
        if span['id'] not in used:
            blocks.append(dict(kind='furniture' if span.get('clipped') else 'h' if span['text']=='Chapter One' else 'p',
                               level=1 if span['text']=='Chapter One' else 0,
                               parts=[dict(id=span['id'],join='none')]))
    blocks.append(dict(kind='figure',images=panels,parts=[dict(id=s['id'],join='none') for s in caption]))
    blocks.sort(key=lambda b: -max((next(s['y'] for s in data['spans'] if s['id']==p['id'])
                                   for p in b.get('parts',[])),default=100))
    if mode == 'technical_clip':
        anchor=next(s['id'] for s in data['spans'] if s.get('native_region'))
        assert anchor+1 not in {s['id'] for s in data['spans']}
        blocks.append(dict(kind='p',parts=[dict(id=anchor+1,join='none')]))
    if mode == 'technical_missing_graphic':
        blocks.remove(next(b for b in blocks if b['kind']=='visual'))
    if mode == 'technical_clipped_body':
        next(b for b in blocks if b['kind']=='furniture')['kind']='p'
    return blocks


class ModelStub:
    def __init__(self):
        self.mode = 'valid'
        self.expect_resume = None
        self.calls = 0
        self.reviews = 0
        self.review_candidates = {}
        self.page_ranges = []
        self.join_calls = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert request['model'] == 'deepseek/deepseek-v4.1-flash'
                if request['response_format']['json_schema']['name'] == 'pdf_joins':
                    owner.join_calls += 1
                    pairs = json.loads(request['messages'][1]['content'])['pairs']
                    joins = [dict(left=p['left'], right=p['right'],
                                  join='dehyphen' if p['joined']=='reported' else 'none') for p in pairs]
                    if owner.mode == 'repair_invalid_joins':
                        joins[0]['right'] = -1
                    body=json.dumps({'model':request['model'],'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'joins':joins})}}]}).encode()
                    self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
                    return
                assert request['provider']['require_parameters'] is True
                assert request['provider']['sort'] == 'throughput'
                assert request['response_format']['json_schema']['strict'] is True
                assert all(m['role'] != 'assistant' for m in request['messages'])
                assert 'BROKEN_RESPONSE_SENTINEL' not in json.dumps(request['messages'])
                owner.calls += 1
                if owner.expect_resume is not None and owner.calls==1:
                    assert any(isinstance(m.get('content'),str) and m['content'].startswith('Candidate reconstruction (source-ID data') for m in request['messages']) == owner.expect_resume
                    assert any(isinstance(m.get('content'),str) and m['content'].startswith('Previous response rejected:') for m in request['messages']) == owner.expect_resume
                review = any(m.get('role') == 'user' and isinstance(m.get('content'), str)
                             and 'SOURCE_REVIEW' in m['content'] for m in request['messages'])
                recovery_none = any(isinstance(m.get('content'),str) and m['content'].startswith(
                    ('Previous response rejected:', 'Incomplete model response:')) for m in request['messages'])
                assert request['reasoning'] == {'effort': 'none' if recovery_none else 'low'}
                validation_retry = any(m.get('content', '').startswith('Previous response rejected:')
                                       for m in request['messages'] if isinstance(m.get('content'), str))
                if owner.mode == 'draft_continuation_once' and owner.calls == 2:
                    feedback = next(m['content'] for m in request['messages']
                                    if isinstance(m.get('content'), str) and m['content'].startswith('Previous response rejected:'))
                    assert 'only the same prose role' in feedback and '(li)' in feedback and '(p)' in feedback
                if not review and validation_retry and owner.mode in ('missing', 'missing_once', 'draft_length_then_missing', 'draft_prose_then_missing') and not (owner.mode == 'draft_prose_then_missing' and owner.calls == 2):
                    assert any('Candidate reconstruction (source-ID data' in m.get('content', '')
                               for m in request['messages'] if isinstance(m.get('content'), str))
                if not review and any('Incomplete model response:' in m.get('content', '')
                                      for m in request['messages'] if isinstance(m.get('content'), str)):
                    assert not any('Candidate reconstruction (source-ID data' in m.get('content', '')
                                   for m in request['messages'] if isinstance(m.get('content'), str))
                if review:
                    owner.reviews += 1
                    assert any('Candidate reconstruction (source-ID data' in m.get('content', '')
                               for m in request['messages'] if isinstance(m.get('content'), str))
                content=request['messages'][1]['content']
                assert any(part['type']=='image_url' for part in content)
                data = json.JSONDecoder().raw_decode(content[0]['text'])[0]
                first, last = data['owned_pages']
                owner.page_ranges.append((first,last))
                spans = [s for s in data['spans'] if first <= s['page'] <= last]
                texts = {s['id']: s['text'] for s in spans}
                if owner.mode == 'dense' or owner.mode.startswith('repair'):
                    assert len(content[0]['text'].encode()) <= 256*1024
                    blocks=[]; used=set()
                    if owner.mode.startswith('repair'):
                        for word,tail in [('policy-','guided behavior.'),('fault-','tolerant behavior.'),('months-','equivalent work.'),('re-','ported failures.'),('Ev-','ery ordinary word survives.'),('re-','cover the package.')]:
                            a=next(s for s in spans if s['text']==word and s['id'] not in used)
                            b=next(s for s in spans if s['text']==tail)
                            used.update([a['id'],b['id']])
                            if word=='months-':
                                blocks.extend([dict(kind='p',parts=[dict(id=a['id'],join='dehyphen')]),dict(kind='p',continues=True,parts=[dict(id=b['id'],join='none')])])
                            else:
                                blocks.append(dict(kind='p',parts=[dict(id=a['id'],join='dehyphen'),dict(id=b['id'],join='none')]))
                        header=next(s for s in spans if s.get('native_region') and 'System' in s['native_region']['source_text'])
                        used.add(header['id']); blocks.append(dict(kind='visual',parts=[dict(id=header['id'],join='none')]))
                        rows=[]
                        for values in [('Node A','Low','1 ms'),('Node B','High','2 ms')]:
                            row=[]
                            for value in values:
                                span=next(s for s in spans if s['text']==value);used.add(span['id']);row.append(dict(parts=[dict(id=span['id'],join='none')]))
                            rows.append(row)
                        blocks.append(dict(kind='table',rows=rows))
                    for s in spans:
                        if s['id'] not in used:
                            blocks.append(dict(kind='visual' if s.get('native_region') else 'p',parts=[dict(id=s['id'],join='none')]))
                    body=json.dumps({'model':request['model'],'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'blocks':blocks})}}]}).encode()
                    self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
                    return
                if review:
                    candidate = next(m['content'] for m in request['messages']
                                     if isinstance(m.get('content'), str) and 'Candidate reconstruction (source-ID data' in m['content'])
                    key = (owner.mode, first, last, tuple(texts.items()))
                    assert candidate == owner.review_candidates.setdefault(key, candidate)
                assert len(data['heading_history']) <= 32
                if owner.mode.startswith('technical') or owner.mode in ('prose_geometry','prose_continuation'):
                    if owner.mode in ('prose_geometry','prose_continuation'):
                        parts=[dict(id=s['id'],join='space') for s in spans]
                        blocks=([dict(kind='h',level=1,parts=parts[:1]),dict(kind='p',parts=parts[1:5]),dict(kind='p',continues=True,parts=parts[5:])]
                            if owner.mode=='prose_continuation' else [dict(kind='h',level=1,parts=parts[:1]),dict(kind='p',parts=parts[1:3]),dict(kind='p',parts=parts[3:])])
                    else:
                        blocks=technical_blocks(data, owner.mode)
                    output = json.dumps({'blocks':blocks})
                    body = json.dumps({'model':request['model'],'choices':[{'finish_reason':'stop','message':{'content':output}}]}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(body)))
                    self.end_headers(); self.wfile.write(body)
                    return
                if first > 1:
                    assert any(h['text'] == 'Chapter One' and h['level'] == 1
                               for h in data['heading_history'])
                    assert any(bookmark[2] == 'Chapter One' for bookmark in data['bookmarks'])
                used = set()
                blocks = []

                def block(kind, selected, joins=None, **kwargs):
                    selected = [s for s in selected if s['id'] not in used]
                    if not selected:
                        return
                    ids = [s['id'] for s in selected]
                    used.update(ids)
                    operations = (joins or ['space']*(len(ids)-1))+['none']
                    blocks.append(dict(kind=kind, parts=[dict(id=id,join=operation) for id,operation in zip(ids,operations)], **kwargs))

                # These expected joins are fixture semantics, not a generalized parser.
                for starts, joins, role, kwargs in [
                    (['W','e have machines now','that we can talk to.'], ['none','space'],'p',{}),
                    (['A printed hyphen: GPT-','3 remains visible.'], ['none'],'p',{}),
                    (['Ev-','ery American institution retains 970 and 52.'], ['dehyphen'],'p',{}),
                    (['Intelligence Ex-','plosion'], ['dehyphen'],'h',{'level':1}),
                    (['An inline callout','1'], ['none'],'p',{}),
                    (['We waited','5','minutes.'], ['none','none'],'p',{}),
                    (['The internet—','and its contents.'], ['space'],'p',{}),
                ]:
                    selected = [next((s for s in spans if s['text']==prefix),None) for prefix in starts]
                    if all(selected):
                        block(role, selected, joins, **kwargs)
                for span in spans:
                    if span['id'] in used:
                        continue
                    role = 'h' if span['size'] >= 17 else 'note' if span['size'] < 10 else 'p'
                    kwargs = {'level':1} if role=='h' else {}
                    if span['text']=='1 A retained margin note.':
                        kwargs['reference']=next(s['id'] for s in spans if s['text']=='1')
                    block(role,[span],**kwargs)
                blocks.extend(dict(kind='figure',images=[image['id']]) for image in data['images'])
                # A schema-valid draft error must be repaired by the mandatory source review.
                if not review:
                    for candidate in blocks:
                        if candidate.get('parts') and texts[candidate['parts'][0]['id']] == 'Ev-':
                            candidate['parts'][0]['join'] = 'none'
                if owner.mode == 'draft_continuation_once' and owner.calls == 1:
                    blocks[1]['kind'] = 'li'
                    blocks[1]['continues'] = True
                if (owner.mode == 'missing' or (owner.mode in ('missing_once', 'draft_invalid_then_length') and owner.calls == 1) or (owner.mode in ('draft_length_then_missing', 'draft_prose_then_missing') and owner.calls == 2) or (review and owner.mode == 'review_length_then_missing' and owner.reviews == 2) or (review and owner.mode == 'review_missing')) and blocks:
                    blocks.pop(0)
                if review and (owner.mode == 'review_duplicate' or (owner.mode == 'review_prose_then_duplicate' and owner.reviews == 2)) and blocks:
                    blocks.append(blocks[0])
                if (owner.mode == 'invented' or (owner.mode == 'invented_once' and owner.calls == 1)) and blocks:
                    blocks[0]['text'] = 'Invented prose'
                if owner.mode == 'incomplete' or (owner.mode in ('incomplete_once', 'draft_length_then_missing') and owner.calls == 1) or (owner.mode == 'draft_invalid_then_length' and owner.calls == 2) or (review and (owner.mode == 'review_incomplete' or (owner.mode in ('review_incomplete_once', 'review_length_then_missing') and owner.reviews == 1))):
                    done = False
                else:
                    done = True
                output = json.dumps({'blocks':blocks}) if owner.mode != 'empty' else None
                if review and (owner.mode == 'review_prose' or (owner.mode in ('review_prose_once', 'review_prose_then_duplicate') and owner.reviews == 1)):
                    output = 'BROKEN_RESPONSE_SENTINEL: Let me analyze the source instead of returning JSON.'
                if owner.mode in ('draft_prose_once', 'draft_prose_then_missing') and owner.calls == 1:
                    output = 'BROKEN_RESPONSE_SENTINEL: Analysis instead of JSON.'
                if owner.mode == 'draft_malformed_once' and owner.calls == 1:
                    output = '{"blocks":null,"BROKEN_RESPONSE_SENTINEL":'
                body = json.dumps({'model':request['model'],'choices':[{'finish_reason':'stop' if done else 'length','message':{'content':output}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}'

    def __exit__(self,*args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
