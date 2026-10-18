#!/usr/bin/env python3
"""Exercise the installed/debug CLI and inspect the resulting EPUB package."""
import argparse
import base64
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_STORED

NS = {"h": "http://www.w3.org/1999/xhtml", "m": "http://www.w3.org/1998/Math/MathML", "o": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/", "c": "urn:oasis:names:tc:opendocument:xmlns:container"}
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC")


def inspect(path):
    with ZipFile(path) as book:
        first = book.infolist()[0]
        assert first.filename == "mimetype" and first.compress_type == ZIP_STORED
        assert book.read("mimetype") == b"application/epub+zip"
        container = ET.fromstring(book.read("META-INF/container.xml"))
        opf_path = container.find(".//c:rootfile", NS).get("full-path")
        opf = ET.fromstring(book.read(opf_path))
        assert opf.get("version") == "3.0"
        manifest = {item.get("id"): item for item in opf.findall("o:manifest/o:item", NS)}
        for itemref in opf.findall("o:spine/o:itemref", NS):
            assert itemref.get("idref") in manifest
        assert sum("nav" in item.get("properties", "").split() for item in manifest.values()) == 1
        documents = {name: ET.fromstring(book.read(name)) for name in book.namelist() if name.endswith(".xhtml")}
        for name, document in documents.items():
            ids = [node.get('id') for node in document.iter() if node.get('id')]
            assert len(ids) == len(set(ids)), (name, 'Duplicate document IDs')
            for link in document.findall(".//h:a", NS):
                href = link.get("href", "")
                if "#" in href and not href.startswith(("https:", "http:", "mailto:")):
                    file, anchor = href.split("#", 1)
                    target = str(Path(os.path.normpath(str(Path(name).parent / file)))) if file else name
                    assert target in documents, (name, href)
                    assert any(node.get("id") == anchor for node in documents[target].iter()), (name, href)
            for node in document.iter():
                for attribute in ('aria-describedby','aria-labelledby','headers'):
                    for anchor in node.get(attribute,'').split():
                        assert any(n.get('id')==anchor for n in document.iter()),(name,attribute,anchor)
        return book.namelist(), documents, opf, {name: book.read(name) for name in book.namelist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--epubcheck", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    assert shutil.which("pandoc"), "Pandoc is required for the EPUB journey"
    with tempfile.TemporaryDirectory(prefix="distiller-epub-test-") as temporary:
        root = Path(temporary)
        assets = root / "article"
        assets.mkdir()
        (assets / "figure.png").write_bytes(PNG)
        (assets / "figure.webp").write_bytes(base64.b64decode("UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAfQ//73v/+BiOh/AAA="))
        source = assets / "article.html"
        destination = root / "article.epub"
        code = "fn example() {\n\tlet n = 970;\n\n    n >> 52\n}\n"
        schema = "unstructured replacement " * 250
        html = f"""<!doctype html><html lang="en"><head><title>Reader test</title>
        <meta name="author" content="Test Author"><meta property="article:published_time" content="2024-12-19T00:00:00Z"><script type="application/ld+json">{{"@type":"Article","articleBody":"{schema}"}}</script></head>
        <body><article><h1>Reader test</h1><p>This introduction explains why sections and equations should remain readable.</p>
        <div id="wrapper"><div><h2 id="first">First section</h2><p><a href="#second">Continue</a> x<sup>2</sup> + y<sub>1</sub>.</p>
        <pre><code>{code.replace('>', '&gt;')}</code></pre>
        <figure id="figure-target"><img src="figure.png" alt="A pixel"><figcaption id="caption-target"><p>A retained caption</p><h3>Caption detail</h3><p>Additional explanation.</p></figcaption></figure>
        <p><a href="#figure-target">See figure</a> <a href="#table-target">See table</a>
        <a href="mailto:">author@example.test</a></p>
        <figure><img src="figure.webp" alt="A WebP pixel"><figcaption>A converted WebP figure</figcaption></figure>
        <p><math><mfrac><mi>x</mi><mn>2</mn></mfrac></math></p>
        <table id="table-target"><tr><th colspan="2">Spanning header</th></tr><tr><td rowspan="2">left</td><td>right</td></tr><tr><td>bottom</td></tr></table>
        <table class="empty-layout"><tr id="empty-row"><td id="empty-cell"> </td></tr></table>
        <p><a href="#empty-row">Empty row anchor</a><a href="#empty-cell">Empty cell anchor</a></p>
        <table id="empty-anchor"><tr><td></td></tr></table><p><a href="#empty-anchor">Empty anchor</a></p>
        <table class="image-layout"><tr><td><img src="figure.png" alt="An image in a layout table"></td></tr></table>
        <figure><table class="ltx_eqn_table"><tr><td><math><mi>x</mi></math></td><td>(1)</td></tr></table><figcaption>Equation alignment</figcaption></figure>
        <table class="ltx_tabular bold-labels"><tr><td class="ltx_border_t"><b>Model</b></td><td class="ltx_border_t"><b>Accuracy</b></td></tr><tr><td class="ltx_border_t">Alpha</td><td class="ltx_border_t">97.3</td></tr><tr><td>Beta</td><td>98.1</td></tr></table>
        <table class="ltx_tabular span-labels"><tr><td class="ltx_border_t" rowspan="2">Model</td><td class="ltx_border_t" rowspan="2">Size</td><td class="ltx_border_t">Train</td></tr><tr><td>Steps</td></tr><tr><td class="ltx_border_t">Alpha</td><td class="ltx_border_t">512</td><td class="ltx_border_t">100K</td></tr><tr><td>Beta</td><td>256</td><td>50K</td></tr></table>
        <table class="ltx_tabular numeric-matrix"><tr><td class="ltx_border_t"><b>1</b></td><td class="ltx_border_t"><b>2</b></td></tr><tr><td class="ltx_border_t">3</td><td class="ltx_border_t">4</td></tr><tr><td>5</td><td>6</td></tr></table>
        <table class="ltx_tabular crossing-span"><tr><td class="ltx_border_t" rowspan="2">Model</td><td class="ltx_border_t" rowspan="2">Size</td><td class="ltx_border_t" rowspan="2">Count</td><td class="ltx_border_t" rowspan="3">Group</td><td class="ltx_border_t">Train</td></tr><tr><td>Steps</td></tr><tr><td class="ltx_border_t">Alpha</td><td class="ltx_border_t">512</td><td class="ltx_border_t">8</td><td class="ltx_border_t">100K</td></tr><tr><td>Beta</td><td>256</td><td>4</td><td>Small</td><td>50K</td></tr></table>
        <table class="explicit-headers"><thead><tr><td>Region</td><td>Total</td></tr></thead><tbody><tr><th id="north-header" class="ltx_th_row">North</th><td>200</td></tr><tr><td>South</td><th id="mixed-header" scope="col">Subtotal</th></tr><tr><th scope="row"></th><td>300</td></tr></tbody></table>
        <figure><img src="figure.png" alt="Outer illustration"><figcaption>Outer description</figcaption><figure><img src="figure.png" alt="Inner illustration"><figcaption>Inner description</figcaption></figure></figure>
        <h2 id="second">Second section</h2><p>This second section discusses the examples and includes a footnote<a id="ref" href="#note">1</a>.</p>
        <p id="note">The footnote text. <a class="footnote-backref" href="#ref">Return</a> <a href="#wrapper">Wrapper</a></p></div></div>
        </article></body></html>"""
        source.write_text(html)
        # Default EPUB creation must fail closed without credentials or cache.
        offline = {key: value for key, value in os.environ.items() if key != 'MODEL_API_KEY'}
        offline['DISTILLER_COVER_CACHE'] = str(root/'empty-cover-cache')
        missing_cover = root/'mandatory-cover.epub'
        for inputs in ([str(source)], ['--stdin', '--base-url', 'https://example.test/blog/']):
            inline_html = html.replace('figure.png', 'data:image/png;base64,'+base64.b64encode(PNG).decode()).replace('figure.webp', 'data:image/png;base64,'+base64.b64encode(PNG).decode())
            result = subprocess.run([str(binary), *inputs, '--mode', 'epub', '-o', str(missing_cover)], input=inline_html if inputs[0] == '--stdin' else None, env=offline, capture_output=True, text=True, timeout=45)
            assert result.returncode != 0 and 'MODEL_API_KEY' in result.stderr and not missing_cover.exists(), result.stderr
        assert not list((root/'empty-cover-cache').iterdir()), 'No paid reservation without credentials'
        command = [str(binary), str(source), "--mode", "epub", "--base-url", "https://example.test/blog/", "--cover", str(assets/'figure.png'), "-o", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr
        names, documents, opf, files = inspect(destination)
        assert opf.find("o:metadata/dc:title", NS).text == "Reader test"
        assert opf.find("o:metadata/dc:creator", NS).text == "Test Author"
        assert opf.find("o:metadata/dc:language", NS).text == "en"
        assert opf.find("o:metadata/dc:source", NS).text == "https://example.test/blog/"
        assert opf.find("o:metadata/dc:date", NS).text == "2024-12-19"
        covers = [item for item in opf.findall('o:manifest/o:item', NS) if 'cover-image' in item.get('properties','').split()]
        assert len(covers) == 1 and covers[0].get('media-type') == 'image/png'
        assert files['EPUB/' + covers[0].get('href')] == PNG
        rendered = "\n".join(ET.tostring(doc, encoding="unicode") for doc in documents.values())
        assert "unstructured replacement" not in rendered
        blocks = [pre for doc in documents.values() for pre in doc.findall(".//h:pre", NS)]
        assert any("".join(pre.itertext()).rstrip("\n") == code.rstrip("\n") for pre in blocks)
        assert sum(len(doc.findall(".//m:math", NS)) for doc in documents.values()) == 2
        assert 'colspan="2"' in rendered and 'rowspan="2"' in rendered
        assert "A retained caption" in rendered
        assert 'href="mailto:"' not in rendered and 'author@example.test' in rendered
        assert any(data == PNG for name, data in files.items() if "/media/" in name)
        assert sum(len(d.findall('.//h:img',NS)) for d in documents.values()) == 5
        tables={c:t for d in documents.values() for t in d.findall('.//h:table',NS) for c in t.get('class','').split()}
        assert 'empty-layout' not in tables and 'image-layout' in tables and 'ltx_eqn_table' in tables
        assert tables['ltx_eqn_table'].get('role')=='presentation','Equation alignment is exposed as a data table'
        assert tables['ltx_eqn_table'].get('aria-describedby') is None,'Generated description overrides the presentation role'
        assert tables['explicit-headers'].get('role')!='presentation','Data-table semantics were hidden'
        for name in ('bold-labels','span-labels'):
            assert tables[name].find('h:thead',NS) is not None,(name,'Missing header band')
            assert all(n.get('scope')=='col' for n in tables[name].findall('.//h:thead/h:tr/h:th',NS))
            assert tables[name].find('.//h:tbody/h:tr/h:td',NS) is not None,'Data cells became headers'
        assert not tables['numeric-matrix'].findall('.//h:th',NS),'Numeric matrix gained invented headers'
        assert tables['crossing-span'].find('h:thead',NS) is None,'Rowspan crossing the header boundary gained invented headers'
        headers={n.get('id'):n for n in tables['explicit-headers'].iter() if n.get('id')}
        assert headers['north-header'].get('role')=='rowheader'
        assert headers['mixed-header'].get('role')=='columnheader'
        assert all(n.get('scope') is None for d in documents.values() for n in d.findall('.//h:td',NS)),'Pandoc demoted a scoped header into invalid td markup'
        for d in documents.values():
            for f in d.findall('.//h:figure',NS):
                caption=f.find('h:figcaption',NS)
                for image in f.findall('h:img',NS):
                    target=image.get('aria-describedby')
                    assert target and any(n.get('id')==target and ''.join(n.itertext()).strip() for n in caption.iter()),'Caption description lost text or ID'
                    if image.get('alt')=='Inner illustration':
                        assert ''.join(caption.itertext()).strip()=='Inner description','Nested figure inherited outer description'
        assert all(data.startswith(b'\x89PNG\r\n\x1a\n') for name,data in files.items() if '/media/' in name)
        assert all("font-family: serif" in ET.tostring(doc, encoding="unicode") for doc in documents.values())
        for doc in documents.values():
            styles='\n'.join(''.join(n.itertext()) for n in doc.findall('.//h:style',NS))
            equation_rule=styles.split('table.ltx_eqn_table {',1)[1].split('}',1)[0]
            assert 'border: 0;' in equation_rule,'Equation tables inherited Pandoc borders'
            assert 'table.ltx_eqn_table tbody { border: 0; }' in styles,'Equation row groups inherited Pandoc borders'
            assert 'table { display: table;' in styles,'Data tables cannot paginate as table rows'
        assert sum(len(doc.findall(".//h:h2", NS)) for doc in documents.values()) >= 2
        navigation = [a for doc in documents.values() for nav in doc.findall('.//h:nav', NS) if nav.get('role') == 'doc-toc' for a in nav.findall('.//h:a', NS)]
        assert [''.join(a.itertext()) for a in navigation if ''.join(a.itertext()) in ('First section', 'Second section')] == ['First section', 'Second section'], 'Nested headings missing or out of order in navigation'
        assert any(a.text == "Return" for doc in documents.values() for a in doc.findall(".//h:a", NS))
        original = destination.read_bytes()
        assert subprocess.run(command, capture_output=True).returncode != 0
        assert destination.read_bytes() == original

        source.write_text(html.replace('<div id="wrapper">', '<div id="wrapper" lang="ar" dir="rtl">'))
        rejected = root / 'language-wrapper.epub'
        result = subprocess.run([str(binary), str(source), '--mode', 'epub', '-o', str(rejected)], capture_output=True, text=True, timeout=45)
        assert result.returncode != 0 and 'language/direction' in result.stderr and not rejected.exists()
        source.write_text(html)

        source.write_text(html.replace('<table class="empty-layout">', '<table class="empty-layout"><tr><td><mglyph /></td></tr>'))
        rejected = root / 'unsupported-empty-layout.epub'
        result = subprocess.run([str(binary), str(source), '--mode', 'epub', '--cover', str(assets/'figure.png'), '-o', str(rejected)], capture_output=True, text=True, timeout=45)
        assert result.returncode != 0 and 'cannot preserve' in result.stderr and not rejected.exists(), result.stderr
        source.write_text(html)

        outside = root / "outside.png"
        outside.write_bytes(PNG)
        (assets / "escape.png").symlink_to(outside)
        (assets / "fake.png").write_text("not an image")
        corrupt = bytearray(PNG)
        corrupt[29] ^= 1  # IHDR checksum: valid signature, corrupt image.
        (assets / "corrupt.png").write_bytes(corrupt)
        os.mkfifo(assets / "pipe.png")
        for cover in ('corrupt.png', 'pipe.png', 'missing.png', 'figure.webp'):
            rejected = root / 'bad-cover.epub'
            result = subprocess.run([str(binary), str(source), '--mode', 'epub', '--cover', str(assets/cover), '-o', str(rejected)], capture_output=True, text=True, timeout=45)
            assert result.returncode != 0 and not rejected.exists(), cover
        assert subprocess.run([str(binary), str(source), '--cover', str(assets/'figure.png'), '--stdout'], capture_output=True, timeout=45).returncode != 0
        for image in ("../outside.png", str(outside), "escape.png", "fake.png", "corrupt.png", "pipe.png", "missing.png", "https://example.invalid/image.png"):
            source.write_text(html.replace("figure.png", image))
            rejected = root / "rejected.epub"
            result = subprocess.run([str(binary), str(source), "--mode", "epub", "-o", str(rejected)], capture_output=True, text=True, timeout=45)
            assert result.returncode != 0, (image, result.stderr)
            assert not rejected.exists(), image
        source.write_text(html)
        # Missing Pandoc must fail without publishing a partial book.
        failed = root / "failed.epub"
        result = subprocess.run([str(binary), str(source), "--mode", "epub", "-o", str(failed)], env={**offline, "PATH": str(root)}, capture_output=True, text=True, timeout=45)
        assert result.returncode != 0 and not failed.exists()
        if args.epubcheck:
            subprocess.run(["java", "-jar", str(args.epubcheck.resolve()), str(destination)], check=True, timeout=60)
        print("EPUB journey passed: structure, navigation, metadata, math, code, images, containment and atomic failure.")


if __name__ == "__main__":
    main()
