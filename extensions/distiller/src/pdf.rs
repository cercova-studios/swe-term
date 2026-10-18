//! Bounded deterministic PDF evidence extraction, followed by validated cloud reconstruction.
use crate::{
    pdf_reconstruct::{Graphic, Layout, Raster, Span, TextStyle},
    types::DistillerResponse,
};
use anyhow::{ensure, Context, Result};
use image::ImageEncoder;
use pdfium_render::prelude::*;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::Path,
    process::Stdio,
    time::Duration,
};
use tokio::process::Command;

struct Glyph {
    source_index: usize,
    text: char,
    x: f32,
    right: f32,
    y: f32,
    size: f32,
    hyphen: bool,
    bounds: [f32; 4],
    bold: bool,
    italic: bool,
    math: bool,
    rotated: bool,
    clipped: bool,
}

fn spans(mut glyphs: Vec<Glyph>, page: usize, first_id: usize) -> Vec<Span> {
    glyphs.sort_by(|a, b| b.y.total_cmp(&a.y).then(a.x.total_cmp(&b.x)));
    let mut lines: Vec<Vec<Glyph>> = Vec::new();
    for glyph in glyphs {
        if let Some(line) = lines
            .iter_mut()
            .rev()
            .find(|l| (l[0].y - glyph.y).abs() < l[0].size.min(glyph.size) * 0.25)
        {
            line.push(glyph);
        } else {
            lines.push(vec![glyph]);
        }
    }
    let mut result: Vec<Span> = Vec::new();
    for mut line in lines {
        line.sort_by(|a, b| a.x.total_cmp(&b.x));
        let mut right = None;
        for glyph in line {
            let split = right.is_none()
                || result.last().is_some_and(|span| {
                    (span.size - glyph.size).abs() > glyph.size * 0.25
                        || glyph.x - right.unwrap_or(glyph.x) > glyph.size * 0.6
                        || (!glyph.text.is_whitespace() && span.clipped != glyph.clipped)
                });
            if split {
                result.push(Span {
                    id: first_id + result.len(),
                    page,
                    x: glyph.x,
                    right: glyph.x,
                    y: glyph.y,
                    size: glyph.size,
                    text: String::new(),
                    hyphen: false,
                    bounds: glyph.bounds,
                    styles: Vec::new(),
                    math: false,
                    rotated: false,
                    clipped: true,
                });
            }
            let span = result.last_mut().unwrap();
            if right.is_some_and(|r| glyph.x - r > glyph.size * 0.15)
                && !span.text.is_empty()
                && !span.text.ends_with(char::is_whitespace)
                && !glyph.text.is_whitespace()
            {
                span.text.push(' ');
            }
            let start = span.text.len();
            span.text.push(glyph.text);
            if !glyph.text.is_whitespace() {
                span.hyphen = glyph.hyphen;
                span.right = span.right.max(glyph.right);
                span.bounds = union(span.bounds, glyph.bounds);
                span.math |= glyph.math;
                span.rotated |= glyph.rotated;
                span.clipped &= glyph.clipped;
                if glyph.bold || glyph.italic {
                    if let Some(previous) = span.styles.last_mut().filter(|s| {
                        span.text[s.end..start].chars().all(char::is_whitespace)
                            && s.bold == glyph.bold
                            && s.italic == glyph.italic
                    }) {
                        previous.end = span.text.len();
                    } else {
                        span.styles.push(TextStyle {
                            start,
                            end: span.text.len(),
                            bold: glyph.bold,
                            italic: glyph.italic,
                        });
                    }
                }
            }
            right = Some(glyph.right);
        }
    }
    for span in &mut result {
        let leading = span.text.len() - span.text.trim_start().len();
        span.text = span.text.trim().into();
        for style in &mut span.styles {
            style.start = style.start.saturating_sub(leading).min(span.text.len());
            style.end = style.end.saturating_sub(leading).min(span.text.len());
        }
        span.styles.retain(|s| s.start < s.end);
    }
    result.retain(|s| !s.text.is_empty());
    for (index, span) in result.iter_mut().enumerate() {
        span.id = first_id + index;
    }
    result
}

fn union(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn graphic_groups(mut boxes: Vec<[f32; 4]>) -> Vec<[f32; 4]> {
    let mut groups: Vec<[f32; 4]> = Vec::new();
    while let Some(mut bounds) = boxes.pop() {
        // Paths can meet through another component; repeat until the region is closed.
        loop {
            let previous = boxes.len();
            boxes.retain(|other| {
                let touches = bounds[0] <= other[2] + 2.
                    && bounds[2] + 2. >= other[0]
                    && bounds[1] <= other[3] + 2.
                    && bounds[3] + 2. >= other[1];
                if touches {
                    bounds = union(bounds, *other);
                }
                !touches
            });
            if previous == boxes.len() {
                break;
            }
        }
        groups.push(bounds);
    }
    groups.sort_by(|a, b| b[3].total_cmp(&a[3]).then(a[0].total_cmp(&b[0])));
    groups
}

fn visit_objects(
    object: &mut PdfPageObject<'_>,
    depth: u8,
    transform: PdfMatrix,
    inherited_clip: Option<[f32; 4]>,
    visited: &mut usize,
    visit: &mut impl FnMut(&mut PdfPageObject<'_>, PdfMatrix, Option<[f32; 4]>) -> Result<()>,
) -> Result<()> {
    *visited += 1;
    ensure!(
        depth <= 16 && *visited <= 100_000,
        "PDF page object tree exceeds limits"
    );
    // Clip bounding boxes are conservative: outside is proof of exclusion; inside is not proof of visibility.
    let local_clip = object.get_clip_path().and_then(|clip| {
        let mut bounds: Option<[f32; 4]> = None;
        for path in clip.iter() {
            for segment in path.iter() {
                if !matches!(
                    segment.segment_type(),
                    PdfPathSegmentType::MoveTo | PdfPathSegmentType::LineTo
                ) {
                    return None;
                }
                let (x, y) = transform.apply_to_points(segment.x(), segment.y());
                if !x.value.is_finite() || !y.value.is_finite() {
                    return None;
                }
                let point = [x.value, y.value, x.value, y.value];
                bounds = Some(bounds.map_or(point, |b| union(b, point)));
            }
        }
        bounds
    });
    let clip = match (inherited_clip, local_clip) {
        (Some(a), Some(b)) => Some([
            a[0].max(b[0]),
            a[1].max(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
        ]),
        (a, b) => a.or(b),
    };
    if let Some(form) = object.as_x_object_form_object() {
        let matrix = form.matrix()?;
        ensure!(
            matrix.a().is_finite()
                && matrix.d().is_finite()
                && matrix.a() > 0.
                && matrix.d() > 0.
                && matrix.b().abs() < 0.00001
                && matrix.c().abs() < 0.00001
                && (matrix.a() - matrix.d()).abs() < 0.00001,
            "Rotated, skewed or stretched PDF Form objects are unsupported"
        );
        for index in form.as_range() {
            visit_objects(
                &mut form.get(index)?,
                depth + 1,
                matrix.multiply(transform),
                clip,
                visited,
                visit,
            )?;
        }
    } else {
        visit(object, transform, clip)?;
    }
    Ok(())
}

fn engine() -> Result<Pdfium> {
    let library = std::env::var_os("HOME").map(|h| {
        Pdfium::pdfium_platform_library_name_at_path(
            &Path::new(&h).join(".local/share/distiller/lib"),
        )
    });
    let bindings=if let Some(path)=library.filter(|p|p.is_file()){Pdfium::bind_to_library(path)}else{Pdfium::bind_to_system_library()}.context("PDF export requires PDFium 7881 in ~/.local/share/distiller/lib or the system library path")?;
    Ok(Pdfium::new(bindings))
}

fn source_bytes(input: &Path) -> Result<Vec<u8>> {
    ensure!(input.is_file(), "PDF must be a regular file");
    let mut bytes = Vec::new();
    fs::File::open(input)?
        .take(200 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 200 * 1024 * 1024, "PDF exceeds 200 MiB");
    Ok(bytes)
}

/// Internal worker writes source spans/geometry and unmodified raster assets, without a model call.
pub fn extract(input: &Path, output: &Path) -> Result<()> {
    let bytes = source_bytes(input)?;
    let pdfium = engine()?;
    let document = pdfium
        .load_pdf_from_byte_slice(&bytes, None)
        .context("Load unencrypted PDF")?;
    ensure!(document.pages().len() <= 400, "PDF exceeds 400 pages");
    let metadata = document.metadata();
    let title = metadata
        .get(PdfDocumentMetadataTagType::Title)
        .map(|m| m.value().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_default();
    let author = metadata
        .get(PdfDocumentMetadataTagType::Author)
        .map(|m| m.value().to_string())
        .unwrap_or_default();
    let mut layout = Layout {
        title,
        author,
        source_sha256: format!("{:x}", Sha256::digest(&bytes)),
        pages: Vec::new(),
        spans: Vec::new(),
        images: Vec::new(),
        graphics: Vec::new(),
        regions: Vec::new(),
        outline: Vec::new(),
        warnings: Vec::new(),
    };
    for bookmark in document.bookmarks().iter() {
        if let (Some(title), Some(destination)) = (bookmark.title(), bookmark.destination()) {
            let mut depth = 1;
            let mut parent = bookmark.parent();
            while let Some(p) = parent {
                depth += 1;
                parent = p.parent();
            }
            layout
                .outline
                .push((destination.page_index()? as usize + 1, depth.min(6), title));
        }
    }
    let mut image_total = 0usize;
    let mut fonts = BTreeMap::new();
    for (index, page) in document.pages().iter().enumerate() {
        layout.pages.push([page.width().value, page.height().value]);
        let text = page.text()?;
        ensure!(
            text.chars().len() <= 100_000,
            "Page {} exceeds 100000 glyphs",
            index + 1
        );
        let mut glyphs = Vec::new();
        for character in text.chars().iter() {
            let hyphen = character.is_hyphen()?;
            let value = if hyphen {
                '-'
            } else {
                character
                    .unicode_char()
                    .context("PDF character lacks a Unicode mapping")?
            };
            if value.is_control() {
                continue;
            }
            let advance_bounds = character.loose_bounds()?;
            let bounds = character.tight_bounds()?;
            let (x, y) = character.origin()?;
            let size = character.scaled_font_size().value;
            let name = character.font_name();
            let (bold, italic, math) = *fonts.entry(name.clone()).or_insert_with(|| {
                let font = name.to_ascii_lowercase();
                let font = font.rsplit('+').next().unwrap_or(&font);
                (
                    matches!(
                        character.font_weight(),
                        Some(
                            PdfFontWeight::Weight600
                                | PdfFontWeight::Weight700Bold
                                | PdfFontWeight::Weight800
                                | PdfFontWeight::Weight900
                        )
                    ) || font.contains("bold"),
                    character.font_is_italic()
                        || font.contains("italic")
                        || font.contains("oblique"),
                    ["cmmi", "cmsy", "cmex", "msam", "msbm"]
                        .iter()
                        .any(|prefix| font.starts_with(prefix))
                        || font.contains("math")
                        || font.contains("symbol"),
                )
            });
            ensure!(
                [
                    x.value,
                    y.value,
                    size,
                    bounds.left().value,
                    bounds.right().value,
                    bounds.bottom().value,
                    bounds.top().value
                ]
                .iter()
                .all(|n| n.is_finite()),
                "Invalid PDF glyph geometry"
            );
            let rotation = character.angle_degrees()?.rem_euclid(180.);
            glyphs.push(Glyph {
                source_index: character.index(),
                text: value,
                x: x.value,
                right: advance_bounds.right().value,
                y: y.value,
                size,
                hyphen,
                bounds: [
                    bounds.left().value,
                    bounds.bottom().value,
                    bounds.right().value,
                    bounds.top().value,
                ],
                bold,
                italic,
                rotated: rotation > 1. && rotation < 179.,
                clipped: false,
                math: (math && !matches!(value, '∗' | '*' | '†' | '‡' | '•' | '·' | '–' | '—'))
                    || matches!(
                        value,
                        '=' | '√'
                            | '∑'
                            | '∏'
                            | '∫'
                            | '∞'
                            | '≤'
                            | '≥'
                            | '≠'
                            | '±'
                            | '×'
                            | '÷'
                    ),
            });
        }
        let expected: BTreeMap<char, usize> = glyphs
            .iter()
            .filter(|g| !g.text.is_whitespace())
            .fold(BTreeMap::new(), |mut m, g| {
                *m.entry(g.text).or_default() += 1;
                m
            });
        let mut images = 0;
        let mut graphic_boxes = Vec::new();
        let mut clipped_glyphs = BTreeSet::new();
        let mut clipped_ink = BTreeMap::new();
        let mut visited = 0;
        let mut visit = |object: &mut PdfPageObject<'_>,
                         transform: PdfMatrix,
                         clip: Option<[f32; 4]>|
         -> Result<()> {
            if let (Some(clip), Some(object)) = (clip, object.as_text_object()) {
                let bounds = object.bounds()?.to_rect().transform(transform);
                let contained = bounds.left().value >= clip[0]
                    && bounds.bottom().value >= clip[1]
                    && bounds.right().value <= clip[2]
                    && bounds.top().value <= clip[3];
                // chars_for_object scans the page; contained text cannot cross this clip envelope.
                if !contained {
                    for character in text.chars_for_object(object)?.iter() {
                        let ink = character.tight_bounds()?;
                        if clip[0] >= clip[2]
                            || clip[1] >= clip[3]
                            || ink.right().value <= clip[0]
                            || ink.left().value >= clip[2]
                            || ink.top().value <= clip[1]
                            || ink.bottom().value >= clip[3]
                        {
                            clipped_glyphs.insert(character.index());
                        } else {
                            clipped_ink.insert(
                                character.index(),
                                [
                                    ink.left().value.max(clip[0]),
                                    ink.bottom().value.max(clip[1]),
                                    ink.right().value.min(clip[2]),
                                    ink.top().value.min(clip[3]),
                                ],
                            );
                        }
                    }
                }
            }
            if matches!(
                object.object_type(),
                PdfPageObjectType::Path
                    | PdfPageObjectType::Shading
                    | PdfPageObjectType::Unsupported
            ) {
                ensure!(!expected.is_empty(), "Page {} has graphics without selectable text; graphic-only reflow is unsupported", index + 1);
                let bounds = object.bounds()?.to_rect().transform(transform);
                let bounds = [
                    bounds.left().value,
                    bounds.bottom().value,
                    bounds.right().value,
                    bounds.top().value,
                ];
                ensure!(
                    bounds.iter().all(|n| n.is_finite()),
                    "Invalid PDF graphic geometry"
                );
                // Stroke extents are not included in a path's geometric bounds.
                let mut painted = [
                    bounds[0] - 1.,
                    bounds[1] - 1.,
                    bounds[2] + 1.,
                    bounds[3] + 1.,
                ];
                if let Some(clip) = clip {
                    painted = [
                        painted[0].max(clip[0]),
                        painted[1].max(clip[1]),
                        painted[2].min(clip[2]),
                        painted[3].min(clip[3]),
                    ];
                }
                if painted[0] < painted[2] && painted[1] < painted[3] {
                    graphic_boxes.push(painted);
                }
            }
            if let Some(image) = object.as_image_object_mut() {
                ensure!(
                    !expected.is_empty(),
                    "Page {} has images without selectable text; OCR is unsupported",
                    index + 1
                );
                ensure!(
                    image.width()? <= 12000
                        && image.height()? <= 12000
                        && (image.width()? as u64) * (image.height()? as u64) <= 32_000_000,
                    "PDF raster image exceeds pixel limits"
                );
                let matrix = image.matrix()?;
                ensure!(
                    matrix.a() > 0.
                        && matrix.d() > 0.
                        && matrix.b().abs() < 0.00001
                        && matrix.c().abs() < 0.00001,
                    "Rotated, mirrored or skewed PDF images are unsupported"
                );
                let placement = matrix.multiply(transform);
                let original = [
                    placement.e(),
                    placement.f(),
                    placement.e() + placement.a(),
                    placement.f() + placement.d(),
                ];
                let clip = clip
                    .map(|c| {
                        [
                            c[0].max(original[0]),
                            c[1].max(original[1]),
                            c[2].min(original[2]),
                            c[3].min(original[3]),
                        ]
                    })
                    .filter(|c| *c != original);
                ensure!(
                    clip.is_none_or(|c| c[0] < c[2] && c[1] < c[3]),
                    "Wholly clipped PDF rasters are unsupported"
                );
                // Render at integer source-pixel scale to avoid placement-dependent resampling.
                image.reset_matrix(PdfMatrix::new(
                    image.width()? as f32,
                    0.,
                    0.,
                    image.height()? as f32,
                    0.,
                    0.,
                ))?;
                let decoded = image.get_processed_image(&document);
                image.reset_matrix(matrix)?;
                let decoded = decoded?;
                let raw = image.get_raw_image_data()?;
                // Preserve JPEG compression only when PDFium confirms masks/transforms do not change pixels.
                let (bytes, extension) = if raw.starts_with(b"\xff\xd8\xff")
                    && image.get_raw_image()?.to_rgba8() == decoded.to_rgba8()
                {
                    (raw, "jpg")
                } else {
                    let mut bytes = Vec::new();
                    image::codecs::png::PngEncoder::new_with_quality(
                        &mut bytes,
                        image::codecs::png::CompressionType::Best,
                        image::codecs::png::FilterType::Adaptive,
                    )
                    .write_image(
                        decoded.as_bytes(),
                        decoded.width(),
                        decoded.height(),
                        decoded.color().into(),
                    )?;
                    (bytes, "png")
                };
                image_total += bytes.len();
                ensure!(
                    bytes.len() <= 4 * 1024 * 1024 && image_total <= 64 * 1024 * 1024,
                    "Page {} raster image ({} bytes, {} total) exceeds EPUB byte limits",
                    index + 1,
                    bytes.len(),
                    image_total
                );
                images += 1;
                let name = format!("page-{}-image-{images}.{extension}", index + 1);
                fs::write(output.parent().unwrap().join(&name), bytes)?;
                layout.images.push(Raster {
                    id: layout.images.len(),
                    page: index + 1,
                    x: placement.e(),
                    y: placement.f(),
                    width: placement.a(),
                    height: placement.d(),
                    file: name,
                    clip,
                });
            }
            Ok(())
        };
        for number in page.objects().as_range() {
            visit_objects(
                &mut page.objects().get(number)?,
                0,
                PdfMatrix::identity(),
                None,
                &mut visited,
                &mut visit,
            )?;
        }
        for glyph in &mut glyphs {
            glyph.clipped = clipped_glyphs.contains(&glyph.source_index);
            if let Some(bounds) = clipped_ink.get(&glyph.source_index) {
                glyph.bounds = *bounds;
            }
        }
        let page_spans = spans(glyphs, index + 1, layout.spans.len());
        let retained: BTreeMap<char, usize> = page_spans
            .iter()
            .flat_map(|s| s.text.chars())
            .filter(|c| !c.is_whitespace())
            .fold(BTreeMap::new(), |mut m, c| {
                *m.entry(c).or_default() += 1;
                m
            });
        ensure!(
            expected == retained,
            "Page {} glyph coverage changed during span extraction",
            index + 1
        );
        layout.spans.extend(page_spans);
        for bounds in graphic_groups(graphic_boxes) {
            layout.graphics.push(Graphic {
                id: layout.graphics.len(),
                page: index + 1,
                bounds,
            });
        }
        let preview = page
            .render_with_config(
                &PdfRenderConfig::new()
                    .set_target_width(1200)
                    .set_maximum_height(1800),
            )?
            .as_image()?
            .to_luma8();
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 80)
            .encode_image(&image::DynamicImage::ImageLuma8(preview))?;
        ensure!(bytes.len() <= 1024 * 1024, "PDF preview exceeds 1 MiB");
        fs::write(
            output
                .parent()
                .unwrap()
                .join(format!("page-{}-preview.jpg", index + 1)),
            bytes,
        )?;
    }
    let regions = crate::pdf_reconstruct::prepare_regions(&mut layout)?;
    fs::write(
        output.parent().unwrap().join("regions.json"),
        serde_json::to_vec(&regions)?,
    )?;
    fs::write(output, serde_json::to_vec(&layout)?)?;
    Ok(())
}

/// Native source render; rectangles originate in deterministic source segmentation.
pub fn render_regions(input: &Path, manifest: &Path, output: &Path) -> Result<()> {
    let bytes = source_bytes(input)?;
    ensure!(
        fs::metadata(manifest)?.len() <= 2 * 1024 * 1024,
        "Source region manifest exceeds 2 MiB"
    );
    let manifest: crate::pdf_reconstruct::RegionManifest =
        serde_json::from_slice(&fs::read(manifest)?)?;
    ensure!(
        manifest.source_sha256 == format!("{:x}", Sha256::digest(&bytes)),
        "PDF changed after source extraction"
    );
    ensure!(manifest.regions.len() <= 10_000, "Too many source regions");
    let pdfium = engine()?;
    let document = pdfium.load_pdf_from_byte_slice(&bytes, None)?;
    let directory = output
        .parent()
        .context("Source regions require an output directory")?;
    let mut total = 0u64;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("page-") && name.contains("-image-") {
            total += entry.metadata()?.len();
        }
    }
    let mut records = Vec::new();
    let mut names = BTreeSet::new();
    let mut regions: Vec<_> = manifest.regions.iter().collect();
    regions.sort_by_key(|r| r.page);
    let mut rendered = None;
    let mut rendered_page = 0;
    for region in regions {
        ensure!(
            region.page > 0 && region.page <= document.pages().len() as usize,
            "Invalid source region page"
        );
        ensure!(
            region
                .file
                .strip_prefix("source-region-")
                .and_then(|s| s.strip_suffix(".png"))
                .is_some_and(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())),
            "Invalid source region file name"
        );
        ensure!(
            names.insert(&region.file),
            "Repeated source region file name"
        );
        let page = document.pages().get((region.page - 1) as i32)?;
        let width = page.width().value;
        let height = page.height().value;
        let [left, bottom, right, top] = region.bounds;
        ensure!(
            region.bounds.iter().all(|v| v.is_finite())
                && left >= 0.
                && bottom >= 0.
                && right <= width
                && top <= height
                && left < right
                && bottom < top,
            "Invalid source region rectangle"
        );
        ensure!(
            (width * 3.).ceil() <= 12000.
                && (height * 3.).ceil() <= 12000.
                && (width * 3.).ceil() * (height * 3.).ceil() <= 32_000_000.,
            "Source region page exceeds render pixel limits"
        );
        if rendered_page != region.page {
            rendered = Some(
                page.render_with_config(&PdfRenderConfig::new().scale_page_by_factor(3.))?
                    .as_image()?
                    .to_rgba8(),
            );
            rendered_page = region.page;
        }
        let image = rendered.as_ref().unwrap();
        let sx = image.width() as f32 / width;
        let sy = image.height() as f32 / height;
        let x = (left * sx).floor() as u32;
        let y = ((height - top) * sy).floor() as u32;
        let right = ((right * sx).ceil() as u32).min(image.width());
        let bottom = (((height - bottom) * sy).ceil() as u32).min(image.height());
        ensure!(right > x && bottom > y, "Empty source region pixels");
        let crop = image::imageops::crop_imm(image, x, y, right - x, bottom - y).to_image();
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new_with_quality(
            &mut bytes,
            image::codecs::png::CompressionType::Best,
            image::codecs::png::FilterType::Adaptive,
        )
        .write_image(
            crop.as_raw(),
            crop.width(),
            crop.height(),
            image::ExtendedColorType::Rgba8,
        )?;
        total += bytes.len() as u64;
        ensure!(
            bytes.len() <= 4 * 1024 * 1024 && total <= 64 * 1024 * 1024,
            "Source region exceeds EPUB image byte limits"
        );
        fs::write(directory.join(&region.file), &bytes)?;
        records.push(serde_json::json!({"file":region.file,"page":region.page,"bounds":region.bounds,"pixel_rect":[x,y,right,bottom],"width":crop.width(),"height":crop.height(),"png_sha256":format!("{:x}",Sha256::digest(&bytes)),"rgba_sha256":format!("{:x}",Sha256::digest(crop.as_raw()))}));
    }
    fs::write(
        output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"source_sha256":manifest.source_sha256,"scale":3,"color":"source RGBA","regions":records}),
        )?,
    )?;
    Ok(())
}

pub(crate) async fn extract_layout(input: &Path, work: &Path) -> Result<Layout> {
    let source = work.join("source.json");
    let errors = work.join("adapter.stderr");
    let mut child = Command::new(std::env::current_exe()?)
        .arg(input)
        .args(["--mode", "epub", "--pdf-worker", "-o"])
        .arg(&source)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(fs::File::create(&errors)?)
        .kill_on_drop(true)
        .spawn()?;
    let status = tokio::time::timeout(Duration::from_secs(180), child.wait())
        .await
        .context("PDF extraction exceeded 180 seconds")??;
    ensure!(
        status.success(),
        "PDF extraction failed: {}",
        fs::read_to_string(&errors)?
            .chars()
            .take(4000)
            .collect::<String>()
    );
    Ok(serde_json::from_slice(&fs::read(&source)?)?)
}

pub(crate) async fn render_native(input: &Path, work: &Path) -> Result<()> {
    let manifest = work.join("regions.json");
    let receipt = work.join("region-receipt.json");
    let errors = work.join("adapter.stderr");
    let mut child = Command::new(std::env::current_exe()?)
        .arg(input)
        .args(["--mode", "epub", "--pdf-worker", "--pdf-regions"])
        .arg(&manifest)
        .arg("-o")
        .arg(&receipt)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(fs::File::create(&errors)?)
        .kill_on_drop(true)
        .spawn()?;
    let status = tokio::time::timeout(Duration::from_secs(180), child.wait())
        .await
        .context("PDF source rendering exceeded 180 seconds")??;
    ensure!(
        status.success(),
        "PDF source rendering failed: {}",
        fs::read_to_string(&errors)?
            .chars()
            .take(4000)
            .collect::<String>()
    );
    Ok(())
}

pub async fn write_epub(input: &Path, output: &Path, cover: Option<&Path>) -> Result<()> {
    ensure!(
        !output.try_exists()? && fs::symlink_metadata(output).is_err(),
        "Output already exists: {}",
        output.display()
    );
    let work = tempfile::tempdir()?;
    let layout = extract_layout(input, work.path()).await?;
    render_native(input, work.path()).await?;
    let html = crate::pdf_reconstruct::reconstruct(&layout, work.path()).await?;
    let title = if layout.title.is_empty() {
        scraper::Html::parse_fragment(&html)
            .select(&scraper::Selector::parse("h1:not(.notes-heading)").unwrap())
            .next()
            .map(|h| h.text().collect::<String>())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                input
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
    } else {
        layout.title
    };
    let article = DistillerResponse {
        content: html,
        title,
        author: layout.author,
        language: "en".into(),
        ..Default::default()
    };
    crate::epub::write_epub(&article, None, Some(work.path()), false, output, cover).await?;
    eprintln!(
        "PDF reconstruction completed: {} pages. Semantic fidelity requires source comparison.",
        layout.pages.len()
    );
    Ok(())
}
