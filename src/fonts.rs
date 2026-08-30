//! Fonts used by the UI.
//!
//! egui's built-in fonts are Ubuntu-Light and Hack, which look out of place next
//! to native applications and offer no bold face. On platforms where the system
//! UI font can be loaded we use that instead, which also gives us a real bold to
//! set headings in.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use egui::epaint::text::{FontData, FontTweak, VariationCoords};
use egui::{FontDefinitions, FontFamily};

/// Name of the font family holding the bold face.
///
/// Always registered, so callers can name it without checking whether the
/// platform supplied a font that actually has a bold weight. Use
/// [`has_real_bold`] to find out whether it will look any different.
const BOLD_FAMILY: &str = "bold";

static REAL_BOLD: AtomicBool = AtomicBool::new(false);

/// Install the UI fonts into `ctx`. Call once, before the first frame.
pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();

    // Keep egui's fonts after the system one so its emoji coverage still
    // applies to glyphs the system font is missing.
    let mut bold_family = fonts.families[&FontFamily::Proportional].clone();

    if let Some((regular, bold)) = system_ui_font() {
        fonts
            .font_data
            .insert("system-ui".to_owned(), Arc::new(regular));
        fonts
            .font_data
            .insert("system-ui-bold".to_owned(), Arc::new(bold));

        fonts
            .families
            .entry(FontFamily::Proportional)
            .or_default()
            .insert(0, "system-ui".to_owned());
        bold_family.insert(0, "system-ui-bold".to_owned());

        REAL_BOLD.store(true, Ordering::Relaxed);
    }

    fonts
        .families
        .insert(FontFamily::Name(BOLD_FAMILY.into()), bold_family);

    ctx.set_fonts(fonts);
}

/// Font to set bold text in, at `size` points.
pub fn bold(size: f32) -> egui::FontId {
    egui::FontId::new(size, FontFamily::Name(BOLD_FAMILY.into()))
}

/// Whether [`bold`] returns a font that is genuinely heavier than the body text.
pub fn has_real_bold() -> bool {
    REAL_BOLD.load(Ordering::Relaxed)
}

/// Load the system UI font in a regular and a bold weight, if it is available.
#[cfg(target_os = "macos")]
fn system_ui_font() -> Option<(FontData, FontData)> {
    // SF Pro, the macOS UI font. It is variable, so both weights come from the
    // same 8MB file: read it once and share it between the two faces.
    let bytes = std::fs::read("/System/Library/Fonts/SFNS.ttf").ok()?;
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());

    // `opsz` is the optical size axis. Its default of 28 is the display cut,
    // drawn for headlines; the whole UI runs at text sizes, so pin it to the
    // low end of the axis where the letterforms are opened up for small text.
    let face = |weight: f32| {
        FontData::from_static(bytes).tweak(FontTweak {
            coords: VariationCoords::new([(b"opsz", 17.0), (b"wght", weight)]),
            ..Default::default()
        })
    };

    Some((face(400.0), face(700.0)))
}

#[cfg(not(target_os = "macos"))]
fn system_ui_font() -> Option<(FontData, FontData)> {
    None
}
