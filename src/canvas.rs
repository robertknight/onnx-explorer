//! Pan-and-zoom canvas that draws a laid-out graph.
//!
//! The canvas holds only view state: where the viewport is over the scene and
//! how far it is zoomed in. Everything drawn comes from a [`Layout`], which is
//! computed once per graph and reused across frames.
//!
//! Two things keep this usable on large models. Only items intersecting the
//! viewport are drawn, and detail is dropped as the zoom falls: first the
//! subtitles, then the titles, then the individual boxes themselves in favour
//! of a density plot. Without the last step, zooming out over a 39,000 node
//! graph would ask the tessellator for tens of thousands of shapes every
//! frame.

use std::collections::HashSet;

use egui::{
    Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Ui, Vec2, pos2, vec2,
};

use crate::layout::{ItemKind, Layout};
use crate::model::OpCategory;
use crate::text::elide;

const MAX_ZOOM: f32 = 4.0;

/// Zoom at which node titles become legible enough to draw.
const ZOOM_SHOW_TITLE: f32 = 0.42;
/// Zoom at which there is room for the node's name under its op type.
const ZOOM_SHOW_SUBTITLE: f32 = 0.85;
/// Zoom at which edges are labelled with the type flowing along them.
const ZOOM_SHOW_EDGE_LABELS: f32 = 1.5;

/// Most boxes drawn individually before falling back to a density plot.
const MAX_DRAWN_NODES: usize = 6000;
/// Most edges drawn before they are dropped from an overview.
const MAX_DRAWN_EDGES: usize = 5000;

/// Screen size of one density plot cell, in points.
const DENSITY_CELL: f32 = 3.0;

/// Zoom used when opening a graph. Graphs are typically far taller than they
/// are wide, so scaling the whole drawing to the window would leave every node
/// a few pixels across. Opening at a legible scale on the first input is more
/// useful.
const HOME_ZOOM: f32 = 1.0;

/// Gap left above the entry node when opening a graph.
const HOME_MARGIN: f32 = 56.0;

/// How much of the drawing must remain within the viewport when panning, so
/// the view cannot be scrolled somewhere the graph is not visible at all.
const KEEP_VISIBLE: f32 = 96.0;

/// What the user did on the canvas this frame.
pub enum CanvasEvent {
    None,
    /// A box was clicked. The value is an index into [`Layout::nodes`].
    Selected(usize),
    /// The background was clicked.
    Cleared,
}

/// A change of view to apply on the next frame, once the viewport size is
/// known.
enum PendingView {
    None,
    /// Open at the entry node, at a legible zoom.
    Home,
    /// Bring this scene rect into view.
    Focus(Rect),
}

pub struct Canvas {
    /// Translation from scene coordinates to the viewport, in points.
    pan: Vec2,
    zoom: f32,
    pending: PendingView,
    /// Whether the last frame fell back to the density plot.
    simplified: bool,
    /// Reused across frames so the density plot does not allocate.
    density: Vec<u32>,
}

impl Canvas {
    pub fn new() -> Canvas {
        Canvas {
            pan: Vec2::ZERO,
            zoom: HOME_ZOOM,
            pending: PendingView::Home,
            simplified: false,
            density: Vec::new(),
        }
    }

    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// Whether the view is currently too dense to draw in full detail.
    pub fn is_simplified(&self) -> bool {
        self.simplified
    }

    /// Open the graph at its entry node, at a legible zoom.
    pub fn request_home(&mut self) {
        self.pending = PendingView::Home;
    }

    /// Scroll and zoom so that `rect` is centred and readable.
    pub fn focus_on(&mut self, rect: Rect) {
        self.pending = PendingView::Focus(rect);
    }

    pub fn show(&mut self, ui: &mut Ui, layout: &Layout, selected: Option<usize>) -> CanvasEvent {
        let (viewport, response) =
            ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

        match std::mem::replace(&mut self.pending, PendingView::None) {
            PendingView::None => {}
            PendingView::Home => self.home(viewport, layout),
            PendingView::Focus(target) => self.focus(viewport, target),
        }

        self.handle_input(ui, viewport, &response, layout.bounds);
        // Also applied outside `zoom_at`, since resizing the window changes
        // the zoom at which the drawing fits.
        self.zoom = self.zoom.clamp(min_zoom(viewport, layout.bounds), MAX_ZOOM);
        self.clamp_pan(viewport, layout.bounds);
        self.draw(ui, viewport, layout, selected);

        if response.clicked()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let scene = self.to_scene(viewport, pointer);
            let hit = layout
                .nodes
                .iter()
                .position(|node| node.rect.contains(scene));
            return match hit {
                Some(index) => CanvasEvent::Selected(index),
                None => CanvasEvent::Cleared,
            };
        }

        CanvasEvent::None
    }
}

// View transform.
impl Canvas {
    fn to_screen(&self, viewport: Rect, point: Pos2) -> Pos2 {
        viewport.min + (point.to_vec2() * self.zoom + self.pan)
    }

    fn to_screen_rect(&self, viewport: Rect, rect: Rect) -> Rect {
        Rect::from_min_max(
            self.to_screen(viewport, rect.min),
            self.to_screen(viewport, rect.max),
        )
    }

    fn to_scene(&self, viewport: Rect, point: Pos2) -> Pos2 {
        (((point - viewport.min) - self.pan) / self.zoom).to_pos2()
    }

    /// The region of the scene currently visible, used for culling.
    fn visible_scene(&self, viewport: Rect) -> Rect {
        Rect::from_min_max(
            self.to_scene(viewport, viewport.min),
            self.to_scene(viewport, viewport.max),
        )
    }

    /// Place the entry node near the top of the viewport at a readable zoom.
    fn home(&mut self, viewport: Rect, layout: &Layout) {
        self.zoom = HOME_ZOOM;

        let Some(entry) = layout.entry_node() else {
            self.pan = viewport.size() / 2.0;
            return;
        };

        let rect = layout.nodes[entry].rect;
        self.pan = vec2(
            viewport.width() / 2.0 - rect.center().x * self.zoom,
            HOME_MARGIN - rect.min.y * self.zoom,
        );
    }

    /// Keep part of the drawing on screen.
    ///
    /// Without this it is easy to fling the view into empty space on a large
    /// graph and be left with no way back except the toolbar.
    fn clamp_pan(&mut self, viewport: Rect, bounds: Rect) {
        if !bounds.is_finite() {
            return;
        }
        let size = viewport.size();
        self.pan.x = clamp_axis(
            self.pan.x,
            bounds.min.x * self.zoom,
            bounds.max.x * self.zoom,
            size.x,
            KEEP_VISIBLE,
        );
        self.pan.y = clamp_axis(
            self.pan.y,
            bounds.min.y * self.zoom,
            bounds.max.y * self.zoom,
            size.y,
            KEEP_VISIBLE,
        );
    }

    fn focus(&mut self, viewport: Rect, target: Rect) {
        if self.zoom < ZOOM_SHOW_SUBTITLE {
            self.zoom = ZOOM_SHOW_SUBTITLE;
        }
        self.centre_on(viewport, target.center());
    }

    fn centre_on(&mut self, viewport: Rect, scene_point: Pos2) {
        self.pan = viewport.size() / 2.0 - scene_point.to_vec2() * self.zoom;
    }

    /// Zoom by `factor`, keeping the scene point under the pointer fixed.
    fn zoom_at(&mut self, viewport: Rect, pointer: Pos2, factor: f32, lowest: f32) {
        let previous = self.zoom;
        self.zoom = (self.zoom * factor).clamp(lowest, MAX_ZOOM);
        let ratio = self.zoom / previous;
        let local = pointer - viewport.min;
        self.pan = local - (local - self.pan) * ratio;
    }

    fn handle_input(&mut self, ui: &Ui, viewport: Rect, response: &egui::Response, bounds: Rect) {
        if response.dragged() {
            self.pan += response.drag_delta();
        }

        if !response.hovered() {
            return;
        }

        let (zoom_delta, scroll) = ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta()));

        // egui folds pinch gestures and ctrl+scroll into `zoom_delta`, leaving
        // plain scrolling to pan.
        if zoom_delta != 1.0 {
            if let Some(pointer) = response.hover_pos() {
                self.zoom_at(viewport, pointer, zoom_delta, min_zoom(viewport, bounds));
            }
        } else if scroll != Vec2::ZERO {
            self.pan += scroll;
        }
    }
}

// Drawing.
impl Canvas {
    fn draw(&mut self, ui: &Ui, viewport: Rect, layout: &Layout, selected: Option<usize>) {
        let painter = ui.painter_at(viewport);
        let palette = Palette::new(ui.visuals());
        painter.rect_filled(viewport, 0, palette.background);

        let visible = self.visible_scene(viewport);

        let visible_nodes: Vec<usize> = layout
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| visible.intersects(node.rect))
            .map(|(index, _)| index)
            .collect();

        self.simplified = visible_nodes.len() > MAX_DRAWN_NODES;

        let surface = Surface {
            painter: &painter,
            viewport,
            palette: &palette,
        };

        if self.simplified {
            self.draw_density(surface, layout, &visible_nodes);
            return;
        }

        let (highlight_edges, highlight_nodes) = self.highlights(layout, selected);
        self.draw_edges(surface, layout, visible, &highlight_edges);
        self.draw_nodes(surface, layout, &visible_nodes, selected, &highlight_nodes);
    }

    /// Edges and nodes adjacent to the selection, which are drawn emphasised.
    fn highlights(
        &self,
        layout: &Layout,
        selected: Option<usize>,
    ) -> (HashSet<usize>, HashSet<usize>) {
        let mut edges = HashSet::new();
        let mut nodes = HashSet::new();
        let Some(selected) = selected else {
            return (edges, nodes);
        };
        for &edge_index in layout.incident_edges(selected) {
            edges.insert(edge_index);
            let edge = &layout.edges[edge_index];
            nodes.insert(edge.from);
            nodes.insert(edge.to);
        }
        nodes.remove(&selected);
        (edges, nodes)
    }

    fn draw_edges(
        &self,
        surface: Surface,
        layout: &Layout,
        visible: Rect,
        highlighted: &HashSet<usize>,
    ) {
        let Surface {
            painter,
            viewport,
            palette,
        } = surface;
        let width = (1.1 * self.zoom).clamp(0.7, 2.0);
        let mut drawn = 0;

        for (index, edge) in layout.edges.iter().enumerate() {
            if !visible.intersects(edge.bounds) {
                continue;
            }
            drawn += 1;
            if drawn > MAX_DRAWN_EDGES && !highlighted.contains(&index) {
                continue;
            }

            let is_highlighted = highlighted.contains(&index);
            let stroke = if is_highlighted {
                Stroke::new(width * 2.0, palette.edge_highlight)
            } else {
                Stroke::new(width, palette.edge)
            };

            let points: Vec<Pos2> = edge
                .points
                .iter()
                .map(|point| self.to_screen(viewport, *point))
                .collect();
            painter.add(Shape::line(points.clone(), stroke));

            if self.zoom >= ZOOM_SHOW_TITLE
                && let [.., before, tip] = points.as_slice()
            {
                arrow_head(painter, *tip, *before, 7.0 * self.zoom, stroke.color);
            }

            if self.zoom >= ZOOM_SHOW_EDGE_LABELS && !edge.label.is_empty() {
                painter.text(
                    polyline_middle(&points) + vec2(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    elide(&edge.label, 28),
                    FontId::proportional(9.0 * self.zoom),
                    palette.text_weak,
                );
            }
        }
    }

    fn draw_nodes(
        &self,
        surface: Surface,
        layout: &Layout,
        visible_nodes: &[usize],
        selected: Option<usize>,
        highlighted: &HashSet<usize>,
    ) {
        let Surface {
            painter,
            viewport,
            palette,
        } = surface;
        let radius = (4.0 * self.zoom).clamp(0.0, 6.0) as u8;
        let show_title = self.zoom >= ZOOM_SHOW_TITLE;
        let show_subtitle = self.zoom >= ZOOM_SHOW_SUBTITLE;

        for &index in visible_nodes {
            let node = &layout.nodes[index];
            let rect = self.to_screen_rect(viewport, node.rect);
            let is_selected = selected == Some(index);

            painter.rect_filled(
                rect,
                radius,
                palette.fill(node.kind, node.category, is_selected),
            );

            let stroke = if is_selected {
                Stroke::new(2.0, palette.selected_stroke)
            } else if highlighted.contains(&index) {
                Stroke::new(1.5, palette.edge_highlight)
            } else {
                Stroke::new(1.0, palette.node_stroke)
            };
            painter.rect_stroke(rect, radius, stroke, StrokeKind::Inside);

            if !show_title {
                continue;
            }

            let title_size = 13.0 * self.zoom;
            let capacity = ((rect.width() - 10.0) / (title_size * 0.56)).max(0.0) as usize;
            let centre = rect.center();

            if show_subtitle && !node.subtitle.is_empty() {
                painter.text(
                    pos2(centre.x, centre.y - rect.height() * 0.16),
                    Align2::CENTER_CENTER,
                    elide(&node.title, capacity),
                    FontId::proportional(title_size),
                    palette.text,
                );
                let subtitle_size = 10.0 * self.zoom;
                let subtitle_capacity =
                    ((rect.width() - 10.0) / (subtitle_size * 0.56)).max(0.0) as usize;
                painter.text(
                    pos2(centre.x, centre.y + rect.height() * 0.24),
                    Align2::CENTER_CENTER,
                    elide(&node.subtitle, subtitle_capacity),
                    FontId::proportional(subtitle_size),
                    palette.text_weak,
                );
            } else {
                painter.text(
                    centre,
                    Align2::CENTER_CENTER,
                    elide(&node.title, capacity),
                    FontId::proportional(title_size),
                    palette.text,
                );
            }
        }
    }

    /// Draw a coarse density plot instead of individual boxes.
    ///
    /// At this zoom a node covers a pixel or two, so plotting how many fall in
    /// each cell shows the shape of the model at a fraction of the cost, and
    /// reads more clearly than thousands of overlapping slivers.
    fn draw_density(&mut self, surface: Surface, layout: &Layout, visible_nodes: &[usize]) {
        let Surface {
            painter,
            viewport,
            palette,
        } = surface;
        let columns = (viewport.width() / DENSITY_CELL).ceil().max(1.0) as usize;
        let rows = (viewport.height() / DENSITY_CELL).ceil().max(1.0) as usize;

        self.density.clear();
        self.density.resize(columns * rows, 0);

        let mut peak = 1;
        for &index in visible_nodes {
            let centre = self.to_screen(viewport, layout.nodes[index].rect.center());
            let column = ((centre.x - viewport.min.x) / DENSITY_CELL) as usize;
            let row = ((centre.y - viewport.min.y) / DENSITY_CELL) as usize;
            if column >= columns || row >= rows {
                continue;
            }
            let cell = &mut self.density[row * columns + column];
            *cell += 1;
            peak = peak.max(*cell);
        }

        for row in 0..rows {
            for column in 0..columns {
                let count = self.density[row * columns + column];
                if count == 0 {
                    continue;
                }
                // Square-root scaling keeps sparse regions visible next to the
                // dense trunk of the graph.
                let weight = (count as f32 / peak as f32).sqrt().clamp(0.25, 1.0);
                let cell = Rect::from_min_size(
                    pos2(
                        viewport.min.x + column as f32 * DENSITY_CELL,
                        viewport.min.y + row as f32 * DENSITY_CELL,
                    ),
                    Vec2::splat(DENSITY_CELL),
                );
                painter.rect_filled(cell, 0, palette.density.gamma_multiply(weight));
            }
        }
    }
}

/// The middle of a polyline: its central point where there is an odd number
/// of them, or the midpoint of the two central points where there is not.
///
/// Indexing the halfway element instead lands on an endpoint for a two point
/// line, which put edge labels on top of the node the edge arrives at.
fn polyline_middle(points: &[Pos2]) -> Pos2 {
    match points.len() {
        0 => Pos2::ZERO,
        1 => points[0],
        len if len % 2 == 1 => points[len / 2],
        len => {
            let (before, after) = (points[len / 2 - 1], points[len / 2]);
            before + (after - before) * 0.5
        }
    }
}

fn arrow_head(painter: &egui::Painter, tip: Pos2, from: Pos2, size: f32, color: Color32) {
    let direction = tip - from;
    if direction.length_sq() < 0.01 {
        return;
    }
    let direction = direction.normalized();
    let perpendicular = vec2(-direction.y, direction.x);
    let base = tip - direction * size;
    painter.add(Shape::convex_polygon(
        vec![
            tip,
            base + perpendicular * size * 0.42,
            base - perpendicular * size * 0.42,
        ],
        color,
        Stroke::NONE,
    ));
}

/// The target the drawing helpers paint onto. Bundled so that they take a
/// destination rather than repeating three arguments apiece.
#[derive(Copy, Clone)]
struct Surface<'a> {
    painter: &'a egui::Painter,
    viewport: Rect,
    palette: &'a Palette,
}

/// Colours for the canvas.
///
/// The category fills are pale tints so that the text over them stays readable
/// and no one kind of box dominates — except compute, which is the arithmetic
/// the rest of the graph exists to feed, and is saturated enough to pick out
/// from across a large model.
struct Palette {
    background: Color32,
    node_fill: Color32,
    compute_fill: Color32,
    unary_fill: Color32,
    binary_fill: Color32,
    /// Shared by normalization, pooling and reduction, which all replace an
    /// extent of a tensor with a statistic over it.
    statistic_fill: Color32,
    movement_fill: Color32,
    /// Blocks keep their own hue but take the saturation and lightness of
    /// [`Palette::compute_fill`], so a block reads as strongly as the
    /// arithmetic it stands in for.
    group_fill: Color32,
    node_stroke: Color32,
    input_fill: Color32,
    output_fill: Color32,
    selected_fill: Color32,
    selected_stroke: Color32,
    edge: Color32,
    edge_highlight: Color32,
    density: Color32,
    text: Color32,
    text_weak: Color32,
}

impl Palette {
    fn new(visuals: &egui::Visuals) -> Palette {
        if visuals.dark_mode {
            Palette {
                background: Color32::from_rgb(24, 24, 28),
                node_fill: Color32::from_rgb(46, 46, 54),
                compute_fill: Color32::from_rgb(112, 78, 24),
                unary_fill: Color32::from_rgb(36, 54, 58),
                binary_fill: Color32::from_rgb(48, 53, 42),
                statistic_fill: Color32::from_rgb(59, 44, 54),
                movement_fill: Color32::from_rgb(42, 47, 60),
                group_fill: Color32::from_rgb(48, 24, 112),
                node_stroke: Color32::from_rgb(84, 84, 96),
                input_fill: Color32::from_rgb(32, 62, 50),
                output_fill: Color32::from_rgb(34, 50, 76),
                selected_fill: Color32::from_rgb(64, 68, 96),
                selected_stroke: Color32::from_rgb(132, 164, 255),
                edge: Color32::from_rgb(96, 96, 110),
                edge_highlight: Color32::from_rgb(132, 164, 255),
                density: Color32::from_rgb(150, 160, 200),
                text: Color32::from_rgb(226, 226, 234),
                text_weak: Color32::from_rgb(150, 150, 162),
            }
        } else {
            Palette {
                background: Color32::from_rgb(250, 250, 252),
                node_fill: Color32::from_rgb(255, 255, 255),
                compute_fill: Color32::from_rgb(252, 218, 150),
                unary_fill: Color32::from_rgb(232, 247, 245),
                binary_fill: Color32::from_rgb(242, 247, 230),
                statistic_fill: Color32::from_rgb(252, 238, 243),
                movement_fill: Color32::from_rgb(238, 242, 249),
                group_fill: Color32::from_rgb(188, 150, 252),
                node_stroke: Color32::from_rgb(188, 188, 200),
                input_fill: Color32::from_rgb(226, 244, 234),
                output_fill: Color32::from_rgb(224, 236, 252),
                selected_fill: Color32::from_rgb(222, 232, 255),
                selected_stroke: Color32::from_rgb(48, 96, 220),
                edge: Color32::from_rgb(154, 154, 166),
                edge_highlight: Color32::from_rgb(48, 96, 220),
                density: Color32::from_rgb(70, 80, 130),
                text: Color32::from_rgb(28, 28, 34),
                text_weak: Color32::from_rgb(112, 112, 124),
            }
        }
    }

    fn fill(&self, kind: ItemKind, category: Option<OpCategory>, selected: bool) -> Color32 {
        if selected {
            return self.selected_fill;
        }
        match kind {
            ItemKind::Op(_) => match category {
                Some(OpCategory::Compute) => self.compute_fill,
                Some(OpCategory::Unary) => self.unary_fill,
                Some(OpCategory::Binary) => self.binary_fill,
                Some(OpCategory::Normalization | OpCategory::Pooling | OpCategory::Reduction) => {
                    self.statistic_fill
                }
                Some(OpCategory::DataMovement) => self.movement_fill,
                Some(OpCategory::Other) | None => self.node_fill,
            },
            ItemKind::Group(_) => self.group_fill,
            ItemKind::Input(_) => self.input_fill,
            ItemKind::Output(_) => self.output_fill,
        }
    }
}

/// The furthest the view may zoom out: the point at which the whole drawing
/// fits in the viewport.
///
/// Zooming out beyond this only shrinks a drawing that is already wholly
/// visible. The result is capped at the zoom a graph opens at, so a drawing
/// smaller than the window can still be viewed at its natural size rather
/// than being forced to fill it.
fn min_zoom(viewport: Rect, bounds: Rect) -> f32 {
    if !bounds.is_finite() || bounds.width() <= 0.0 || bounds.height() <= 0.0 {
        return HOME_ZOOM;
    }
    let scale_x = viewport.width() / bounds.width();
    let scale_y = viewport.height() / bounds.height();
    scale_x.min(scale_y).min(HOME_ZOOM)
}

/// Clamp a pan offset along one axis so that at least `keep` points of the
/// content stay inside a viewport of length `size`.
///
/// `min` and `max` are the content's extent along the axis, already scaled by
/// the zoom. Content shorter than `keep` is held entirely on screen rather
/// than being allowed to hang off an edge.
fn clamp_axis(pan: f32, min: f32, max: f32, size: f32, keep: f32) -> f32 {
    let keep = keep.min(max - min).min(size);
    // The content's far edge must not rise above `keep`, and its near edge must
    // not fall below `size - keep`.
    let lowest = keep - max;
    let highest = size - keep - min;
    if lowest > highest {
        // Content shorter than the slack available: centre it.
        (lowest + highest) / 2.0
    } else {
        pan.clamp(lowest, highest)
    }
}

#[cfg(test)]
mod tests {
    use egui::{Pos2, Rect, Vec2, pos2, vec2};

    use super::{HOME_ZOOM, KEEP_VISIBLE, clamp_axis, min_zoom, polyline_middle};

    fn viewport() -> Rect {
        Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 800.0))
    }

    fn drawing(width: f32, height: f32) -> Rect {
        Rect::from_min_size(Pos2::ZERO, vec2(width, height))
    }

    #[test]
    fn test_polyline_middle_of_a_straight_edge() {
        // Most edges are two points. Indexing the halfway element gives the
        // arrow tip, which put the label on top of the target node.
        let middle = polyline_middle(&[pos2(0.0, 0.0), pos2(10.0, 100.0)]);
        assert_eq!(middle, pos2(5.0, 50.0));
    }

    #[test]
    fn test_polyline_middle_of_a_routed_edge() {
        // An odd number of points has a central one.
        let points = [pos2(0.0, 0.0), pos2(20.0, 50.0), pos2(0.0, 100.0)];
        assert_eq!(polyline_middle(&points), pos2(20.0, 50.0));

        // An even number falls between the two central points.
        let points = [
            pos2(0.0, 0.0),
            pos2(20.0, 40.0),
            pos2(20.0, 80.0),
            pos2(0.0, 120.0),
        ];
        assert_eq!(polyline_middle(&points), pos2(20.0, 60.0));
    }

    #[test]
    fn test_polyline_middle_of_a_degenerate_line() {
        assert_eq!(polyline_middle(&[]), Pos2::ZERO);
        assert_eq!(polyline_middle(&[pos2(3.0, 4.0)]), pos2(3.0, 4.0));
    }

    #[test]
    fn test_min_zoom_stops_where_the_drawing_fits() {
        // Twice the viewport in both directions, so it fits at half scale.
        assert_eq!(min_zoom(viewport(), drawing(2000.0, 1600.0)), 0.5);
    }

    #[test]
    fn test_min_zoom_follows_the_tighter_axis() {
        // The very tall drawings deep models produce are limited by height.
        assert_eq!(min_zoom(viewport(), drawing(1000.0, 80_000.0)), 0.01);
    }

    #[test]
    fn test_min_zoom_leaves_a_small_drawing_at_its_natural_size() {
        // Fitting this would mean magnifying it ten times, which is not what
        // a limit on zooming out should do.
        assert_eq!(min_zoom(viewport(), drawing(100.0, 80.0)), HOME_ZOOM);
    }

    #[test]
    fn test_min_zoom_handles_an_empty_drawing() {
        let zoom = min_zoom(viewport(), Rect::from_min_size(Pos2::ZERO, Vec2::ZERO));
        assert!(
            zoom.is_finite(),
            "a zero-sized drawing must not divide by zero"
        );
        assert_eq!(zoom, HOME_ZOOM);
    }

    #[test]
    fn test_clamp_axis_allows_free_movement_in_range() {
        // A 1000pt drawing in a 500pt viewport: a pan that keeps the content
        // across the whole viewport is left alone.
        assert_eq!(clamp_axis(-250.0, 0.0, 1000.0, 500.0, KEEP_VISIBLE), -250.0);
    }

    #[test]
    fn test_clamp_axis_keeps_content_on_screen() {
        // Scrolled far past the end: the content's far edge is pulled back to
        // `keep` inside the viewport.
        let pan = clamp_axis(-5000.0, 0.0, 1000.0, 500.0, KEEP_VISIBLE);
        assert_eq!(pan, KEEP_VISIBLE - 1000.0);
        assert!(1000.0 + pan >= KEEP_VISIBLE);

        // Scrolled far before the start, in the other direction.
        let pan = clamp_axis(5000.0, 0.0, 1000.0, 500.0, KEEP_VISIBLE);
        assert_eq!(pan, 500.0 - KEEP_VISIBLE);
        assert!(pan <= 500.0 - KEEP_VISIBLE);
    }

    #[test]
    fn test_clamp_axis_centres_content_smaller_than_the_slack() {
        // A 20pt drawing is shorter than `keep`, so it is held fully on screen
        // rather than being allowed to hang off an edge.
        let pan = clamp_axis(-9000.0, 0.0, 20.0, 500.0, KEEP_VISIBLE);
        assert!(pan >= 0.0 && pan + 20.0 <= 500.0, "pan was {pan}");
    }

    #[test]
    fn test_clamp_axis_handles_content_larger_than_viewport() {
        // A drawing far taller than the window still permits scrolling through
        // its whole length.
        let top = clamp_axis(1e9, 0.0, 100_000.0, 800.0, KEEP_VISIBLE);
        let bottom = clamp_axis(-1e9, 0.0, 100_000.0, 800.0, KEEP_VISIBLE);
        assert!(top > bottom);
        assert!(
            bottom <= -99_000.0,
            "should reach the far end, got {bottom}"
        );
    }
}
