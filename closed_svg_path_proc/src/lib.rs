use proc_macro2::Literal;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;

use quote::{quote, format_ident};
use syn::{parse_macro_input, LitStr, Token};
use std::env;
use std::fs;
use std::path::Path;
use svgtypes::{PathParser, PathSegment};
use closed_svg_path::{BezierSegment};

// Custom parser for the macro input that accepts any token sequence for file_id
struct MacroInput {
    file_id_tokens: proc_macro2::TokenStream,
    file_path: String,
}

impl syn::parse::Parse for MacroInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Parse tokens until we hit a comma
        let mut file_id_tokens = proc_macro2::TokenStream::new();
        while !input.peek(Token![,]) && !input.is_empty() {
            let token: proc_macro2::TokenTree = input.parse()?;
            file_id_tokens.extend(std::iter::once(token));
        }
        
        let _: Token![,] = input.parse()?;
        let file_path: LitStr = input.parse()?;
        
        Ok(MacroInput {
            file_id_tokens,
            file_path: file_path.value(),
        })
    }
}

// Helper function to create a safe identifier from token stream
fn tokens_to_ident_suffix(tokens: &proc_macro2::TokenStream) -> String {
    let token_str = tokens.to_string();
    token_str
        .chars()
        .filter_map(|c| {
            if c.is_alphanumeric() || c == '_' { 
                Some(c) 
            } else if c.is_whitespace() { 
                Some('_') 
            } else { 
                None 
            }
        })
        .collect::<String>()
        .replace("__", "_")
        .trim_matches('_')
        .to_string()
}

/// Find the average between two points
const fn avg_points(a: [f32; 2], b: [f32; 2]) -> [f32; 2]
{
    [(a[0] + b[0]) / 2., (a[1] + b[1]) / 2.]
}

/// Split a Bezier segment in two
const fn split_bez_segment(seg: BezierSegment) -> (BezierSegment, BezierSegment)
{
    // Split the bezier segment into two segments
    let p0 = seg.0[0];
    let p1 = seg.0[1];
    let p2 = seg.0[2];
    let p3 = seg.0[3];
    let p0p = avg_points(p0, p1);
    let p1p = avg_points(p1, p2);
    let p2p = avg_points(p2, p3);
    let p0pp = avg_points(p0p, p1p);
    let p1pp = avg_points(p1p, p2p);
    let p0ppp = avg_points(p0pp, p1pp);

    // First sub-curve: [p0,p0p,p0pp,p0ppp]
    // Second sub-curve:[p0ppp, p1pp, p2p, p3]

    return ( BezierSegment([p0,p0p,p0pp,p0ppp]), BezierSegment([p0ppp, p1pp, p2p, p3]) )
}

/// Convert Bezier segments to closed polygon vertices

fn bez_segs_to_closed_poly_points(bezier_segments: &Vec<[[f32; 2]; 4]> , poly_points: &mut  Vec<[i32; 2]>)
{
    for seg in bezier_segments {
        // Split each bezier segment into eight segments for approximation
        let (seg0, seg1) = split_bez_segment(BezierSegment(*seg));
        let (seg00, seg01) = split_bez_segment(seg0);
        let (seg10, seg11) = split_bez_segment(seg1);
        let (seg000, seg001) = split_bez_segment(seg00);
        let (seg010, seg011) = split_bez_segment(seg01);
        let (seg100, seg101) = split_bez_segment(seg10);
        let (seg110, seg111) = split_bez_segment(seg11);

        poly_points.push(round_segment_start(seg000)); 
        poly_points.push(round_segment_start(seg001)); 
        poly_points.push(round_segment_start(seg010)); 
        poly_points.push(round_segment_start(seg011)); 
        poly_points.push(round_segment_start(seg100)); 
        poly_points.push(round_segment_start(seg101)); 
        poly_points.push(round_segment_start(seg110)); 
        poly_points.push(round_segment_start(seg111)); 
    }

    if let Some(first_poly_pt) = poly_points.first() {
        poly_points.push(*first_poly_pt);
    }
}

/// Select the point at the start of the segment, at round it to the closest integer value
fn round_segment_start(seg: BezierSegment) -> [i32; 2]
{
    [seg.0[0][0].round() as i32, seg.0[0][1].round() as i32]
}

/// Bounding box tuple for top_left.x, top_left.y, size.width, size.height 
type BboxTuple =  (i32, i32, u32, u32);

/// Helper function to calculate bounding box of polygon vertices at compile time
fn calculate_bounding_box(vertices: &[[i32; 2]]) -> BboxTuple {
    // compute bounding box
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for v in vertices {
        min_x = min_x.min(v[0]);
        min_y = min_y.min(v[1]);
        max_x = max_x.max(v[0]);
        max_y = max_y.max(v[1]);
    }

    let width = (max_x - min_x) as u32;
    let height = (max_y - min_y) as u32;
    
    (min_x, min_y, width, height)
}

/// Helper struct for first calculating then analyzing polygon edges
#[derive(Clone, Copy, Debug)]
struct Edge {
    y_max: i32,     // exclusive: edge is active for y in [y_min, y_max)
    current_x: f64, // x at the current scanline
    inv_slope: f64, // dx/dy
}

/// Returns a Vec<Vec<i32>>, where result[y - bbox.min_y] holds the integer x intersections
/// (rounded) for scanline y, for y in [bbox.min_y, bbox.max_y).
///
/// - `vertices` must be a closed polygon (first == last) or the function will treat it cyclically.
/// - Horizontal edges are skipped, because their fill is covered by neighboring edges.
///
/// This code runs at compile time, during macro expansion, so we're not overly concerned with performance.
fn scanline_intersections(vertices: &[[i32; 2]], bbox: BboxTuple) -> Vec<Vec<i32>> {
    assert!(vertices.len() >= 3, "need at least 3 vertices for fill");
    let min_x = bbox.0;
    let max_x = min_x + bbox.2 as i32;
    let min_y = bbox.1;
    let max_y = min_y + bbox.3 as i32;

    assert!(min_y != max_y, "polygon shape is a horizontal line");
    assert!(min_x != max_x, "polygon shape is a vertical line");

    // Edge Table: bucket edges by ymin relative to min_y.
    let height = (max_y - min_y) as usize; // number of integer scanlines in [min_y, max_y)
    let mut edge_table: Vec<Vec<Edge>> = vec![Vec::new(); height];

    // Build edge table from polygon
    let n = vertices.len();
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n]; // wrap last->first vertex 

        let x0 = a[0];
        let y0 = a[1];
        let x1 = b[0];
        let y1 = b[1];

        // Skip horizontal edges
        if y0 == y1 {
            continue;
        }

        // y_min, y_max for half-open interval [ymin, ymax)
        let (y_min, x_at_ymin, y_max, x_at_ymax) = 
            if y0 < y1 {
                (y0, x0 as f64, y1, x1 as f64)
            } else {
                (y1, x1 as f64, y0, x0 as f64)
            };

        let dy = (y_max - y_min) as f64;
        // inv_slope = dx / dy
        let inv_slope = (x_at_ymax - x_at_ymin) / dy;

        let bucket_index = (y_min - min_y) as isize;
        if bucket_index >= 0 && (bucket_index as usize) < edge_table.len() {
            edge_table[bucket_index as usize].push(Edge {
                y_max,
                current_x: x_at_ymin, // starting x at scanline y_min
                inv_slope,
            });
        }
    }

    // Active Edge Table
    let mut aet: Vec<Edge> = Vec::new();
    // let mut result = Vec::<Vec::<i32>>::with_capacity(height);
    let mut result: Vec<Vec<i32>> = vec![Vec::new(); height];

    // Scan each integer scanline y in [min_y, max_y)
    for rel_scan_yidx in 0..height {
        let y = min_y + rel_scan_yidx as i32;

        // 1) Add edges whose ymin == y
        let new_edges = std::mem::take(&mut edge_table[rel_scan_yidx]);
        for e in new_edges {
            aet.push(e);
        }

        // 2) Remove edges for which y >= y_max (they are no longer active).
        // Note: because we use half-open [ymin, ymax) convention,
        // an edge with y_max == y will not be active at y.
        aet.retain(|e| y < e.y_max);

        // 3) For each active edge compute intersection x at this scanline.
        // We stored current_x as x_at_ymin and will update it after use.
        // Make a temporary vector of intersections (x)
        let mut xs: Vec<f64> = Vec::with_capacity(aet.len());
        for e in &aet {
            xs.push(e.current_x);
        }

        // 4) Sort intersections
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // 5) Convert to integers (rounded). You can change rounding strategy here.
        let xi: Vec<i32> = xs.into_iter().map(|xf| xf.round() as i32).collect();
        result[rel_scan_yidx] = xi;
        assert!(result[rel_scan_yidx].len() >= 2, "Each scanline should have min 2 intersections.");

        // 6) Increment current_x for each active edge for next scanline (y+1)
        for e in &mut aet {
            e.current_x += e.inv_slope;
        }
    }

    return result;
}


fn vec_to_scanline_intersections_expr(data: &Vec<Vec<i32>>) -> TokenStream2 {
    let inner_arrays: Vec<TokenStream2> = data
        .into_iter()
        .map(|row| {
            let elements = row.into_iter();
            quote! { &[#(#elements),*] }
        })
        .collect();
    
    quote! {
        ScanlineIntersections {
            data: &[#(#inner_arrays),*],
        }
    }
}

/// Convert an elliptical arc to cubic Bezier segments
/// Returns a vector of Bezier segments that approximate the arc
fn elliptical_arc_to_bezier_segments(
    start: [f32; 2],
    end: [f32; 2],
    rx: f32,
    ry: f32,
    x_axis_rotation: f32,
    large_arc_flag: bool,
    sweep_flag: bool,
) -> Vec<[[f32; 2]; 4]> {
    // Handle degenerate cases
    if rx == 0.0 || ry == 0.0 {
        // Degenerate to a line
        let p0 = start;
        let p3 = end;
        let delta = [(p3[0] - p0[0]) / 3.0, (p3[1] - p0[1]) / 3.0];
        let p1 = [p0[0] + delta[0], p0[1] + delta[1]];
        let p2 = [p0[0] + 2.0 * delta[0], p0[1] + 2.0 * delta[1]];
        return vec![[p0, p1, p2, p3]];
    }

    if start[0] == end[0] && start[1] == end[1] {
        // Start and end are the same, no arc to draw
        return vec![];
    }

    let mut rx = rx.abs();
    let mut ry = ry.abs();
    let phi = x_axis_rotation.to_radians();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    // Step 1: Compute (x1, y1)
    let dx = (start[0] - end[0]) / 2.0;
    let dy = (start[1] - end[1]) / 2.0;
    let x1 = cos_phi * dx + sin_phi * dy;
    let y1 = -sin_phi * dx + cos_phi * dy;

    // Step 2: Ensure radii are large enough
    let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry);
    if lambda > 1.0 {
        rx *= lambda.sqrt();
        ry *= lambda.sqrt();
    }

    // Step 3: Compute (cx, cy)
    let sign = if large_arc_flag == sweep_flag { -1.0 } else { 1.0 };
    let sq = ((rx * rx) * (ry * ry) - (rx * rx) * (y1 * y1) - (ry * ry) * (x1 * x1))
        / ((rx * rx) * (y1 * y1) + (ry * ry) * (x1 * x1));
    let coeff = sign * sq.max(0.0).sqrt();
    let cx1 = coeff * (rx * y1 / ry);
    let cy1 = coeff * -(ry * x1 / rx);

    // Step 4: Compute (cx, cy)
    let sx2 = (start[0] + end[0]) / 2.0;
    let sy2 = (start[1] + end[1]) / 2.0;
    let cx = sx2 + (cos_phi * cx1 - sin_phi * cy1);
    let cy = sy2 + (sin_phi * cx1 + cos_phi * cy1);

    // Step 5: Compute angles
    let ux = (x1 - cx1) / rx;
    let uy = (y1 - cy1) / ry;
    let vx = (-x1 - cx1) / rx;
    let vy = (-y1 - cy1) / ry;

    // Compute angle1
    let n = (ux * ux + uy * uy).sqrt();
    let p = ux;
    let angle1 = if uy < 0.0 { -p.acos() } else { p.acos() } / n.max(1e-10);

    // Compute dtheta
    let n = ((ux * ux + uy * uy) * (vx * vx + vy * vy)).sqrt();
    let p = ux * vx + uy * vy;
    let mut dtheta = (p / n.max(1e-10)).acos();
    if ux * vy - uy * vx < 0.0 {
        dtheta = -dtheta;
    }

    if !sweep_flag && dtheta > 0.0 {
        dtheta -= 2.0 * std::f32::consts::PI;
    } else if sweep_flag && dtheta < 0.0 {
        dtheta += 2.0 * std::f32::consts::PI;
    }

    // Convert arc to Bezier segments
    arc_to_bezier_segments(cx, cy, rx, ry, phi, angle1, angle1 + dtheta)
}

/// Convert a circular/elliptical arc to cubic Bezier segments
fn arc_to_bezier_segments(
    cx: f32, cy: f32,    // center
    rx: f32, ry: f32,    // radii
    phi: f32,            // rotation angle
    theta1: f32,         // start angle
    theta2: f32,         // end angle
) -> Vec<[[f32; 2]; 4]> {
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();
    
    let mut segments = Vec::new();
    let mut start_angle = theta1;
    let total_angle = theta2 - theta1;
    
    // Split large arcs into smaller segments (max 90 degrees each)
    let max_segment_angle = std::f32::consts::PI / 2.0;
    let num_segments = (total_angle.abs() / max_segment_angle).ceil() as i32;
    let segment_angle = total_angle / num_segments as f32;
    
    for _i in 0..num_segments {
        let end_angle = start_angle + segment_angle;
        
        // Create Bezier segment for this arc portion
        let segment = convert_small_arc_to_bezier_segment(
            cx, cy, rx, ry, cos_phi, sin_phi, start_angle, end_angle
        );
        segments.push(segment);
        
        start_angle = end_angle;
    }
    
    segments
}

/// Convert  a small arc (≤ 90 degrees) to a single Bezier segment 
fn convert_small_arc_to_bezier_segment(
    cx: f32, cy: f32,           // center
    rx: f32, ry: f32,           // radii
    cos_phi: f32, sin_phi: f32, // rotation
    theta1: f32, theta2: f32,   // start and end angles
) -> [[f32; 2]; 4] {
    let alpha = (theta2 - theta1).sin() * ((((theta2 - theta1) / 2.0).cos() - 1.0).abs().sqrt()) / 3.0;
    
    let cos_start = theta1.cos();
    let sin_start = theta1.sin();
    let cos_end = theta2.cos();
    let sin_end = theta2.sin();
    
    // Points on the unit circle
    let q1x = cos_start;
    let q1y = sin_start;
    let q2x = cos_end;
    let q2y = sin_end;
    
    // Control points on unit circle
    let q1_ctrl_x = q1x - alpha * sin_start;
    let q1_ctrl_y = q1y + alpha * cos_start;
    let q2_ctrl_x = q2x + alpha * sin_end;
    let q2_ctrl_y = q2y - alpha * cos_end;
    
    // Transform to ellipse and apply rotation
    let transform_point = |x: f32, y: f32| -> [f32; 2] {
        let ex = rx * x;
        let ey = ry * y;
        [
            cx + cos_phi * ex - sin_phi * ey,
            cy + sin_phi * ex + cos_phi * ey,
        ]
    };
    
    let p0 = transform_point(q1x, q1y);
    let p1 = transform_point(q1_ctrl_x, q1_ctrl_y);
    let p2 = transform_point(q2_ctrl_x, q2_ctrl_y);
    let p3 = transform_point(q2x, q2y);
    
    [p0, p1, p2, p3]
}

/// Convert an ellipse element to four Bezier segments
fn ellipse_to_bezier_segments(cx: f32, cy: f32, rx: f32, ry: f32, rotation: f32) -> Vec<[[f32; 2]; 4]> {
    // Magic number for cubic Bezier approximation of a circle
    // This is the distance from circle center to control point for a 90-degree arc
    const KAPPA: f32 = 0.5522847498; // 4/3 * tan(π/8)
    
    let cos_rot = rotation.cos();
    let sin_rot = rotation.sin();
    
    // Control point distances
    let cp_x = rx * KAPPA;
    let cp_y = ry * KAPPA;
    
    // TODO this rotation seems to incorrectly tweak the bounding box
    
    // Define the four 90-degree segments of the ellipse in local coordinates
    let segments_local = [
        // Right to top (0° to 90°)
        [[rx, 0.0], [rx, cp_y], [cp_x, ry], [0.0, ry]],
        // Top to left (90° to 180°) 
        [[0.0, ry], [-cp_x, ry], [-rx, cp_y], [-rx, 0.0]],
        // Left to bottom (180° to 270°)
        [[-rx, 0.0], [-rx, -cp_y], [-cp_x, -ry], [0.0, -ry]],
        // Bottom to right (270° to 360°)
        [[0.0, -ry], [cp_x, -ry], [rx, -cp_y], [rx, 0.0]],
    ];
    
    // Transform each segment: apply rotation and translation
    segments_local
        .iter()
        .map(|segment| {
            segment.map(|[x, y]| {
                [
                    cx + cos_rot * x - sin_rot * y,
                    cy + sin_rot * x + cos_rot * y,
                ]
            })
        })
        .collect()
}

/// Parse rotation angle from SVG transform attribute (simplified parser)
fn parse_rotation_from_transform(transform: &str) -> f32 {
    // Simple regex-free parsing for rotate() transform
    // Looks for pattern like "rotate(45)" or "rotate(45 100 100)"
    if let Some(start) = transform.find("rotate(") {
        let start_idx = start + 7; // length of "rotate("
        if let Some(end_idx) = transform[start_idx..].find(')') {
            let rotate_content = &transform[start_idx..start_idx + end_idx];
            // Split by whitespace and take first value (the angle)
            if let Some(angle_str) = rotate_content.split_whitespace().next() {
                return angle_str.parse::<f32>().unwrap_or(0.0).to_radians();
            }
        }
    }
    0.0
}

#[proc_macro]
pub fn import_svg_paths(input: TokenStream) -> TokenStream {
    let macro_input = parse_macro_input!(input as MacroInput);
    let file_id_suffix = tokens_to_ident_suffix(&macro_input.file_id_tokens);
    let rel_path = macro_input.file_path;

    // Use CARGO_MANIFEST_DIR to resolve paths relative to the caller's crate root
    let cargo_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| {
        panic!("CARGO_MANIFEST_DIR not set. Ensure the macro is called from a crate with a valid Cargo.toml.");
    });
    let full_path = Path::new(&cargo_dir).join(&rel_path);
    let full_path = match full_path.canonicalize() {
        Ok(path) => path,
        Err(e) => panic!(
            "Failed to resolve SVG file path '{}' relative to CARGO_MANIFEST_DIR '{}': {}. Ensure the path is correct relative to your crate's root directory (where Cargo.toml is located). For example, if your SVG file is in 'img/simple-heart.svg' relative to Cargo.toml, use 'img/simple-heart.svg'.",
            rel_path, cargo_dir, e
        ),
    };
    let svg_content = match fs::read_to_string(&full_path) {
        Ok(content) => content,
        Err(e) => panic!(
            "Failed to read SVG file '{}': {}. Ensure the file exists, is readable, and the path is correctly specified relative to your crate's root.",
            full_path.display(), e
        ),
    };

    let doc = roxmltree::Document::parse(&svg_content).expect("Invalid SVG XML");

    let mut paths = Vec::new();
    for node in doc.descendants() {
        if node.tag_name().name() == "path" {
            if let Some(id) = node.attribute("id") {
                eprintln!("process {} id {} ...",node.tag_name().name(), id );
                if let Some(d) = node.attribute("d") {
                    let path_segments: Vec<PathSegment> = PathParser::from(d).collect::<Result<_, _>>().expect("Invalid path data");

                    let mut bezier_segments: Vec<[[f32; 2]; 4]> = Vec::new();
                    let mut current_pos = [0.0f32, 0.0];
                    let mut start_pos = [0.0f32, 0.0];
                    let mut has_start = false;
                    let mut last_cubic_cp2 = [0.0f32, 0.0];
                    let mut last_quad_cp = [0.0f32, 0.0];
                    let mut last_command_was_cubic_curve = false;
                    let mut last_command_was_quad_curve = false;

                    let mut poly_points: Vec<[i32; 2]> = Vec::new();

                    for seg in path_segments {
                        match seg {
                            PathSegment::MoveTo { abs, x, y } => {
                                let nx = x as f32;
                                let ny = y as f32;
                                let new_pos = if abs { [nx, ny] } else { [current_pos[0] + nx, current_pos[1] + ny] };
                                if has_start {
                                    panic!("Multiple subpaths not supported");
                                }
                                start_pos = new_pos;
                                current_pos = new_pos;
                                has_start = true;
                                last_command_was_cubic_curve = false;
                                last_command_was_quad_curve = false;
                            }
                            PathSegment::LineTo { abs, x, y } => {
                                let nx = x as f32;
                                let ny = y as f32;
                                let p3 = if abs { [nx, ny] } else { [current_pos[0] + nx, current_pos[1] + ny] };
                                let p0 = current_pos;
                                let delta = [(p3[0] - p0[0]) / 3.0, (p3[1] - p0[1]) / 3.0];
                                let p1 = [p0[0] + delta[0], p0[1] + delta[1]];
                                let p2 = [p0[0] + 2.0 * delta[0], p0[1] + 2.0 * delta[1]];
                                bezier_segments.push([p0, p1, p2, p3]);
                                current_pos = p3;
                                last_command_was_cubic_curve = false;
                                last_command_was_quad_curve = false;
                            }
                            PathSegment::HorizontalLineTo { abs, x } => {
                                let nx = x as f32;
                                let p3 = if abs { [nx, current_pos[1]] } else { [current_pos[0] + nx, current_pos[1]] };
                                let p0 = current_pos;
                                let delta = [(p3[0] - p0[0]) / 3.0, (p3[1] - p0[1]) / 3.0];
                                let p1 = [p0[0] + delta[0], p0[1] + delta[1]];
                                let p2 = [p0[0] + 2.0 * delta[0], p0[1] + 2.0 * delta[1]];
                                bezier_segments.push([p0, p1, p2, p3]);
                                current_pos = p3;
                                last_command_was_cubic_curve = false;
                                last_command_was_quad_curve = false;
                            }
                            PathSegment::VerticalLineTo { abs, y } => {
                                let ny = y as f32;
                                let p3 = if abs { [current_pos[0], ny] } else { [current_pos[0], current_pos[1] + ny] };
                                let p0 = current_pos;
                                let delta = [(p3[0] - p0[0]) / 3.0, (p3[1] - p0[1]) / 3.0];
                                let p1 = [p0[0] + delta[0], p0[1] + delta[1]];
                                let p2 = [p0[0] + 2.0 * delta[0], p0[1] + 2.0 * delta[1]];
                                bezier_segments.push([p0, p1, p2, p3]);
                                current_pos = p3;
                                last_command_was_cubic_curve = false;
                                last_command_was_quad_curve = false;
                            }
                            PathSegment::CurveTo { abs, x1, y1, x2, y2, x, y } => {
                                let nx1 = x1 as f32;
                                let ny1 = y1 as f32;
                                let nx2 = x2 as f32;
                                let ny2 = y2 as f32;
                                let nx = x as f32;
                                let ny = y as f32;
                                let p1 = if abs { [nx1, ny1] } else { [current_pos[0] + nx1, current_pos[1] + ny1] };
                                let p2 = if abs { [nx2, ny2] } else { [current_pos[0] + nx2, current_pos[1] + ny2] };
                                let p3 = if abs { [nx, ny] } else { [current_pos[0] + nx, current_pos[1] + ny] };
                                let p0 = current_pos;
                                bezier_segments.push([p0, p1, p2, p3]);
                                current_pos = p3;
                                last_cubic_cp2 = p2;
                                last_command_was_cubic_curve = true;
                                last_command_was_quad_curve = false;
                            }
                            PathSegment::SmoothCurveTo { abs, x2, y2, x, y } => {
                                let nx2 = x2 as f32;
                                let ny2 = y2 as f32;
                                let nx = x as f32;
                                let ny = y as f32;
                                let p2 = if abs { [nx2, ny2] } else { [current_pos[0] + nx2, current_pos[1] + ny2] };
                                let p3 = if abs { [nx, ny] } else { [current_pos[0] + nx, current_pos[1] + ny] };
                                let p0 = current_pos;
                                let p1 = if last_command_was_cubic_curve {
                                    [2.0 * p0[0] - last_cubic_cp2[0], 2.0 * p0[1] - last_cubic_cp2[1]]
                                } else {
                                    p0
                                };
                                bezier_segments.push([p0, p1, p2, p3]);
                                current_pos = p3;
                                last_cubic_cp2 = p2;
                                last_command_was_cubic_curve = true;
                                last_command_was_quad_curve = false;
                            }
                            PathSegment::Quadratic { abs, x1, y1, x, y } => {
                                let nx1 = x1 as f32;
                                let ny1 = y1 as f32;
                                let nx = x as f32;
                                let ny = y as f32;
                                let qp1 = if abs { [nx1, ny1] } else { [current_pos[0] + nx1, current_pos[1] + ny1] };
                                let p3 = if abs { [nx, ny] } else { [current_pos[0] + nx, current_pos[1] + ny] };
                                let p0 = current_pos;
                                let cp1 = [p0[0] + (2.0 / 3.0) * (qp1[0] - p0[0]), p0[1] + (2.0 / 3.0) * (qp1[1] - p0[1])];
                                let cp2 = [p3[0] + (2.0 / 3.0) * (qp1[0] - p3[0]), p3[1] + (2.0 / 3.0) * (qp1[1] - p3[1])];
                                bezier_segments.push([p0, cp1, cp2, p3]);
                                current_pos = p3;
                                last_quad_cp = qp1;
                                last_command_was_quad_curve = true;
                                last_command_was_cubic_curve = false;
                            }
                            PathSegment::SmoothQuadratic { abs, x, y } => {
                                let nx = x as f32;
                                let ny = y as f32;
                                let p3 = if abs { [nx, ny] } else { [current_pos[0] + nx, current_pos[1] + ny] };
                                let p0 = current_pos;
                                let qp1 = if last_command_was_quad_curve {
                                    [2.0 * p0[0] - last_quad_cp[0], 2.0 * p0[1] - last_quad_cp[1]]
                                } else {
                                    p0
                                };
                                let cp1 = [p0[0] + (2.0 / 3.0) * (qp1[0] - p0[0]), p0[1] + (2.0 / 3.0) * (qp1[1] - p0[1])];
                                let cp2 = [p3[0] + (2.0 / 3.0) * (qp1[0] - p3[0]), p3[1] + (2.0 / 3.0) * (qp1[1] - p3[1])];
                                bezier_segments.push([p0, cp1, cp2, p3]);
                                current_pos = p3;
                                last_quad_cp = qp1;
                                last_command_was_quad_curve = true;
                                last_command_was_cubic_curve = false;
                            }
                            PathSegment::EllipticalArc { abs, rx, ry, x_axis_rotation, large_arc, sweep, x, y } => {
                                let nx = x as f32;
                                let ny = y as f32;
                                let end_pos = if abs { [nx, ny] } else { [current_pos[0] + nx, current_pos[1] + ny] };
                                assert!(nx < 0., "testing");
                                let arc_segments = elliptical_arc_to_bezier_segments(
                                    current_pos,
                                    end_pos,
                                    rx as f32,
                                    ry as f32,
                                    x_axis_rotation as f32,
                                    large_arc,
                                    sweep,
                                );
                                
                                for segment in arc_segments {
                                    bezier_segments.push(segment);
                                }
                                
                                current_pos = end_pos;
                                last_command_was_cubic_curve = false;
                                last_command_was_quad_curve = false;
                            }
                            PathSegment::ClosePath { .. } => {
                                if current_pos != start_pos {
                                    panic!("Path '{}' not closed properly", id.to_string());
                                }
                                last_command_was_cubic_curve = false;
                                last_command_was_quad_curve = false;
                            }
                        }
                    }

                    bez_segs_to_closed_poly_points(&bezier_segments, &mut poly_points);

                    paths.push((id.to_string(), poly_points));
                }
            }
        }
        else if node.tag_name().name() == "ellipse" {
            if let Some(id) = node.attribute("id") {
                eprintln!("process {} id {} ...",node.tag_name().name(), id );
                // Parse ellipse attributes with defaults
                let cx = node.attribute("cx")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let cy = node.attribute("cy")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let rx = node.attribute("rx")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let ry = node.attribute("ry")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                
                // Handle transform attribute for rotation (simplified - only handles rotate)
                let rotation = if let Some(transform_attr) = node.attribute("transform") {
                    parse_rotation_from_transform(transform_attr)
                } else {
                    0.0
                };
                eprintln!("ellipse rotation: {}", rotation);

                if rx > 0.0 && ry > 0.0 {
                    let bezier_segments = ellipse_to_bezier_segments(cx, cy, rx, ry, rotation);
                    let mut poly_points: Vec<[i32; 2]> = Vec::new();
                    bez_segs_to_closed_poly_points(&bezier_segments, &mut poly_points);
                    paths.push((id.to_string(), poly_points));
                }
            }
}
    }

    let mut static_decls = Vec::new();
    let mut match_data = Vec::new();

    for (index, (id, poly_points)) in paths.into_iter().enumerate() {
        // Include file_id in the static identifier names to avoid collisions
        let points_ident = format_ident!("SVG_FILE_{}_PATH_{}_POINTS", file_id_suffix, index);
        let polyline_ident = format_ident!("SVG_FILE_{}_PATH_{}_POLYLINE", file_id_suffix, index);
        let bbox_ident = format_ident!("SVG_FILE_{}_PATH_{}_BBOX", file_id_suffix, index);
        let scanlines_ident = format_ident!("SVG_FILE_{}_PATH_{}_SCANLS", file_id_suffix, index);
        let polygon_ident = format_ident!("SVG_FILE_{}_PATH_{}_POLYGON", file_id_suffix, index);

        eprintln!("path id: '{}' ", id);

        let points_len = poly_points.len();
        let point_inits: Vec<_> = poly_points.iter().map(|[x, y]| {
            quote! {
                Point::new(#x, #y)
            }
        }).collect();

        // Calculate bounding box at macro expansion time
        let (bbox_x, bbox_y, bbox_width, bbox_height) = calculate_bounding_box(&poly_points);
        let bbox_tuple: BboxTuple = (bbox_x, bbox_y, bbox_width, bbox_height);
        // Calculate the scanline intersections at macro expansion time
        let min_scanline_y: i32 = bbox_y;
        let max_scanline_y: i32 = bbox_y + bbox_height as i32;
        let expected_scanlines = max_scanline_y - min_scanline_y;
        let scanlines:  Vec<Vec<i32>> = scanline_intersections(&poly_points, bbox_tuple);
        // let scanlines: Vec<Vec<i32>> = find_all_scanline_intersections(&poly_points, min_scanline_y, max_scanline_y);
        let found_scanlines:i32 = scanlines.len().try_into().unwrap();
        if expected_scanlines != found_scanlines {
            panic!("Expected {} scanlines, found {}", expected_scanlines, found_scanlines);
        }
        let scanlines_init = vec_to_scanline_intersections_expr(&scanlines);

        let path_name_lit = Literal::string(&id.clone());

        // Generate static declarations for all components
        static_decls.push(quote! {
            // #id.clone()
            #[doc = concat!("SVG path id: ", #path_name_lit)]
            static #points_ident: [Point; #points_len] = [#(#point_inits),*];
            static #polyline_ident: Polyline<'static> = Polyline::new(&#points_ident);
            static #bbox_ident: Rectangle =  Rectangle {
                top_left: Point::new(#bbox_x,#bbox_y),
                size: Size::new(#bbox_width,#bbox_height),
            };
            static #scanlines_ident: ScanlineIntersections<'static> = #scanlines_init;
            static #polygon_ident: ClosedPolygon<'static> = ClosedPolygon {
                // vertices: &#points_ident,
                polyline: #polyline_ident,
                bounding_box: #bbox_ident,
                scanlines: Some(#scanlines_ident),
            };
        });

        match_data.push((id.clone(), polygon_ident.clone()));
    }

    // Generate a function name specific to this file_id
    let get_svg_path_fn = format_ident!("get_svg_path_by_id_file_{}", file_id_suffix);

    // Generate match arms for the function
    let match_arms: Vec<_> = match_data.iter().map(|(id, polygon_ident)| {
        quote! {
            #id => Some(&#polygon_ident)
        }
    }).collect();

    let output = quote! {
        #(#static_decls)*

        pub fn #get_svg_path_fn(path_id: &str) -> Option<&'static ClosedPolygon<'static>> {
            match path_id {
                #(#match_arms,)*
                _ => None,
            }
        }
    };

    output.into()
}