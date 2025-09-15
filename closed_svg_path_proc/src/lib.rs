use proc_macro2::Literal;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;

use quote::{quote, format_ident};
use syn::{parse_macro_input, LitStr, Token};
use std::env;
use std::fs;
use std::path::Path;
use std::f32::consts::PI;
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




/// Converts an ellipse to 4 cubic Bezier segments after applying SVG rotation
/// rotation_tuple = (angle_rad, Option<[x, y]>)
fn ellipse_to_bezier_segments(
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    rotation_tuple: (f32, Option<[f32; 2]>),
) -> Vec<[[f32; 2]; 4]> {
    let (angle, center) = rotation_tuple;

    // Precompute constants for cubic Bezier approximation
    // https://spencermortensen.com/articles/bezier-circle/
    let kappa = 4.0 * (PI / 8.0).tan() / 3.0; // 4/3 * tan(π/8) = ~0.5522847 

    // Four Bezier segments, defined by points relative to ellipse center
    let mut segments = vec![
        // Each segment: [start, control1, control2, end]
        [[-rx, 0.0], [-rx, kappa * ry], [-kappa * rx, ry], [0.0, ry]],
        [[0.0, ry], [kappa * rx, ry], [rx, kappa * ry], [rx, 0.0]],
        [[rx, 0.0], [rx, -kappa * ry], [kappa * rx, -ry], [0.0, -ry]],
        [[0.0, -ry], [-kappa * rx, -ry], [-rx, -kappa * ry], [-rx, 0.0]],
    ];

    // Rotation matrix components
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Determine rotation center
    let (ox, oy) = if let Some(origin) = center {
        (origin[0], origin[1])
    }
    else {
        (0.,0.)
    };

    // Apply rotation and translation for each point
    for segment in segments.iter_mut() {
        for point in segment.iter_mut() {
            // Translate point by ellipse (cx,cy) then relative to rotation center
            let x_rel = (point[0] + cx) - ox;
            let y_rel = (point[1] + cy) - oy;

            // Rotate
            let x_rot = x_rel * cos_a - y_rel * sin_a;
            let y_rot = x_rel * sin_a + y_rel * cos_a;

            // Translate back by rotation center
            point[0] = x_rot + ox;
            point[1] = y_rot + oy;
        }
    }

    segments
}

/// Parse rotation angle from SVG transform attribute (simplified parser)
fn parse_rotation_from_transform(transform: &str) -> (f32, Option<[f32; 2]>) {
    // Remove whitespace and ensure it's a rotate() string
    let transform = transform.trim();
    if !transform.starts_with("rotate(") || !transform.ends_with(')') {
        return (0.0, None);
    }

    // Extract the content inside the parentheses
    let content = &transform[7..transform.len() - 1];
    // Split by whitespace or commas
    let parts: Vec<&str> = content
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();

    if parts.is_empty() {
        return (0.0, None);
    }

    // Parse the rotation angle (in degrees -> radians)
    let angle_deg: f32 = parts[0].parse().unwrap_or(0.0);
    let angle_rad = angle_deg.to_radians();

    // Parse optional center point
    let center = if parts.len() >= 3 {
        let x: f32 = parts[1].parse().unwrap_or(0.0);
        let y: f32 = parts[2].parse().unwrap_or(0.0);
        Some([x, y])
    } else {
        None
    };

    (angle_rad, center)
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
                    let mut saw_seg_start = false;
                    let mut last_cubic_cp2 = [0.0f32, 0.0];
                    let mut last_quad_cp = [0.0f32, 0.0];
                    let mut last_command_was_cubic_curve = false;
                    let mut last_command_was_quad_curve = false;

                    let mut poly_points: Vec<[i32; 2]> = Vec::new();

                    for seg_idx in 0..path_segments.len() {
                        let seg = path_segments[seg_idx];
                        match seg {
                            PathSegment::MoveTo { abs, x, y } => {
                                let nx = x as f32;
                                let ny = y as f32;
                                let new_pos = if abs { [nx, ny] } else { [current_pos[0] + nx, current_pos[1] + ny] };
                                assert!(!saw_seg_start,"Multiple subpaths not supported");

                                current_pos = [new_pos[0].round(), new_pos[1].round()];
                                start_pos = current_pos;
                                saw_seg_start = true;
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
                                let p3: [f32; 2] = [p3[0].round(), p3[1].round()];
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
                                let p3: [f32; 2] = [p3[0].round(), p3[1].round()];
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
                                let p3: [f32; 2] = [p3[0].round(), p3[1].round()];
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
                                let p3: [f32; 2] = [p3[0].round(), p3[1].round()];
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
                                let p3: [f32; 2] = [p3[0].round(), p3[1].round()];
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
                                let p3: [f32; 2] = [p3[0].round(), p3[1].round()];
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
                                let p3: [f32; 2] = [p3[0].round(), p3[1].round()];
                                bezier_segments.push([p0, cp1, cp2, p3]);
                                current_pos = p3;
                                last_quad_cp = qp1;
                                last_command_was_quad_curve = true;
                                last_command_was_cubic_curve = false;
                            }
                            PathSegment::EllipticalArc { .. } => {
                                panic!("EllipticalArc path segments unsupported!");
                            }
                            PathSegment::ClosePath { .. } => {
                                let (final_x, final_y) = (current_pos[0], current_pos[1]);
                                let (first_x , first_y) = (start_pos[0], start_pos[1]);
                                if (final_x, final_y) != (first_x , first_y) {
                                    eprintln!("Path '{}' not closed properly cur: {:?} start: {:?}", id.to_string(), (final_x, final_y), (first_x , first_y));
                                    
                                }
                                last_command_was_cubic_curve = false;
                                last_command_was_quad_curve = false;
                            }
                        }
                    }

                    bez_segs_to_closed_poly_points(&bezier_segments, &mut poly_points);
                    // ensure that polygons are closed
                    let first_pt = poly_points.first().unwrap().clone();
                    let last_pt = poly_points.last().unwrap().clone();
                    let last_idx = poly_points.len() - 1;
                    if last_pt != first_pt {
                        eprintln!("polygon not closed properly: ({},{}) != ({},{})", 
                            first_pt[0],first_pt[1],last_pt[0],last_pt[1]);
                        poly_points.remove(last_idx);
                        poly_points.push(first_pt);
                    }
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
                    (0.0, None)
                };
                eprintln!("ellipse rotation: {:?}", rotation);

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

        // eprintln!("path id: '{}' ", id);

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