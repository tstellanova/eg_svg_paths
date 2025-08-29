use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, LitStr};
use std::env;
use std::fs;
use std::path::{Path};
use svgtypes::{PathParser, PathSegment};
use closed_svg_path::{BezierSegment, ClosedCubicBezierPath};

#[proc_macro]
pub fn svg_paths(input: TokenStream) -> TokenStream {
    let rel_path = parse_macro_input!(input as LitStr).value();
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
                            PathSegment::EllipticalArc { .. } => {
                                panic!("Elliptical arcs not supported");
                            }
                            PathSegment::ClosePath { .. } => {
                                if current_pos != start_pos {
                                    panic!("Path not closed properly");
                                }
                                last_command_was_cubic_curve = false;
                                last_command_was_quad_curve = false;
                            }
                        }
                    }

                    let mut poly_points: Vec<[i32; 2]> = Vec::new();
                    if let Some(first) = bezier_segments.first() {
                        poly_points.push([first[0][0] as i32, first[0][1] as i32]);
                    }
                    for seg in &bezier_segments {
                        let mid_x = (seg[0][0] + seg[3][0]) / 2.0;
                        let mid_y = (seg[0][1] + seg[3][1]) / 2.0;
                        poly_points.push([mid_x as i32, mid_y as i32]);
                        poly_points.push([seg[3][0] as i32, seg[3][1] as i32]);
                    }

                    let segments_array = bezier_segments.clone()
                        .into_iter()
                        .map(BezierSegment)
                        .collect::<Vec<_>>();

                    let path_bbox = ClosedCubicBezierPath::calculate_bounding_box(&segments_array);
                    paths.push((id.to_string(), bezier_segments, path_bbox, poly_points));
                }
            }
        }
    }


    let mut static_decls = Vec::new();
    let mut match_arms = Vec::new();

    for (index, (id, segs, bbox, polys)) in paths.into_iter().enumerate() {
        let seg_ident = format_ident!("SVG_PATH_{}_BEZIER_SEGMENTS", index);
        let poly_ident = format_ident!("SVG_PATH_{}_POLY_POINTS", index);

        println!("bbox: {:?}", bbox);
        let seg_len = segs.len();
        let seg_inits: Vec<_> = segs.iter().map(|[p0, p1, p2, p3]| {
            let p0x = p0[0];
            let p0y = p0[1];
            let p1x = p1[0];
            let p1y = p1[1];
            let p2x = p2[0];
            let p2y = p2[1];
            let p3x = p3[0];
            let p3y = p3[1];
            quote! {
                BezierSegment([[#p0x, #p0y], [#p1x, #p1y], [#p2x, #p2y], [#p3x, #p3y]])
            }
        }).collect();

        static_decls.push(quote! {
            static #seg_ident: [BezierSegment; #seg_len] = [#(#seg_inits),*];
        });

        let poly_len = polys.len();
        let poly_inits: Vec<_> = polys.iter().map(|[x, y]| {
            quote! {
                Point::new(#x, #y)
            }
        }).collect();

        static_decls.push(quote! {
            static #poly_ident: [Point; #poly_len] = [#(#poly_inits),*];
        });

         // Output the Rectangle as a struct expression
        let x = bbox.top_left.x;
        let y = bbox.top_left.y;
        let w = bbox.size.width;
        let h = bbox.size.height;

        let expanded_bbox = quote! {
            Rectangle::new(
                Point::new(#x, #y),
                Size::new(#w, #h),
            )
        };

        match_arms.push(quote! {
            #id => Some(ClosedCubicBezierPath {
                bezier_segments: &#seg_ident[..],
                bounding_box: #expanded_bbox,
                subdivision_count: 8,
                // polyline_approx: Polyline::new(&#poly_ident[..]),
            })
        });
    }

    let output = quote! {
        #(#static_decls)*

        pub fn get_path_by_id(key: &'static str) -> Option<ClosedCubicBezierPath> {
            match key {
                #(#match_arms,)*
                _ => None,
            }
        }
    };

    output.into()
}