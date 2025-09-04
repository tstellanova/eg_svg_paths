use closed_svg_path::ClosedPolygon;
use proc_macro::TokenStream;
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

/// Select the point at the start of the segment, at round it to the closest integer value
fn round_segment_start(seg: BezierSegment) -> [i32; 2]
{
    [seg.0[0][0].round() as i32, seg.0[0][1].round() as i32]
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
                            PathSegment::EllipticalArc { .. } => {
                                println!("Elliptical arc '{}' not supported",id.to_string());
                                continue;
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

                    for seg in &bezier_segments {
                        // simple approximation: split each bezier segment into eight segments
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

                    // force-close the polygon
                    if let Some(first_poly_pt) = poly_points.first() {
                        poly_points.push(*first_poly_pt);
                    }

                    // TODO calculate scanline intersections?
                    paths.push((id.to_string(), poly_points));
                }
            }
        }
    }

    let mut static_decls = Vec::new();
    let mut match_data = Vec::new();

    for (index, (id, polys)) in paths.into_iter().enumerate() {
        // Include file_id in the static identifier names to avoid collisions
        let poly_ident = format_ident!("SVG_FILE_{}_PATH_{}_POLY_POINTS", file_id_suffix, index);

        println!("path id: '{}' ", id);

        let poly_len = polys.len();
        let poly_inits: Vec<_> = polys.iter().map(|[x, y]| {
            quote! {
                Point::new(#x, #y)
            }
        }).collect();

        // TODO calculate intersections here??

        static_decls.push(quote! {
            static #poly_ident: [Point; #poly_len] = [#(#poly_inits),*];
        });

        match_data.push((id.clone(), poly_ident.clone()));
    }

    // Generate a function name specific to this file_id
    let get_svg_path_fn = format_ident!("get_svg_path_by_id_file_{}", file_id_suffix);

    // Generate match arms for the function
    let match_arms: Vec<_> = match_data.iter().map(|(id, poly_ident)| {
        quote! {
            #id => ClosedPolygon::new(&#poly_ident[..])
        }
    }).collect();

    let output = quote! {
        #(#static_decls)*

        pub fn #get_svg_path_fn(path_id: &str) -> Option<ClosedPolygon> {
            match path_id {
                #(#match_arms,)*
                _ => None,
            }
        }
    };

    output.into()
}