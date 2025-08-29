#![no_std]

use num_traits::Float;

use embedded_graphics::{
    geometry::{Dimensions, OriginDimensions, Point, Size},
    primitives::Rectangle,
};

pub mod points;
pub mod styled;

pub use points::*;
pub use styled::*;

#[derive(Copy, Clone, Debug)]
pub struct BezierSegment(pub [[f32; 2]; 4]);

impl BezierSegment {
    /// Evaluate the Bezier curve at parameter t (0.0 to 1.0)
    pub fn evaluate(&self, t: f32) -> [f32; 2] {
        let [p0, p1, p2, p3] = self.0;
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        
        [
            mt3 * p0[0] + 3.0 * mt2 * t * p1[0] + 3.0 * mt * t2 * p2[0] + t3 * p3[0],
            mt3 * p0[1] + 3.0 * mt2 * t * p1[1] + 3.0 * mt * t2 * p2[1] + t3 * p3[1],
        ]
    }
    
    /// Get the derivative (tangent vector) at parameter t
    pub fn derivative(&self, t: f32) -> [f32; 2] {
        let [p0, p1, p2, p3] = self.0;
        let t2 = t * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        
        [
            3.0 * (-mt2 * p0[0] + mt2 * p1[0] - 2.0 * mt * t * p1[0] + 2.0 * mt * t * p2[0] - t2 * p2[0] + t2 * p3[0]),
            3.0 * (-mt2 * p0[1] + mt2 * p1[1] - 2.0 * mt * t * p1[1] + 2.0 * mt * t * p2[1] - t2 * p2[1] + t2 * p3[1]),
        ]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ClosedCubicBezierPath {
    pub bezier_segments: &'static [BezierSegment],
    pub bounding_box: Rectangle,
    /// Number of subdivisions per segment for approximation
    pub subdivision_count: u16,
}

impl ClosedCubicBezierPath {
    pub fn new(bezier_segments: &'static [BezierSegment], subdivision_count: u16) -> Self {
        let bounding_box = Self::calculate_bounding_box(bezier_segments);
        Self {
            bezier_segments,
            bounding_box,
            subdivision_count,
        }
    }
    
    pub fn calculate_bounding_box(segments: &[BezierSegment]) -> Rectangle {
        if segments.is_empty() {
            return Rectangle::new(Point::new(0, 0), Size::new(0, 0));
        }
        
        let mut min_x: f32 = f32::INFINITY;
        let mut max_x: f32 = f32::NEG_INFINITY;
        let mut min_y: f32 = f32::INFINITY;
        let mut max_y: f32 = f32::NEG_INFINITY;
        
        // Sample each segment to find bounds
        for segment in segments {
            for i in 0..=32 {
                let t = i as f32 / 32.0;
                let [x, y] = segment.evaluate(t);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        
        let top_left = Point::new(
            f32::floor(min_x) as i32,
            f32::floor(min_y) as i32,
        );
        let bottom_right = Point::new(
            f32::ceil(max_x) as i32,
            f32::ceil(max_y) as i32,
        );
        
        Rectangle::new(
            top_left,
            Size::new(
                (bottom_right.x - top_left.x) as u32,
                (bottom_right.y - top_left.y) as u32,
            ),
        )
    }
    
    /// Check if a point is inside the closed path using ray casting
    pub fn contains_point(&self, point: Point) -> bool {
        let px = point.x as f32;
        let py = point.y as f32;
        let mut inside = false;
        
        for segment in self.bezier_segments {
            let steps = self.subdivision_count as usize;
            let mut prev_point = segment.evaluate(0.0);
            
            for i in 1..=steps {
                let t = i as f32 / steps as f32;
                let curr_point = segment.evaluate(t);
                
                // Ray casting algorithm
                if ((prev_point[1] > py) != (curr_point[1] > py)) &&
                   (px < (curr_point[0] - prev_point[0]) * (py - prev_point[1]) / (curr_point[1] - prev_point[1]) + prev_point[0]) {
                    inside = !inside;
                }
                
                prev_point = curr_point;
            }
        }
        
        inside
    }
    
    /// Get the minimum distance from a point to the curve outline
    pub fn distance_to_outline(&self, point: Point) -> f32 {
        let px = point.x as f32;
        let py = point.y as f32;
        let mut min_dist = f32::INFINITY;
        
        for segment in self.bezier_segments {
            let steps = self.subdivision_count as usize;
            let mut prev_point = segment.evaluate(0.0);
            
            for i in 1..=steps {
                let t = i as f32 / steps as f32;
                let curr_point = segment.evaluate(t);
                
                // Distance to line segment
                let dx = curr_point[0] - prev_point[0];
                let dy = curr_point[1] - prev_point[1];
                let len_sq = dx * dx + dy * dy;
                
                if len_sq > 1e-6 {
                    let t = ((px - prev_point[0]) * dx + (py - prev_point[1]) * dy) / len_sq;
                    let t = t.max(0.0).min(1.0);
                    let proj_x = prev_point[0] + t * dx;
                    let proj_y = prev_point[1] + t * dy;
                    let dist_sq = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y);
                    min_dist = min_dist.min(f32::sqrt(dist_sq));
                }
                
                prev_point = curr_point;
            }
        }
        
        min_dist
    }
}

// Implement required traits for embedded-graphics integration
impl OriginDimensions for ClosedCubicBezierPath {
    fn size(&self) -> Size {
        self.bounding_box.size
    }
}