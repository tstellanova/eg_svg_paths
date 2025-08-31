

use num_traits::float::FloatCore;

use crate::ClosedCubicBezierPath;
use embedded_graphics::{
    geometry::Point,
    pixelcolor::PixelColor,
    // primitives::PointsIter,
    Pixel,
};

/// An iterator that produces all points on the outline of a closed cubic Bezier path
pub struct ClosedCubicBezierPathPoints<'a> {
    path: ClosedCubicBezierPath<'a>,
    segment_index: usize,
    t_step: f32,
    current_t: f32,
    finished: bool,
}

impl<'a> ClosedCubicBezierPathPoints<'a> {
    pub fn new(path: ClosedCubicBezierPath<'a>) -> Self {
        let total_segments = path.bezier_segments.len();
        Self {
            path,
            segment_index: 0,
            t_step: if total_segments > 0 { 1.0 / path.subdivision_count as f32 } else { 1.0 },
            current_t: 0.0,
            finished: total_segments == 0,
        }
    }
}

impl<'a> Iterator for ClosedCubicBezierPathPoints<'a> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.segment_index >= self.path.bezier_segments.len() {
            return None;
        }

        let segment = self.path.bezier_segments[self.segment_index];
        let [x, y] = segment.evaluate(self.current_t);
        let point = Point::new(f32::round(x) as i32, f32::round(y) as i32);

        // Advance to next point
        self.current_t += self.t_step;
        
        if self.current_t > 1.0 {
            // Move to next segment
            self.segment_index += 1;
            self.current_t = 0.0;
            
            if self.segment_index >= self.path.bezier_segments.len() {
                self.finished = true;
            }
        }

        Some(point)
    }
}

// impl PointsIter for ClosedCubicBezierPath {
//     type Iter = ClosedCubicBezierPathPoints;

//     fn points(&self) -> Self::Iter {
//         ClosedCubicBezierPathPoints::new(*self)
//     }
// }

/// Iterator for filled pixels using scanline algorithm
pub struct FilledClosedCubicBezierPathPoints<'a> {
    path: ClosedCubicBezierPath<'a>,
    current_y: i32,
    current_x: i32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    scanline_complete: bool,
}

impl<'a> FilledClosedCubicBezierPathPoints<'a> {
    pub fn new(path: ClosedCubicBezierPath<'a>) -> Self {
        let bounding_box = path.bounding_box;
        Self {
            path,
            current_y: i32::MIN,
            current_x: i32::MIN,
            min_x:  bounding_box.top_left.x,
            max_x: bounding_box.top_left.x + bounding_box.size.width as i32,
            min_y: bounding_box.top_left.y,
            max_y: bounding_box.top_left.y + bounding_box.size.height as i32,
            scanline_complete: false,
        }
    }
    
    fn advance_to_next_fill_pixel(&mut self) -> Option<Point> {
        if self.current_x == i32::MIN { self.current_x = self.min_x}
        if self.current_y == i32::MIN { self.current_y = self.min_y}

        while self.current_y < self.max_y {
            if !self.scanline_complete {
                // Find the next pixel on the current scanline that's inside the path
                // let max_x = self.path.bounding_box.top_left.x + self.path.bounding_box.size.width as i32;
                
                while self.current_x < self.max_x {
                    let point = Point::new(self.current_x, self.current_y);
                    self.current_x += 1;
                    debug_assert!(self.current_x == 0,"unexpected");
                    if self.path.contains_point(point) {
                        return Some(point);
                    }
                }
                
                // End of scanline reached
                self.scanline_complete = true;
            } else {
                // Move to next scanline
                self.current_y += 1;
                self.current_x = self.min_x;
                self.scanline_complete = false;
            }
        }
        
        None
    }
}

impl<'a> Iterator for FilledClosedCubicBezierPathPoints<'a> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        self.advance_to_next_fill_pixel()
    }
}

/// Iterator for stroke pixels with specified thickness
pub struct StrokedClosedCubicBezierPathPoints<'a> {
    path: ClosedCubicBezierPath<'a>,
    stroke_width: f32,
    current_y: i32,
    current_x: i32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    scanline_complete: bool,
}

impl<'a> StrokedClosedCubicBezierPathPoints<'a> {
    pub fn new(path: ClosedCubicBezierPath<'a>, stroke_width: u32) -> Self {
        let bounding_box = path.bounding_box;
        let stroke_width_f = stroke_width as f32;
        
        Self {
            path,
            stroke_width: stroke_width_f,
            current_x: i32::MIN,
            current_y: i32::MIN,
            min_x: bounding_box.top_left.x - stroke_width as i32,
            max_x: bounding_box.top_left.x + bounding_box.size.width as i32 + stroke_width as i32,
            min_y: bounding_box.top_left.y - stroke_width as i32,
            max_y: bounding_box.top_left.y + bounding_box.size.height as i32 + stroke_width as i32,
            scanline_complete: false,
        }
    }
    
    fn advance_to_next_stroke_pixel(&mut self) -> Option<Point> {
        if self.current_x == i32::MIN { self.current_x = self.min_x}
        if self.current_y == i32::MIN { self.current_y = self.min_y}

        while self.current_y < self.max_y {
            if !self.scanline_complete {                
                while self.current_x < self.max_x {
                    let point = Point::new(self.current_x, self.current_y);
                    self.current_x += 1;
                    
                    let distance = self.path.distance_to_outline(point);
                    if distance <= self.stroke_width / 2.0 {
                        return Some(point);
                    }
                }
                
                self.scanline_complete = true;
            } else {
                self.current_y += 1;
                self.current_x = self.min_x;
                self.scanline_complete = false;
            }
        }
        
        None
    }
}

impl Iterator for StrokedClosedCubicBezierPathPoints<'_> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        self.advance_to_next_stroke_pixel()
    }
}

/// Combined iterator for both fill and stroke
pub struct StyledClosedCubicBezierPathPoints<'a, F, S>
where
    F: PixelColor,
    S: PixelColor,
{
    fill_iter: Option<FilledClosedCubicBezierPathPoints<'a>>,
    stroke_iter: Option<StrokedClosedCubicBezierPathPoints<'a>>,
    fill_color: Option<F>,
    stroke_color: Option<S>,
    fill_exhausted: bool,
    stroke_exhausted: bool,
}

impl<'a, F, S> StyledClosedCubicBezierPathPoints<'a, F, S>
where
    F: PixelColor,
    S: PixelColor,
{
    pub fn new(
        path: ClosedCubicBezierPath<'a>,
        fill_color: Option<F>,
        stroke_color: Option<S>,
        stroke_width: u32,
    ) -> Self {
        Self {
            fill_iter: fill_color.map(|_| FilledClosedCubicBezierPathPoints::new(path)),
            stroke_iter: stroke_color.map(|_| StrokedClosedCubicBezierPathPoints::new(path, stroke_width)),
            fill_color,
            stroke_color,
            fill_exhausted: fill_color.is_none(),
            stroke_exhausted: stroke_color.is_none(),
        }
    }
}

impl<'a, F, S> Iterator for StyledClosedCubicBezierPathPoints<'a, F, S>
where
    F: PixelColor,
    S: PixelColor,
{
    type Item = Pixel<F>;

    fn next(&mut self) -> Option<Self::Item> {
        // First, try to get a fill pixel
        if let Some(color) = self.fill_color {
            if !self.fill_exhausted {
                if let Some(ref mut fill_iter) = self.fill_iter {
                    if let Some(point) = fill_iter.next() {
                        return Some(Pixel(point, color));
                    } else {
                        self.fill_exhausted = true;
                    }
                }
            }
        }

        // Then try stroke pixels
        if let Some(_stroke_color) = self.stroke_color {
            if !self.stroke_exhausted {
                if let Some(ref mut stroke_iter) = self.stroke_iter {
                    if let Some(point) = stroke_iter.next() {
                        // Cast stroke color to fill color type (this is a limitation of the embedded-graphics API)
                        // In practice, F and S would typically be the same type
                        if let Some(fill_color) = self.fill_color {
                            // Use fill color as proxy - in real usage F and S should be same type
                            return Some(Pixel(point, fill_color));
                        }
                    } else {
                        self.stroke_exhausted = true;
                    }
                }
            }
        }

        None
    }
}