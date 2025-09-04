use embedded_graphics::{
    draw_target::DrawTarget,
    geometry::{Dimensions, Point},
    pixelcolor::PixelColor,
    primitives::{
        polyline::Polyline,
        Line, PrimitiveStyle, Rectangle, Styled,
    },
    Drawable, Pixel,
};

impl<'a, C> StyledPolygonIterator<'a, C>
where
    C: PixelColor,
{
    fn new(polygon: &'a StyledClosedPolygon<'a, C>) -> Self {
        // Use the existing polyline for the main stroke
        let polyline_iter = if polygon.style.stroke_color.is_some() {
            let styled_polyline = Styled::new(polygon.polygon.polyline.clone(), polygon.style);
            Some(styled_polyline.pixels())
        } else {
            None
        };

        // Create closing line from last point to first point
        let closing_line_iter = if polygon.style.stroke_color.is_some() && polygon.polygon.vertices().len() >= 2 {
            let vertices = polygon.polygon.vertices();
            let last_point = vertices[vertices.len() - 1];
            let first_point = vertices[0];
            
            if last_point != first_point { // Only add closing line if not already closed
                let closing_line = Line::new(last_point, first_point);
                let styled_closing_line = Styled::new(closing_line, polygon.style);
                Some(styled_closing_line.pixels())
            } else {
                None
            }
        } else {
            None
        };

        let fill_iter = if let Some(fill_color) = polygon.style.fill_color {
            Some(PolygonFillIterator::new(&polygon.polygon, fill_color))
        } else {
            None
        };

        let has_closing_stroke = closing_line_iter.is_some();
        
        Self {
            polyline_iter,
            closing_line_iter,
            fill_iter,
            drawing_fill: polygon.style.fill_color.is_some(),
            drawing_polyline_stroke: polygon.style.stroke_color.is_some(),
            drawing_closing_stroke: has_closing_stroke,
        }
    }
}



/// Maximum number of intersections per scanline
const MAX_INTERSECTIONS: usize = 64;

/// A closed polygon primitive that extends Polyline functionality
/// 
/// This struct represents a polygon where the last point is automatically connected
/// to the first point to form a closed shape. It supports both stroke and fill operations
/// using scanline algorithms suitable for embedded environments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosedPolygon<'a> {
    /// The original points that define the polygon vertices
    points: &'a [Point],
    /// The underlying polyline that defines the polygon vertices
    polyline: Polyline<'a>,
    /// Cached bounding box for performance
    bounding_box: Rectangle,
}

impl<'a> ClosedPolygon<'a> {
    /// Create a new closed polygon from a slice of points
    /// 
    /// The polygon will automatically be closed by connecting the last point to the first.
    /// Requires at least 3 points to form a valid polygon.
    pub fn new(points: &'a [Point]) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }

        // Create polyline with original points
        let polyline = Polyline::new(points);
        let bounding_box = polyline.bounding_box();

        Some(ClosedPolygon {
            points,
            polyline,
            bounding_box,
        })
    }

    pub fn new_static(points: &'static [Point]) -> Self {
        // Create polyline with original points
        let polyline = Polyline::new(points);
        let bounding_box = polyline.bounding_box();

        // const MAX_SCANLINES: i32 = 240;
        // let mut all_intersections: [IntersectionBuffer; MAX_SCANLINES as usize] = [IntersectionBuffer::new() ; MAX_SCANLINES as usize];
        // for y in 0..MAX_SCANLINES {
        //     Self::find_scanline_intersections(points, y, &mut all_intersections[y as usize]);
        // }
        ClosedPolygon {
            points,
            polyline,
            bounding_box,
        }
    }

    /// Get the vertices of the polygon
    pub fn vertices(&self) -> &[Point] {
        self.points
    }

    /// Find intersections of a horizontal scanline at y with our polygon edges
    fn find_scanline_intersections(vertices: &[Point], y: i32, scanline_intersections: &mut IntersectionBuffer) {
        scanline_intersections.clear();
        let n = vertices.len();

        for i in 0..n {
            let p1 = vertices[i];
            let p2 = vertices[(i + 1) % n]; // Wrap around to close the polygon

            // Skip horizontal edges
            if p1.y == p2.y {
                continue;
            }

            // Check if scanline intersects this edge
            let min_y = p1.y.min(p2.y);
            let max_y = p1.y.max(p2.y);

            if y >= min_y && y < max_y {
                // Calculate intersection x coordinate
                let x = p1.x + ((y - p1.y) * (p2.x - p1.x)) / (p2.y - p1.y);
                let _ = scanline_intersections.push(x);
            }
        }

        scanline_intersections.sort();
    }
    
    /// Create a styled version of this polygon
    pub fn into_styled<C>(self, style: PrimitiveStyle<C>) -> StyledClosedPolygon<'a, C>
    where
        C: PixelColor,
    {
        StyledClosedPolygon::new(self, style)
    }
}

impl<'a> Dimensions for ClosedPolygon<'a> {
    fn bounding_box(&self) -> Rectangle {
        self.bounding_box
    }
}

/// A styled closed polygon that can be drawn with both stroke and fill
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StyledClosedPolygon<'a, C>
where
    C: PixelColor,
{
    polygon: ClosedPolygon<'a>,
    style: PrimitiveStyle<C>,
}

impl<'a, C> StyledClosedPolygon<'a, C>
where
    C: PixelColor,
{
    /// Create a new styled closed polygon
    pub fn new(polygon: ClosedPolygon<'a>, style: PrimitiveStyle<C>) -> Self {
        Self { polygon, style }
    }
}

impl<'a, C> Dimensions for StyledClosedPolygon<'a, C>
where
    C: PixelColor,
{
    fn bounding_box(&self) -> Rectangle {
        self.polygon.bounding_box()
    }
}

/// Fixed-size intersection buffer for scanline algorithm
#[derive(Copy, Clone, Debug)]
struct IntersectionBuffer {
    intersections: [i32; MAX_INTERSECTIONS],
    count: usize,
}

impl IntersectionBuffer {
    fn new() -> Self {
        Self {
            intersections: [0; MAX_INTERSECTIONS],
            count: 0,
        }
    }

    fn clear(&mut self) {
        self.count = 0;
    }

    fn push(&mut self, x: i32) -> bool {
        if self.count < MAX_INTERSECTIONS {
            self.intersections[self.count] = x;
            self.count += 1;
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Option<i32> {
        if index < self.count {
            Some(self.intersections[index])
        } else {
            None
        }
    }

    /// Sort intersections using bubble sort (no_std compatible)
    fn sort(&mut self) {
        for i in 0..self.count {
            for j in 0..self.count.saturating_sub(1).saturating_sub(i) {
                if self.intersections[j] > self.intersections[j + 1] {
                    self.intersections.swap(j, j + 1);
                }
            }
        }
    }
}

/// Iterator for polygon fill pixels using scanline algorithm
pub struct PolygonFillIterator<'a, C>
where
    C: PixelColor,
{
    polygon: &'a ClosedPolygon<'a>,
    color: C,
    current_y: i32,
    end_y: i32,
    current_x: i32,
    current_span_end: i32,
    intersection_index: usize,
    intersections: IntersectionBuffer,
    spans_processed: bool,
}

impl<'a, C> PolygonFillIterator<'a, C>
where
    C: PixelColor,
{
    fn new(polygon: &'a ClosedPolygon<'a>, color: C) -> Self {
        let bbox = polygon.bounding_box();
        Self {
            polygon,
            color,
            current_y: bbox.top_left.y,
            end_y: bbox.top_left.y + bbox.size.height as i32,
            current_x: 0,
            current_span_end: 0,
            intersection_index: 0,
            intersections: IntersectionBuffer::new(),
            spans_processed: false,
        }
    }

    /// Find intersections of a horizontal line at y with polygon edges
    fn find_intersections(&mut self, y: i32) {
        self.intersections.clear();
        let vertices = self.polygon.vertices();
        let n = vertices.len();

        for i in 0..n {
            let p1 = vertices[i];
            let p2 = vertices[(i + 1) % n]; // Wrap around to close the polygon

            // Skip horizontal edges
            if p1.y == p2.y {
                continue;
            }

            // Check if scanline intersects this edge
            let min_y = p1.y.min(p2.y);
            let max_y = p1.y.max(p2.y);

            if y >= min_y && y < max_y {
                // Calculate intersection x coordinate
                let x = p1.x + ((y - p1.y) * (p2.x - p1.x)) / (p2.y - p1.y);
                let _ = self.intersections.push(x);
            }
        }

        self.intersections.sort();
    }

    /// Process current scanline and setup for pixel iteration
    fn process_scanline(&mut self) -> bool {
        while self.current_y < self.end_y {
            self.find_intersections(self.current_y);
            
            if self.intersections.len() >= 2 {
                self.intersection_index = 0;
                self.spans_processed = false;
                return self.setup_next_span();
            }
            
            self.current_y += 1;
        }
        false
    }

    /// Setup the next span for pixel iteration
    fn setup_next_span(&mut self) -> bool {
        while self.intersection_index + 1 < self.intersections.len() {
            if let (Some(x_start), Some(x_end)) = (
                self.intersections.get(self.intersection_index),
                self.intersections.get(self.intersection_index + 1),
            ) {
                self.current_x = x_start;
                self.current_span_end = x_end;
                self.intersection_index += 2;
                return true;
            }
            self.intersection_index += 2;
        }
        false
    }
}

impl<'a, C> Iterator for PolygonFillIterator<'a, C>
where
    C: PixelColor,
{
    type Item = Pixel<C>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we haven't processed spans for current scanline, do it now
            if !self.spans_processed {
                if !self.process_scanline() {
                    return None;
                }
                self.spans_processed = true;
            }

            // If we're within the current span, return a pixel
            if self.current_x <= self.current_span_end {
                let pixel = Pixel(Point::new(self.current_x, self.current_y), self.color);
                self.current_x += 1;
                return Some(pixel);
            }

            // Try to setup the next span
            if self.setup_next_span() {
                continue;
            }

            // Move to next scanline
            self.current_y += 1;
            self.spans_processed = false;
            
            if self.current_y >= self.end_y {
                return None;
            }
        }
    }
}

/// Iterator that combines stroke and fill pixels with proper layering
pub struct StyledPolygonIterator<'a, C>
where
    C: PixelColor,
{
    polyline_iter: Option<embedded_graphics::primitives::polyline::StyledPixelsIterator<'a, C>>,
    closing_line_iter: Option<embedded_graphics::primitives::line::StyledPixelsIterator<C>>,
    fill_iter: Option<PolygonFillIterator<'a, C>>,
    drawing_fill: bool,
    drawing_polyline_stroke: bool,
    drawing_closing_stroke: bool,
}


impl<'a, C> Iterator for StyledPolygonIterator<'a, C>
where
    C: PixelColor,
{
    type Item = Pixel<C>;

    fn next(&mut self) -> Option<Self::Item> {
        // First draw all fill pixels (background layer)
        if self.drawing_fill {
            if let Some(ref mut fill_iter) = self.fill_iter {
                if let Some(pixel) = fill_iter.next() {
                    return Some(pixel);
                }
            }
            self.drawing_fill = false;
        }

        // Then draw polyline stroke pixels (main edges of polygon)
        if self.drawing_polyline_stroke {
            if let Some(ref mut polyline_iter) = self.polyline_iter {
                if let Some(pixel) = polyline_iter.next() {
                    return Some(pixel);
                }
            }
            self.drawing_polyline_stroke = false;
        }

        // Finally draw closing line stroke pixels (edge from last to first point)
        if self.drawing_closing_stroke {
            if let Some(ref mut closing_line_iter) = self.closing_line_iter {
                if let Some(pixel) = closing_line_iter.next() {
                    return Some(pixel);
                }
            }
            self.drawing_closing_stroke = false;
        }

        None
    }
}

impl<'a, C> Drawable for StyledClosedPolygon<'a, C>
where
    C: PixelColor,
{
    type Color = C;
    type Output = ();

    fn draw<D>(&self, target: &mut D) -> Result<Self::Output, D::Error>
    where
        D: DrawTarget<Color = Self::Color>,
    {
        target.draw_iter(StyledPolygonIterator::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_graphics::{
        mock_display::MockDisplay,
        pixelcolor::BinaryColor,
        prelude::*,
    };

    #[test]
    fn test_polygon_creation() {
        let points = [
            Point::new(10, 10),
            Point::new(20, 10),
            Point::new(15, 20),
        ];
        
        let polygon = ClosedPolygon::new(&points);
        assert!(polygon.is_some());
        
        let polygon = polygon.unwrap();
        assert_eq!(polygon.vertices().len(), 3);
    }

    #[test]
    fn test_polygon_bounding_box() {
        let points = [
            Point::new(10, 5),
            Point::new(20, 15),
            Point::new(5, 20),
        ];
        
        let polygon = ClosedPolygon::new(&points).unwrap();
        let bbox = polygon.bounding_box();
        
        assert_eq!(bbox.top_left, Point::new(5, 5));
        assert_eq!(bbox.size, Size::new(16, 16));
    }

    #[test]
    fn test_styled_polygon_drawing() {
        let points = [
            Point::new(5, 5),
            Point::new(15, 5),
            Point::new(10, 15),
        ];
        
        let polygon = ClosedPolygon::new(&points).unwrap();
        let styled = polygon.into_styled(
            PrimitiveStyle::with_fill(BinaryColor::On)
        );

        let mut display = MockDisplay::new();
        styled.draw(&mut display).unwrap();
        
        // The triangle should have some filled pixels
        assert!(display.affected_area().size.width > 0 && display.affected_area().size.height > 0);
    }

    #[test]
    fn test_invalid_polygon() {
        let points = [Point::new(10, 10), Point::new(20, 20)]; // Only 2 points
        let polygon = ClosedPolygon::new(&points);
        assert!(polygon.is_none());
    }

    #[test]
    fn test_intersection_buffer() {
        let mut buffer = IntersectionBuffer::new();
        
        assert!(buffer.push(10));
        assert!(buffer.push(5));
        assert!(buffer.push(15));
        
        buffer.sort();
        
        assert_eq!(buffer.get(0), Some(5));
        assert_eq!(buffer.get(1), Some(10));
        assert_eq!(buffer.get(2), Some(15));
        assert_eq!(buffer.len(), 3);
    }
}