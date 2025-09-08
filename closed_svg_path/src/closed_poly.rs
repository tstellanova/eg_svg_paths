use embedded_graphics::{
    draw_target::DrawTarget,
    geometry::{Dimensions, Point},
    pixelcolor::PixelColor,
    primitives::{
        polyline::{self,Polyline},
        PrimitiveStyle, Rectangle, Styled,
    },
    Drawable, Pixel,
};


/// A closed polygon primitive that extends Polyline functionality.
/// 
/// This struct represents a polygon where the last point is automatically connected
/// to the first point to form a closed shape. It supports both stroke and fill operations
/// using scanline algorithms suitable for embedded environments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosedPolygon<'a> {
    /// The closed polygon vertices (first point matches last point)
    // pub vertices: &'a [Point],
    /// The polyline defined by the closed polygon vertices
    pub polyline: Polyline<'a>,
    /// Cached polygon bounding box
    pub bounding_box: Rectangle,
    /// Maybe cached scanline intersections
    pub scanlines: Option<ScanlineIntersections<'a>>,
}

impl<'a> ClosedPolygon<'a> {
    /// Create a new closed polygon from a slice of points
    /// 
    /// The polygon will automatically be closed by connecting the last point to the first.
    /// Requires at least 3 points to form a valid polygon.
    pub fn new(vertices: &'a [Point]) -> Option<Self> {
        if vertices.len() < 3 {
            return None;
        }

        // Create polyline with original points
        let polyline = Polyline::new(vertices);
        let bounding_box = polyline.bounding_box();

        Some(ClosedPolygon {
            // vertices,
            polyline,
            bounding_box,
            scanlines: None,
        })
    }

    /// Get the vertices of the polygon
    // pub fn vertices(&self) -> &[Point] {
    //     self.vertices
    // }

    pub fn total_size(&self) -> usize {
        // Base size of the struct itself (includes fat pointer + flag + padding)
        let mut total = size_of::<Self>();

        // add the statically allocated size of points
        // total += self.vertices.len() * size_of::<[i32;2]>();

        // Add the sizes of all scanlines
        if let Some(scanlines) = self.scanlines {
            total += scanlines.total_size();
        }

        total
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



/// Variable scanline intersection type
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScanlineIntersections<'a> {
    /// Scanlines containing intersections of the form: [x0, x1, x2...xN] in a jagged array.
    /// The scanline y coordinate is implicit from context, where a polygon
    /// with a bounding box from min_y .. max_y should have at least one scanline
    /// intersection at every y in that range.
    pub data: &'a [&'a [i32]],
}

impl ScanlineIntersections<'_> {
    pub fn total_size(&self) -> usize {
        // Base size of the struct itself (includes fat pointer + flag + padding)
        let mut total = size_of::<Self>();

        // Add the size of the slice backing array (the outer slice of inner slice pointers)
        total += self.data.len() * size_of::<&[i32]>();

        // Add the sizes of all inner arrays
        for inner in self.data.iter() {
            total += inner.len() * size_of::<i32>();
        }

        total
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
            spans_processed: false,
        }
    }

    /// Process current scanline and setup for pixel iteration
    fn process_scanline_spans(&mut self) -> bool {
        assert!(self.polygon.scanlines.is_some(), "migrated to preprocessed scanlines");

        while self.current_y < self.end_y {
            self.intersection_index = 0;
            self.spans_processed = false;
            return self.setup_next_span_new();
        }
        self.current_y += 1;
        false
    }

    /// Setup the next span for pixel iteration.
    /// A span is a contiguous group of pixels on a scanline.
    fn setup_next_span_new(&mut self) -> bool {
        let scan_yidx = (self.current_y - self.polygon.bounding_box.top_left.y) as usize;
        let cur_line_data: &'a [i32]= self.polygon.scanlines.unwrap().data[scan_yidx];
        assert!(cur_line_data.len() >= 2, "scanlines contain at least 2 intersections");

        // we evaulate pairs of intersection points
        while self.intersection_index + 1 < cur_line_data.len() {
            let x_start = cur_line_data[self.intersection_index];
            let x_end = cur_line_data[self.intersection_index+1];
            self.current_x = x_start;
            self.current_span_end = x_end;
            self.intersection_index += 2;
            return true;
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
                if !self.process_scanline_spans() {
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
            if self.setup_next_span_new() {
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
    stroke_iter: Option<polyline::StyledPixelsIterator<'a, C>>,
    fill_iter: Option<PolygonFillIterator<'a, C>>,
    fill_dirty: bool,
    stroke_dirty: bool,
}


impl<'a, C> Iterator for StyledPolygonIterator<'a, C>
where
    C: PixelColor,
{
    type Item = Pixel<C>;

    fn next(&mut self) -> Option<Self::Item> {
        // First draw all fill pixels of closed polygon
        if self.fill_dirty {
            let pixel_opt = self.fill_iter.as_mut().unwrap().next();
            if pixel_opt.is_some() {
                return pixel_opt;
            }
            self.fill_dirty = false;
        }
        
        // Then draw edges of closed polygon
        if self.stroke_dirty {
            let pixel_opt =  self.stroke_iter.as_mut().unwrap().next();
            if pixel_opt.is_some() {
                return pixel_opt;
            }
            self.stroke_dirty = false;
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

impl<'a, C> StyledPolygonIterator<'a, C>
where
    C: PixelColor,
{
    fn new(polygon: &'a StyledClosedPolygon<'a, C>) -> Self {
        let have_stroke =  polygon.style.stroke_color.is_some();
        let have_fill = polygon.style.fill_color.is_some();

        let fill_iter = if let Some(fill_color) = polygon.style.fill_color {
            Some(PolygonFillIterator::new(&polygon.polygon, fill_color))
        } else {
            None
        };

        // Use the existing polyline for the stroke
        let polyline_iter = if have_stroke {
            let styled_polyline = Styled::new(polygon.polygon.polyline.clone(), polygon.style);
            Some(styled_polyline.pixels())
        } else {
            None
        };


        
        Self {
            stroke_iter: polyline_iter,
            fill_iter,
            fill_dirty: have_fill,
            stroke_dirty: have_stroke,
        }
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

}