use crate::{ClosedCubicBezierPath, FilledClosedCubicBezierPathPoints, StrokedClosedCubicBezierPathPoints};
use embedded_graphics::{
    draw_target::DrawTarget,
    geometry::{Dimensions, OriginDimensions, Point},
    pixelcolor::PixelColor,
    primitives::{OffsetOutline, PointsIter, PrimitiveStyle, Rectangle},
    transform::Transform,
    Drawable, Pixel,
};

// Create a wrapper type to avoid orphan rule issues
#[derive(Copy, Clone, Debug)]
pub struct StyledClosedCubicBezierPath<C: PixelColor> {
    pub path: ClosedCubicBezierPath,
    pub style: PrimitiveStyle<C>,
}

impl<C: PixelColor> StyledClosedCubicBezierPath<C> {
    pub fn new(path: ClosedCubicBezierPath, style: PrimitiveStyle<C>) -> Self {
        Self { path, style }
    }
}

impl<C: PixelColor> OriginDimensions for StyledClosedCubicBezierPath<C> {
    fn size(&self) -> embedded_graphics::geometry::Size {
        self.path.size()
    }
}

impl<C: PixelColor> Transform for StyledClosedCubicBezierPath<C> {
    fn translate(&self, by: Point) -> Self {
        Self {
            path: self.path.translate(by),
            style: self.style,
        }
    }

    fn translate_mut(&mut self, by: Point) -> &mut Self {
        self.path.translate_mut(by);
        self
    }
}

// impl PointsIter for ClosedCubicBezierPath {
//     type Iter = crate::ClosedCubicBezierPathPoints;

//     fn points(&self) -> Self::Iter {
//         crate::ClosedCubicBezierPathPoints::new(*self)
//     }
// }

impl OffsetOutline for ClosedCubicBezierPath {
    fn offset(&self, offset: i32) -> Self {
        // Create a new path with expanded bounding box
        let expanded_box = Rectangle::new(
            self.bounding_box.top_left - Point::new(offset, offset),
            self.bounding_box.size + embedded_graphics::geometry::Size::new(
                (offset * 2) as u32,
                (offset * 2) as u32,
            ),
        );
        
        Self {
            bezier_segments: self.bezier_segments,
            bounding_box: expanded_box,
            subdivision_count: self.subdivision_count,
        }
    }
}

impl Transform for ClosedCubicBezierPath {
    fn translate(&self, by: Point) -> Self {
        Self {
            bezier_segments: self.bezier_segments, // Note: segments are not translated, only bounding box
            bounding_box: self.bounding_box.translate(by),
            subdivision_count: self.subdivision_count,
        }
    }

    fn translate_mut(&mut self, by: Point) -> &mut Self {
        self.bounding_box.translate_mut(by);
        self
    }
}

/// Iterator that yields pixels for a styled closed cubic Bezier path
pub struct StyledPixels<C>
where
    C: PixelColor,
{
    fill_iter: Option<FilledClosedCubicBezierPathPoints>,
    stroke_iter: Option<StrokedClosedCubicBezierPathPoints>,
    fill_color: Option<C>,
    stroke_color: Option<C>,
    current_mode: DrawingMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DrawingMode {
    Fill,
    Stroke,
    Done,
}

impl<C> StyledPixels<C>
where
    C: PixelColor,
{
    fn new(
        path: ClosedCubicBezierPath,
        style: &PrimitiveStyle<C>,
    ) -> Self {
        let fill_iter = style.fill_color.map(|_| FilledClosedCubicBezierPathPoints::new(path));
        
        let has_stroke = style.stroke_color.is_some() && style.stroke_width > 0;
        let stroke_iter = if has_stroke {
            Some(StrokedClosedCubicBezierPathPoints::new(path, style.stroke_width))
        } else {
            None
        };

        let current_mode = if style.fill_color.is_some() {
            DrawingMode::Fill
        } else if has_stroke {
            DrawingMode::Stroke
        } else {
            DrawingMode::Done
        };

        Self {
            fill_iter,
            stroke_iter,
            fill_color: style.fill_color,
            stroke_color: style.stroke_color,
            current_mode,
        }
    }
}

impl<C> Iterator for StyledPixels<C>
where
    C: PixelColor,
{
    type Item = Pixel<C>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.current_mode {
                DrawingMode::Fill => {
                    if let Some(ref mut fill_iter) = self.fill_iter {
                        if let Some(point) = fill_iter.next() {
                            if let Some(color) = self.fill_color {
                                return Some(Pixel(point, color));
                            }
                        }
                    }
                    // Fill is done, move to stroke
                    self.current_mode = if self.stroke_iter.is_some() {
                        DrawingMode::Stroke
                    } else {
                        DrawingMode::Done
                    };
                }
                DrawingMode::Stroke => {
                    if let Some(ref mut stroke_iter) = self.stroke_iter {
                        if let Some(point) = stroke_iter.next() {
                            if let Some(color) = self.stroke_color {
                                return Some(Pixel(point, color));
                            }
                        }
                    }
                    // Stroke is done
                    self.current_mode = DrawingMode::Done;
                }
                DrawingMode::Done => {
                    return None;
                }
            }
        }
    }
}

impl<C> Drawable for StyledClosedCubicBezierPath<C>
where
    C: PixelColor,
{
    type Color = C;
    type Output = ();

    fn draw<D>(&self, target: &mut D) -> Result<Self::Output, D::Error>
    where
        D: DrawTarget<Color = Self::Color>,
    {
        let pixels = StyledPixels::new(self.path, &self.style);
        target.draw_iter(pixels)
    }
}

// Implement IntoIterator for styled path to make it easier to use
impl<C> IntoIterator for StyledClosedCubicBezierPath<C>
where
    C: PixelColor,
{
    type Item = Pixel<C>;
    type IntoIter = StyledPixels<C>;

    fn into_iter(self) -> Self::IntoIter {
        StyledPixels::new(self.path, &self.style)
    }
}

// Helper functions for creating styled paths
impl ClosedCubicBezierPath {
    /// Create a styled path with the given primitive style
    pub fn into_styled<C>(self, style: PrimitiveStyle<C>) -> StyledClosedCubicBezierPath<C>
    where
        C: PixelColor,
    {
        StyledClosedCubicBezierPath::new(self, style)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_graphics::{
        mock_display::MockDisplay,
        pixelcolor::BinaryColor,
        primitives::PrimitiveStyleBuilder,
    };

    #[test]
    fn test_styled_bezier_drawing() {
        let segments = &[
            crate::BezierSegment([
                [10.0, 10.0],
                [10.0, 40.0],
                [40.0, 40.0],
                [40.0, 10.0],
            ]),
            crate::BezierSegment([
                [40.0, 10.0],
                [70.0, 10.0],
                [70.0, 40.0],
                [40.0, 40.0],
            ]),
        ];

        let path = ClosedCubicBezierPath::new(segments, 20);
        
        let style = PrimitiveStyleBuilder::new()
            .fill_color(BinaryColor::On)
            .stroke_color(BinaryColor::On)
            .stroke_width(2)
            .build();

        let styled_path = path.into_styled(style);

        let mut display = MockDisplay::new();
        styled_path.draw(&mut display).unwrap();

        // The exact pattern depends on the bezier curve, but we should have some pixels
        assert!(!display.affected_area().is_zero_sized());
    }
}