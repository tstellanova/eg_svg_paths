#![no_std]

use num_enum::TryFromPrimitive;

pub mod closed_poly;
pub use closed_poly::*;


/// A 3x3 grid describing the direction the eyes are looking, 
/// from the observer's perspective.
#[derive(Clone, Copy, Debug, Eq, PartialEq, TryFromPrimitive)]
#[repr(u8)]
pub enum GazeDirection {
    NorthWest = 0,
    North = 1,
    NorthEast = 2,
    West = 3,
    StraightAhead = 4,// straight in front  
    East = 5,
    SouthWest = 6,
    South = 7,
    SouthEast = 8,   
    MaxCount
}


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

