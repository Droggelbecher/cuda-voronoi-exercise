use std::collections::HashMap;

use image::{Rgb, RgbImage};
use rand::Rng as _;

use crate::voronoi::{VoronoiProblem, VoronoiSolution};

pub(crate) fn write_image(problem: &VoronoiProblem, solution: &VoronoiSolution, suffix: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = RgbImage::new(problem.map_size.0 as u32, problem.map_size.1 as u32);

    let mut colors = HashMap::new();
    let mut rng = rand::rng();

    for x in 0..problem.map_size.0 {
        for y in 0..problem.map_size.1 {
            let c = solution.map[(x + y * problem.map_size.0) as usize];
            let color = colors.entry(c).or_insert_with(|| {
                Rgb([
                    rng.random::<u8>(),
                    rng.random::<u8>(),
                    rng.random::<u8>(),
                ])
            });

            img.put_pixel(x as u32, y as u32, *color);
        }
    }

    Ok(img.save(format!("voronoi_{}.png", suffix))?)
}
