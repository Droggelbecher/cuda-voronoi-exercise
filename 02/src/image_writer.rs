use std::collections::HashMap;

use image::{Rgb, RgbImage};
use rand::Rng as _;

use crate::voronoi::{VoronoiProblem, VoronoiSolution};

pub(crate) fn write_image(problem: &VoronoiProblem, solution: &VoronoiSolution, suffix: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = RgbImage::new(problem.map_size.0 as u32, problem.map_size.1 as u32);

    let mut colors = HashMap::new();
    let mut rng = rand::rng();

    colors.insert(-1, Rgb([0u8, 0, 0]));

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

    // Make voronoi centers white
    for (x, y) in &problem.centers {
        img.put_pixel(*x as u32, *y as u32, Rgb([0xffu8, 0xff, 0xff]));
    }

    Ok(img.save(format!("voronoi_{}.png", suffix))?)
}
