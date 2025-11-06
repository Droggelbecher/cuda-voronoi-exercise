use std::{env, error::Error};

use crate::{cuda_solver::solve_cuda, image_writer::write_image, voronoi::VoronoiProblem};

mod cuda_solver;
mod image_writer;
mod voronoi;

fn main() -> Result<(), Box<dyn Error>> {

    let args: Vec<String> = env::args().collect();
    let size: i32 = args[1].parse()?;
    let n_centers: usize = args[2].parse()?;

    let problem = VoronoiProblem::new_random(size, size, n_centers);
    let solution = solve_cuda(&problem)?;

    write_image(&problem, &solution, &args[1])?;

    println!("ok.");
    Ok(())
}
