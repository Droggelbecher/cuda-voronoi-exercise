use crate::voronoi::{VoronoiProblem, VoronoiSolution};

use std::error::Error;
use std::ffi::CString;

use rustacuda::{launch, prelude::*};

const BX: i32 = 32;
const BY: i32 = 32;

pub(crate) fn solve_cuda(problem: &VoronoiProblem) -> Result<VoronoiSolution, Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;
    let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    println!("CUDA Device {:?}", device);

    let module = {
        let module_data = CString::new(include_str!("../kernel.ptx"))?;
        Module::load_from_string(&module_data)?
    };

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create buffers on the GPU

    let mut d_centers = DeviceBuffer::from_slice(&problem.centers)?;

    // Create the output map unitialized on the GPU, our kernel
    // is expected to fill it completely.
    let map_size = (problem.map_size.0 * problem.map_size.1) as usize;
    let mut d_map = unsafe { DeviceBuffer::uninitialized(map_size)? };

    let n_blocks = (
        ((problem.map_size.0 + BX - 1) / BX) as u32,
        ((problem.map_size.1 + BY - 1) / BY) as u32,
    );

    println!("Launching CUDA kernel with map size {:?} and {} voronoi centers", problem.map_size, problem.centers.len());
    println!("blocks: {:?} block dim: {:?}", n_blocks, (BX, BY));

    unsafe {
        launch!(
            // n blocks, block size, shared, stream
            module.voronoi<<<n_blocks, (BX as u32, BY as u32), 0, stream>>>(
                d_map.as_device_ptr(),
                problem.map_size.0,
                problem.map_size.1,
                problem.centers.len(),
                d_centers.as_device_ptr()
            )
        ).unwrap();
    }

    stream.synchronize()?;

    let mut map = vec![0; map_size];
    d_map.copy_to(&mut map)?;

    Ok(VoronoiSolution { map })
}