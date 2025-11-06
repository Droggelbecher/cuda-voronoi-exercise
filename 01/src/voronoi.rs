use rand::Rng;

pub(crate) struct VoronoiProblem {
    pub(crate) map_size: (i32, i32),
    pub(crate) centers: Vec<(i32, i32)>,
}

pub(crate) struct VoronoiSolution {
    pub(crate) map: Vec<i32>,
}

impl VoronoiProblem {
    pub fn new_random(w: i32, h: i32, n_centers: usize) -> VoronoiProblem {
        let mut rng = rand::rng();

        let centers = (0..n_centers)
            .map(|_| (rng.random_range(0..w), rng.random_range(0..h)))
            .collect();

        VoronoiProblem {
            map_size: (w, h),
            centers
        }
    }
}
