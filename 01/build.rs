
extern crate cc;

fn main() {
    println!(r"cargo:rustc-link-search=native=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64");
    println!("cargo:rustc-link-lib=static=cudart");

    // println!("cargo:rustc-link-lib=static=kernel");
}