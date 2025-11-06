& ./env.ps1

nvcc -c kernel.cu -ptx
cargo build --release