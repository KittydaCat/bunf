mod bf;
mod bfasm;

mod rustc;

pub use rustc::main as rustc_main;

fn main() {
    rustc::main()
}
