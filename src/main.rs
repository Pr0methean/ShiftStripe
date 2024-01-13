#![feature(array_chunks)]
#![feature(iter_array_chunks)]
#![feature(unboxed_closures)]
#![feature(generic_const_exprs)]
#![feature(iter_collect_into)]
#![feature(slice_as_chunks)]
#![feature(portable_simd)]

mod core;
mod block;
mod prng;
mod hashing;

use std::error::Error;
use std::io::Write;
use rand::{RngCore, thread_rng};
use rand_core::block::{BlockRng64};
use crate::prng::ShiftStripeFeistelRngCore;

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = BlockRng64::new(ShiftStripeFeistelRngCore::from_rng(&mut thread_rng()));
    let mut stdout = std::io::stdout();
    let mut write_result = Ok(());
    let mut out_buffer = [0u8; 1024];
    while write_result.is_ok() {
        rng.fill_bytes(&mut out_buffer);
        write_result = stdout.write_all(&out_buffer);
    }
    Ok(())
}
