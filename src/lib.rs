#![feature(array_chunks)]
#![feature(iter_array_chunks)]
#![feature(unboxed_closures)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(iter_collect_into)]
#![feature(slice_as_chunks)]
#![feature(optimize_attribute)]
#![feature(portable_simd)]

pub mod block;
pub mod core;
pub mod hashing;
pub mod prng;