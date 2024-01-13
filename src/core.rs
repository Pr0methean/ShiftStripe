use core::ops::{Shl};
use std::mem::swap;
use std::ops::Shr;
use std::simd::{Simd, simd_swizzle};
use std::simd::num::SimdUint;
use array_macro::array;

pub type Word = u64;

pub const VECTOR_SIZE: usize = 4;

pub type Vector = Simd<Word, VECTOR_SIZE>;
pub type VectorUsize = Simd<usize, VECTOR_SIZE>;

// (pi * 1.shl(62)) computed at high precision and rounded down
pub const META_PERMUTOR: Word = 0xc90fdaa2_2168c234;

pub const STRIPE_MASKS: [Word; 7] = [
    0xaaaaaaaaaaaaaaaa,
    0xcccccccccccccccc,
    0xf0f0f0f0f0f0f0f0,
    0xff00ff00ff00ff00,
    0xffff0000ffff0000,
    0xffffffff00000000,
    0
];

#[inline]
pub(crate) fn shuffle_lanes(n: Vector) -> Vector {
    simd_swizzle!(n, [1, 3, 0, 2])
}

#[inline]
fn shuffled_mask_indices(n : &mut Vector) -> [VectorUsize; STRIPE_MASKS.len()] {
    let mut indices = array![i => [i; VECTOR_SIZE]; STRIPE_MASKS.len()];
    for i in (1..STRIPE_MASKS.len()).rev() {
        let modulus = Vector::splat(i as Word + 1);
        let j = *n % modulus;
        *n /= &modulus;
        let (left_mut, right_mut) = indices.split_at_mut(i);
        for lane in 0..VECTOR_SIZE {
            swap(&mut left_mut[j[lane] as usize][lane], &mut right_mut[0][lane]);
        }
    }
    array![i => VectorUsize::from_array(indices[i]); STRIPE_MASKS.len()]
}

#[inline]
fn load_mask_vectors(indices: [VectorUsize; STRIPE_MASKS.len()], a : &mut [Vector; STRIPE_MASKS.len()]) {
    a.iter_mut().zip(indices.into_iter()).for_each(|(a, indices)| {
        *a = Vector::gather_or_default(STRIPE_MASKS.as_slice(), indices);
    });
}

#[inline]
pub fn shift_stripe(input: Vector, mut permutor: Vector) -> Vector {
    let mut stripe_masks = [Vector::splat(0); STRIPE_MASKS.len()];
    load_mask_vectors(shuffled_mask_indices(&mut permutor), &mut stripe_masks);
    let swap_selectors = shuffled_mask_indices(&mut permutor);
    let mut swap_masks = [Vector::splat(0); STRIPE_MASKS.len()];
    load_mask_vectors(swap_selectors, &mut swap_masks);
    let mut swap_rotation_amounts: [Vector; STRIPE_MASKS.len()] = [Vector::splat(1); STRIPE_MASKS.len()];
    swap_rotation_amounts.iter_mut().zip(swap_selectors).for_each(|(rotation_amounts, selector)|
        *rotation_amounts <<= selector.cast());
    let mut out = input;
    stripe_masks.into_iter().zip(swap_masks).zip(swap_rotation_amounts).for_each(
        |((stripe_mask, swap_mask), swap_rotation_amount)| {
            out ^= ((out ^ stripe_mask) + Vector::splat(META_PERMUTOR)) >> Vector::splat(3);
            out = (out & swap_mask).shr(swap_rotation_amount)
                | (out & !swap_mask).shl(swap_rotation_amount);
        }
    );
    out
}
