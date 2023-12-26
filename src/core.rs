use core::ops::{Shl, Shr};

pub type Word = u64;

// (pi * 1.shl(62)) computed at high precision and rounded down
pub const META_PERMUTOR: Word = 0xc90fdaa2_2168c234;

pub const STRIPE_MASKS: [Word; 6] = [
    0xaaaaaaaaaaaaaaaa,
    0xcccccccccccccccc,
    0xf0f0f0f0f0f0f0f0,
    0xff00ff00ff00ff00,
    0xffff0000ffff0000,
    0xffffffff00000000
];

// PRIME_ROTATION_AMOUNTS is padded to speed up copying; NUM_PRIMES is its real length.
pub const NUM_PRIMES: usize = 11;
pub const PRIME_ROTATION_AMOUNTS: [u8; NUM_PRIMES] = [
    2, 3, 5, 7,
    11, 13, 17, 19,
    23, 29, 31];

#[inline]
pub fn shift_stripe(input: Word, mut permutor: Word, round: u32) -> Word {
    let mut out = input;
    permutor = permutor.rotate_right(round);
    for i in 0..8 {
        let rotation_selector = (permutor as usize).wrapping_add(i) % NUM_PRIMES;
        out ^= STRIPE_MASKS[(permutor % 6) as usize];
        out ^= out.rotate_right(PRIME_ROTATION_AMOUNTS[rotation_selector] as u32)
            .wrapping_add(META_PERMUTOR);
        let swap_selector = ((permutor / 6) % 6) as usize;
        let swap_mask = STRIPE_MASKS[swap_selector];
        out = (out & swap_mask).shr(1.shl(swap_selector)) | (out & !swap_mask).shl(1.shl(swap_selector));
        permutor >>= 8;
    }
    out
}
