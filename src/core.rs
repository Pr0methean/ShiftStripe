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

#[inline]
pub fn shift_stripe(input: Word, permutor: &mut Word) -> Word {
    let mut out = input;
    for i in 0..5 {
        out ^= (out ^ STRIPE_MASKS[(*permutor % 6) as usize])
            .wrapping_add(META_PERMUTOR).rotate_right((3 + 2*i) as u32);
        let swap_selector = ((*permutor >> 3) % 6) as usize;
        let swap_mask = STRIPE_MASKS[swap_selector];
        out = (out & swap_mask).shr(1.shl(swap_selector)) | (out & !swap_mask).shl(1.shl(swap_selector));
        *permutor >>= 12;
    }
    out
}
