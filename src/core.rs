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
fn shuffle<T>(n : &mut Word, a : &mut [T]) {
    for i in (1..a.len()).rev() {
        let j = *n % (i + 1) as Word;
        *n /= (i + 1);
        a.swap(j as usize, i);
    }
}

#[inline]
pub fn shift_stripe(input: Word, mut permutor: Word) -> Word {
    let mut stripe_masks = STRIPE_MASKS.clone();
    shuffle(&mut permutor, &mut stripe_masks);
    let mut swap_selectors: [Word; 6] = [0, 1, 2, 3, 4, 5];
    shuffle(&mut permutor, &mut swap_selectors);
    let mut out = input;
    stripe_masks.into_iter().zip(swap_selectors).for_each(|(stripe_mask, swap_selector)| {
        let rotation_selector = (permutor & 3) as u32;
        permutor >>= 2;
        out ^= (out ^ stripe_mask)
            .wrapping_add(META_PERMUTOR).rotate_right(1 + 2 * rotation_selector);
        let swap_mask = STRIPE_MASKS[swap_selector];
        out = (out & swap_mask).shr(1.shl(swap_selector)) | (out & !swap_mask).shl(1.shl(swap_selector));
    });
    out
}
