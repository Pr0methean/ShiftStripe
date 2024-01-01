use core::ops::{Shl};

pub type Word = u64;

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
fn shuffle<T>(n : &mut Word, a : &mut [T]) {
    for i in (1..(a.len() as Word)).rev() {
        let j = *n % (i + 1) as Word;
        *n /= (i + 1);
        a.swap(j as usize, i as usize);
    }
}

#[inline]
pub fn shift_stripe(input: Word, mut permutor: Word) -> Word {
    let mut stripe_masks = STRIPE_MASKS.clone();
    shuffle(&mut permutor, &mut stripe_masks);
    let mut swap_selectors: [Word; 7] = [0, 1, 2, 3, 4, 5, 6];
    shuffle(&mut permutor, &mut swap_selectors);
    let mut out = input;
    stripe_masks.into_iter().zip(swap_selectors).for_each(|(stripe_mask, swap_selector)| {
        out ^= (out ^ stripe_mask)
            .wrapping_add(META_PERMUTOR).rotate_right(3);
        let swap_mask = STRIPE_MASKS[swap_selector as usize];
        out = (out & swap_mask).rotate_right(1.shl(swap_selector)) | (out & !swap_mask).rotate_left(1.shl(swap_selector));
    });
    out
}
