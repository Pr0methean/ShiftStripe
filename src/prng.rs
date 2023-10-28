use rand::Rng;
use rand_core::block::BlockRngCore;
use crate::block::{Block, compress_block_to_unit, int_to_block, random_block, WORDS_PER_BLOCK, xor_blocks};
use crate::core::{shift_stripe, Word};

// Equal to:
//  3 rounds at 2 words per block
//  2n+4 rounds at 2n+4 words per block
//  2n+4 rounds at 2n+3 words per block
// TODO: Find some theoretical explanation of why this is the right number.
pub const FEISTEL_ROUNDS_TO_DIFFUSE: u32 = WORDS_PER_BLOCK as u32 + if WORDS_PER_BLOCK <= 3 {
    1
} else {
    0
};

fn shift_stripe_feistel(mut left: Block, mut right: Block, mut permutor: Block, rounds: u32) -> (Block, Block) {
    for round in 0..rounds {
        let new_left = right;
        for unit_index in 0..WORDS_PER_BLOCK {
            let f = shift_stripe(right[unit_index], permutor[unit_index], round);
            right[unit_index] = left[unit_index] ^ f;
            let new_permutor = shift_stripe(permutor[unit_index], left[
                (unit_index + WORDS_PER_BLOCK / 2) % WORDS_PER_BLOCK], u32::MAX - round);
            permutor[unit_index] ^= new_permutor;
        }
        left = new_left;
        left.rotate_right(1);
    }
    (left, right)
}

#[derive(Copy, Clone)]
pub struct ShiftStripeFeistelRngCore {
    permutor: Block,
    counter: Word,
}

impl BlockRngCore for ShiftStripeFeistelRngCore {
    type Item = Word;
    type Results = [Word; 2 * WORDS_PER_BLOCK];

    fn generate(&mut self, results: &mut Self::Results) {
        let (result0, result1) = shift_stripe_feistel(
            int_to_block(self.counter.into()),
            int_to_block(self.counter.into()),
            self.permutor,
            FEISTEL_ROUNDS_TO_DIFFUSE);
        self.counter = self.counter.wrapping_add(1);
        results[0..WORDS_PER_BLOCK].copy_from_slice(&result0);
        results[WORDS_PER_BLOCK..].copy_from_slice(&result1);
    }
}

impl ShiftStripeFeistelRngCore {
    pub fn new(seed: Block) -> ShiftStripeFeistelRngCore {
        let permutor = seed;
        let counter = compress_block_to_unit(&seed);
        ShiftStripeFeistelRngCore {
            permutor, counter
        }
    }

    pub fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeFeistelRngCore {
        Self::new(random_block(rng))
    }
}

#[test]
fn test_diffusion_small_keys() {
    for permutor_short in -128..128 {
        for i in -128..128 {
            assert_eq!(vec![] as Vec<String>, check_diffusion_around(permutor_short, i));
        }
    }
}

#[cfg(test)]
fn check_diffusion_around(permutor: i128, i: i128) -> Vec<String> {
    let mut warnings = check_diffusion(permutor, 0, i - 1, 0, i);
    warnings.extend_from_slice(&check_diffusion(permutor, i - 1, 0, i, 0));

    warnings.extend_from_slice(&check_diffusion(permutor, i - 1, i, i, i));
    warnings.extend_from_slice(&check_diffusion(permutor, i, i - 1, i, i));
    warnings.extend_from_slice(&check_diffusion(permutor, i - 1, i - 1, i, i));
    warnings.extend_from_slice(&check_diffusion(permutor, i - 1, i, i, i));
    warnings
}

#[cfg(test)]
fn check_diffusion(permutor_int: i128, previn1: i128, previn2: i128, thisin1: i128, thisin2: i128) -> Vec<String> {
    use std::mem::size_of;

    let mut warnings = Vec::new();
    let permutor = int_to_block(permutor_int);
    let previn1_unsigned = int_to_block(previn1);
    let previn2_unsigned = int_to_block(previn2);
    let thisin1_unsigned = int_to_block(thisin1);
    let thisin2_unsigned = int_to_block(thisin2);
    let (prev1, prev2) = shift_stripe_feistel(previn1_unsigned, previn2_unsigned, permutor, FEISTEL_ROUNDS_TO_DIFFUSE);
    let (this1, this2) = shift_stripe_feistel(thisin1_unsigned, thisin2_unsigned, permutor, FEISTEL_ROUNDS_TO_DIFFUSE);
    let xor1 = xor_blocks(&prev1, &this1);
    let xor2 = xor_blocks(&prev2, &this2);
    let bits_in_block = 8 * size_of::<Block>();
    let bits_difference_1: usize = xor1.iter().copied().map(|x| x.count_ones() as u64).sum::<u64>().try_into().unwrap();
    let bits_difference_2: usize = xor2.iter().copied().map(|x| x.count_ones() as u64).sum::<u64>().try_into().unwrap();
    if prev1 == this1
        || prev2 == this2
        || prev1 == this2
        || this1 == prev2
        || bits_difference_1 < 16 || bits_difference_1 > bits_in_block - 16
        || bits_difference_2 < 16 || bits_difference_2 > bits_in_block - 16
        || (bits_difference_1 + bits_difference_2) < 64
        || (bits_difference_1 + bits_difference_2) > (2 * bits_in_block) - 64 {
        warnings.push(format!("Warning: for permutor {} and inputs ({}, {}) and ({}, {}), outputs ({:?}, {:?}) and ({:?}, {:?}) differ by ({:?}, {:?})",
                              permutor_int, previn1, previn2, thisin1, thisin2,
                              &prev1, &prev2,
                              &this1, &this2,
                              &xor1, &xor2)
        );
    }
    warnings
}
