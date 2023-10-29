use core::mem::size_of;
use rand::{Rng};
use rand_core::block::BlockRngCore;
use crate::block::{compress_block_to_unit, DefaultArray, random_block};
use crate::core::{shift_stripe, Word};

fn shift_stripe_feistel<const WORDS_PER_BLOCK: usize>(
    left: &mut [Word; WORDS_PER_BLOCK], right: &mut [Word; WORDS_PER_BLOCK], mut permutor: [Word; WORDS_PER_BLOCK], rounds: u32)  {
    for round in 0..rounds {
        for unit_index in 0..WORDS_PER_BLOCK {
            let new_left = right[unit_index].clone();
            let f = shift_stripe(right[unit_index], permutor[unit_index], round);
            right[unit_index] = left[unit_index] ^ f;
            let new_permutor = shift_stripe(permutor[unit_index], left[
                (unit_index + WORDS_PER_BLOCK / 2) % WORDS_PER_BLOCK], u32::MAX - round);
            permutor[unit_index] ^= new_permutor;
            left[unit_index] = new_left;
        }
        left.rotate_right(1);
    }
}

#[derive(Clone, Debug)]
pub struct ShiftStripeFeistelRngCore<const WORDS_PER_BLOCK: usize> {
    permutor: [Word; WORDS_PER_BLOCK],
    counter: Word
}

impl <const WORDS_PER_BLOCK: usize> BlockRngCore for ShiftStripeFeistelRngCore<WORDS_PER_BLOCK>
where [(); 2 * WORDS_PER_BLOCK]:, [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: {
    type Item = Word;
    type Results = DefaultArray<u64, { 2 * WORDS_PER_BLOCK }>;

    fn generate(&mut self, results: &mut Self::Results) {
        results.0[0] = self.counter;
        results.0[2 * WORDS_PER_BLOCK - 1] = self.counter;
        self.counter += 1;
        let mut result_blocks = results.0.array_chunks_mut();
        let first = result_blocks.next().unwrap();
        let second = result_blocks.next().unwrap();
        shift_stripe_feistel(
            first,
            second,
            self.permutor.clone(),
            Self::FEISTEL_ROUNDS_TO_DIFFUSE);
    }
}

impl <const WORDS_PER_BLOCK: usize> ShiftStripeFeistelRngCore<WORDS_PER_BLOCK>
    where [(); 2 * WORDS_PER_BLOCK]: {
    // TODO: Find out why 2 blocks still require 3 rounds.
    const FEISTEL_ROUNDS_TO_DIFFUSE: u32 = WORDS_PER_BLOCK as u32 + if WORDS_PER_BLOCK <= 2 {
        1
    } else {
        0
    };
    pub fn new(seed: [Word; WORDS_PER_BLOCK]) -> ShiftStripeFeistelRngCore<WORDS_PER_BLOCK> {
        let permutor = seed;
        let counter = compress_block_to_unit(&seed);
        ShiftStripeFeistelRngCore {
            permutor,
            counter
        }
    }
}

impl <const WORDS_PER_BLOCK: usize> ShiftStripeFeistelRngCore<WORDS_PER_BLOCK>
    where [(); 2 * WORDS_PER_BLOCK]: {
    pub fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeFeistelRngCore<WORDS_PER_BLOCK> {
        Self::new(random_block(rng))
    }
}

#[cfg(test)]
mod tests {
    use crate::core::Word;
    use core::mem::size_of;
    macro_rules! diffusion_small_keys_test {
        ($num_blocks: expr) => {
            diffusion_small_keys_test!($num_blocks, $num_blocks);
        };
        ($num_blocks: expr, $num_rounds: expr) => {
            #[cfg(test)]
            paste::item! {
                #[test]
                fn [<test_diffusion_small_keys_ $num_blocks _blocks_ $num_rounds _rounds >]() {
                    for permutor_short in -128..128 {
                        for i in -128..128 {
                            assert_eq!(vec![] as Vec<String>, check_diffusion_around::<$num_blocks>(permutor_short, i, $num_rounds));
                        }
                    }
                }
            }
        }
    }
    diffusion_small_keys_test!(02,03);
    diffusion_small_keys_test!(03);
    diffusion_small_keys_test!(04);
    diffusion_small_keys_test!(05);
    diffusion_small_keys_test!(06);
    diffusion_small_keys_test!(07);
    diffusion_small_keys_test!(08);
    diffusion_small_keys_test!(09);
    diffusion_small_keys_test!(10);
    diffusion_small_keys_test!(11);
    diffusion_small_keys_test!(12);
    diffusion_small_keys_test!(13);
    diffusion_small_keys_test!(14);
    diffusion_small_keys_test!(15);
    diffusion_small_keys_test!(16);

    fn check_diffusion_around<const WORDS_PER_BLOCK: usize>(permutor: i128, i: i128, rounds: u32) -> Vec<String>
        where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: {
        let mut warnings = check_diffusion::<WORDS_PER_BLOCK>(permutor, 0, i - 1, 0, i, rounds);
        warnings.extend_from_slice(&check_diffusion::<WORDS_PER_BLOCK>(permutor, i - 1, 0, i, 0, rounds));

        warnings.extend_from_slice(&check_diffusion::<WORDS_PER_BLOCK>(permutor, i - 1, i, i, i, rounds));
        warnings.extend_from_slice(&check_diffusion::<WORDS_PER_BLOCK>(permutor, i, i - 1, i, i, rounds));
        warnings.extend_from_slice(&check_diffusion::<WORDS_PER_BLOCK>(permutor, i - 1, i - 1, i, i, rounds));
        warnings.extend_from_slice(&check_diffusion::<WORDS_PER_BLOCK>(permutor, i, i, i, i - 1, rounds));
        warnings
    }

    fn check_diffusion<const WORDS_PER_BLOCK: usize>(permutor_int: i128, previn1: i128, previn2: i128, thisin1: i128, thisin2: i128,
                                                     rounds: u32) -> Vec<String>
        where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: {
        use crate::block::{int_to_block, xor_blocks};
        use crate::prng::shift_stripe_feistel;

        let mut warnings = Vec::new();
        let permutor: [Word; WORDS_PER_BLOCK] = int_to_block(permutor_int);
        let mut prev1: [Word; WORDS_PER_BLOCK] = int_to_block(previn1);
        let mut prev2: [Word; WORDS_PER_BLOCK] = int_to_block(previn2);
        let mut this1: [Word; WORDS_PER_BLOCK] = int_to_block(thisin1);
        let mut this2: [Word; WORDS_PER_BLOCK] = int_to_block(thisin2);
        shift_stripe_feistel(&mut prev1, &mut prev2, permutor, rounds);
        shift_stripe_feistel(&mut this1, &mut this2, permutor, rounds);
        let xor1 = xor_blocks(&prev1, &this1);
        let xor2 = xor_blocks(&prev2, &this2);
        let bits_in_block = 8 * size_of::<[Word; WORDS_PER_BLOCK]>();
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
}
