use core::mem::size_of;
use log::info;
use rand::{Rng};
use rand_core::block::BlockRngCore;
use crate::block::{random_block, xor_blocks};
use crate::core::{META_PERMUTOR, shift_stripe, Word};

fn shift_stripe_feistel<const WORDS_PER_BLOCK: usize>(
        left: &mut [Word; WORDS_PER_BLOCK], right: &mut [Word; WORDS_PER_BLOCK],
        permutor: &mut [Word; WORDS_PER_BLOCK], rounds: u32)  {
    let mut new_left = [0; WORDS_PER_BLOCK];
    for round in 0..rounds {
        new_left[1..WORDS_PER_BLOCK].copy_from_slice(&right[0..(WORDS_PER_BLOCK - 1)]);
        new_left[0] = right[WORDS_PER_BLOCK - 1];
        for unit_index in 0..WORDS_PER_BLOCK {
            let f = shift_stripe(right[unit_index], permutor[unit_index]);
            right[unit_index] = left[unit_index] ^ f;
            permutor[unit_index] = permutor[unit_index].wrapping_add(META_PERMUTOR).rotate_right(13 + 2 * round);
        }
        left.copy_from_slice(&new_left);
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ShiftStripeFeistelRngCore<const WORDS_PER_BLOCK: usize>
where [(); 2 * WORDS_PER_BLOCK]: {
    state: [Word; 2 * WORDS_PER_BLOCK]
}

impl <const WORDS_PER_BLOCK: usize> BlockRngCore for ShiftStripeFeistelRngCore<WORDS_PER_BLOCK>
where [(); 2 * WORDS_PER_BLOCK]:, [Word; WORDS_PER_BLOCK]: Default, [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: {
    type Item = Word;
    type Results = [Word; WORDS_PER_BLOCK];

    #[inline(always)]
    fn generate(&mut self, results: &mut Self::Results) {
        let mut state_blocks = self.state.array_chunks_mut();
        let first: &mut [Word; WORDS_PER_BLOCK] = state_blocks.next().unwrap();
        let second = state_blocks.next().unwrap();
        let mut temp_block = xor_blocks(first, second);
        shift_stripe_feistel(
            first,
            &mut temp_block,
            second,
            Self::FEISTEL_ROUNDS_TO_DIFFUSE);
        drop(state_blocks);
        if WORDS_PER_BLOCK > 1 {
            self.state.rotate_right(1);
        }
        state_blocks = self.state.array_chunks_mut();
        let first: &mut [Word; WORDS_PER_BLOCK] = state_blocks.next().unwrap();
        *results = xor_blocks(&temp_block, first);
    }
}

impl <const WORDS_PER_BLOCK: usize> ShiftStripeFeistelRngCore<WORDS_PER_BLOCK>
    where [(); 2 * WORDS_PER_BLOCK]:, [Word; WORDS_PER_BLOCK]: Default {
    // TODO: Find out why 2 blocks still require 3 rounds.
    const FEISTEL_ROUNDS_TO_DIFFUSE: u32 = if WORDS_PER_BLOCK <= 3 {
        3
    } else {
        WORDS_PER_BLOCK
    } as u32;
    pub fn new(seed: [Word; 2 * WORDS_PER_BLOCK]) -> ShiftStripeFeistelRngCore<WORDS_PER_BLOCK> {
        ShiftStripeFeistelRngCore {
            state: seed
        }
    }
}

impl <const WORDS_PER_BLOCK: usize> ShiftStripeFeistelRngCore<WORDS_PER_BLOCK>
    where [(); 2 * WORDS_PER_BLOCK]:, [Word; WORDS_PER_BLOCK]: Default {
    pub fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeFeistelRngCore<WORDS_PER_BLOCK> {
        let seed: [Word; WORDS_PER_BLOCK] = random_block(rng);
        info!("PRNG seed: {:016x?}", seed);
        Self::new(random_block(rng))
    }
}

#[cfg(test)]
mod tests {
    use rand_core::block::BlockRng64;
    use crate::core::Word;
    use core::mem::size_of;
    use std::fmt::Debug;
    use rand_core::{Error, RngCore};
    use rusty_fork::rusty_fork_test;
    use testu01::decorators::ReverseBits;
    use crate::prng::ShiftStripeFeistelRngCore;

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
    diffusion_small_keys_test!(17);

    fn check_diffusion_around<const WORDS_PER_BLOCK: usize>(permutor: i128, i: i128, rounds: u32) -> Vec<String>
        where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: {
        let mut warnings = check_diffusion::<WORDS_PER_BLOCK>(permutor, 0, i - 1, 0, i, rounds);
        warnings.extend_from_slice(&check_diffusion::<WORDS_PER_BLOCK>(permutor, i - 1, 0, i, 0, rounds));

        warnings.extend_from_slice(&check_diffusion::<WORDS_PER_BLOCK>(permutor, i - 1, i, i, i, rounds));
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
        shift_stripe_feistel(&mut prev1, &mut prev2, &mut permutor.clone(), rounds);
        shift_stripe_feistel(&mut this1, &mut this2, &mut permutor.clone(), rounds);
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
            warnings.push(format!("Warning: for permutor {} and inputs ({}, {}) and ({}, {}), outputs ({:016x?}, {:016x?}) and ({:016x?}, {:016x?}) differ by ({:016x?}, {:016x?})",
                                  permutor_int, previn1, previn2, thisin1, thisin2,
                                  &prev1, &prev2,
                                  &this1, &this2,
                                  &xor1, &xor2)
            );
        }
        warnings
    }

    fn test_big_crush<T: RngCore + Debug>(prng: T, name: &'static str) {
        use std::ffi::CString;
        use testu01::unif01::Unif01Gen;

        let name = CString::new(name.as_bytes().to_vec()).unwrap();
        let mut u01 = Unif01Gen::new(prng, name);
        testu01::battery::big_crush(&mut u01);
        let p_values = testu01::battery::get_pvalues();
        for (name, p) in p_values.iter() {
            println!("{:20}: {:0.4}", name, p);
        }
        for (name, p) in p_values.into_iter() {
            assert!(p > 0.001 && p < 0.999, "p value out of range for {}", name);
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    struct HalfOutputSelector<T: RngCore> {
        source: T,
        upper_half: bool
    }

    impl <T: RngCore> RngCore for HalfOutputSelector<T> {
        #[inline]
        fn next_u32(&mut self) -> u32 {
            let source = self.source.next_u64();
            if self.upper_half {
                (source >> 32) as u32
            } else {
                source as u32
            }
        }

        fn next_u64(&mut self) -> u64 {
            (self.next_u32() as u64) << 32 | (self.next_u32() as u64)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.chunks_mut(size_of::<u32>()).for_each(|chunk|
                chunk.copy_from_slice(&self.next_u32().to_le_bytes()[0..chunk.len()]));
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_big_crush_basic() {
            test_big_crush(BlockRng64::new(ShiftStripeFeistelRngCore::<1>::new([0; 2])), "ShiftStripeSwap");
        }

        #[test]
        fn test_big_crush_reversed_bits() {
            test_big_crush(ReverseBits { rng: BlockRng64::new(ShiftStripeFeistelRngCore::<1>::new([0; 2])) },
                           "ShiftStripeSwap-ReversedBits")
        }

        #[test]
        fn test_big_crush_upper_half() {
            test_big_crush(HalfOutputSelector {
                source: BlockRng64::new(ShiftStripeFeistelRngCore::<1>::new([0; 2])),
                upper_half: true
            }, "ShiftStripeSwap-UpperHalf")
        }

        #[test]
        fn test_big_crush_upper_half_reversed_bits() {
            test_big_crush(ReverseBits { rng: HalfOutputSelector {
                source: BlockRng64::new(ShiftStripeFeistelRngCore::<1>::new([0; 2])),
                upper_half: true
            }}, "ShiftStripeSwap-UpperHalf-ReversedBits")
        }

        #[test]
        fn test_big_crush_lower_half() {
            test_big_crush(HalfOutputSelector {
                source: BlockRng64::new(ShiftStripeFeistelRngCore::<1>::new([0; 2])),
                upper_half: false
            }, "ShiftStripeSwap-LowerHalf")
        }

        #[test]
        fn test_big_crush_lower_half_reversed_bits() {
            test_big_crush(ReverseBits { rng: HalfOutputSelector {
                source: BlockRng64::new(ShiftStripeFeistelRngCore::<1>::new([0; 2])),
                upper_half: false
            }}, "ShiftStripeSwap-LowerHalf-ReversedBits")
        }
    }
}
