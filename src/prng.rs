use std::mem::swap;
use rand::{Rng};
use rand_core::block::BlockRngCore;
use crate::block::random_block;
use crate::core::{rotate_permutor, shift_stripe, shuffle_lanes, Vector, VECTOR_SIZE, Word};

pub const RNG_ROUNDS: u32 = 3;

fn shift_stripe_feistel(left: &mut Vector, right: &mut Vector, permutor: &mut Vector, rounds: u32)  {
    let new_left = shuffle_lanes(*right);
    for _ in 0..rounds {
        shift_stripe(right, permutor.clone());
        *right ^= *left;
        *permutor = rotate_permutor(*permutor);
        *left = new_left;
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ShiftStripeFeistelRngCore {
    first_state: Vector,
    second_state: Vector
}

impl BlockRngCore for ShiftStripeFeistelRngCore {
    type Item = Word;
    type Results = [Word; VECTOR_SIZE];

    #[inline(always)]
    fn generate(&mut self, results: &mut Self::Results) {
        shift_stripe_feistel(
            &mut self.first_state,
            &mut self.second_state,
            &mut self.second_state,
            RNG_ROUNDS);
        let mut output = self.first_state.clone();
        shift_stripe(&mut output, self.second_state);
        *results = *self.first_state ^ output;
        self.second_state ^= shuffle_lanes(rotate_permutor(self.first_state));
        swap(&mut self.first_state, &mut self.second_state);
    }
}

impl ShiftStripeFeistelRngCore {
    pub fn new(seed: [Word; 2 * VECTOR_SIZE]) -> ShiftStripeFeistelRngCore {
        let mut chunks = seed.array_chunks::<VECTOR_SIZE>().copied();
        ShiftStripeFeistelRngCore {
            first_state: Vector::from_array(chunks.next().unwrap()),
            second_state: Vector::from_array(chunks.next().unwrap()),
        }
    }

    pub fn from_rng<T: Rng>(rng: &mut T) -> ShiftStripeFeistelRngCore {
        Self::new(random_block(rng))
    }
}

#[cfg(test)]
mod tests {
    use rand_core::block::BlockRng64;
    use crate::core::{Vector, VECTOR_SIZE};
    use core::mem::size_of;
    use std::fmt::Debug;
    use rand_core::{Error, RngCore};
    use rand_core::impls::fill_bytes_via_next;
    use rusty_fork::rusty_fork_test;
    use testu01::decorators::ReverseBits;
    use crate::prng::ShiftStripeFeistelRngCore;

    macro_rules! diffusion_small_keys_test {
        ($num_rounds: expr) => {
            #[cfg(test)]
            paste::item! {
                #[test]
                fn [<test_diffusion_small_keys_ $num_rounds _rounds >]() {
                    for permutor_short in -128..128 {
                        for i in -128..128 {
                            assert_eq!(vec![] as Vec<String>, check_diffusion_around(permutor_short, i, $num_rounds));
                        }
                    }
                }
            }
        }
    }
    diffusion_small_keys_test!(03);
    diffusion_small_keys_test!(04);

    fn check_diffusion_around(permutor: i128, i: i128, rounds: u32) -> Vec<String> {
        let mut warnings = check_diffusion(permutor, 0, i - 1, 0, i, rounds);
        warnings.extend_from_slice(&check_diffusion(permutor, i - 1, 0, i, 0, rounds));

        warnings.extend_from_slice(&check_diffusion(permutor, i - 1, i, i, i, rounds));
        warnings.extend_from_slice(&check_diffusion(permutor, i - 1, i - 1, i, i, rounds));
        warnings.extend_from_slice(&check_diffusion(permutor, i, i, i, i - 1, rounds));
        warnings
    }

    fn check_diffusion(permutor_int: i128, previn1: i128, previn2: i128, thisin1: i128, thisin2: i128,
                                                     rounds: u32) -> Vec<String> {
        use crate::block::{int_to_vector};
        use crate::prng::shift_stripe_feistel;

        let mut warnings = Vec::new();
        let permutor: Vector = int_to_vector(permutor_int);
        let mut prev1: Vector = int_to_vector(previn1);
        let mut prev2: Vector = int_to_vector(previn2);
        let mut this1: Vector = int_to_vector(thisin1);
        let mut this2: Vector = int_to_vector(thisin2);
        shift_stripe_feistel(&mut prev1, &mut prev2, &mut permutor.clone(), rounds);
        shift_stripe_feistel(&mut this1, &mut this2, &mut permutor.clone(), rounds);
        let xor1 = prev1 ^ this1;
        let xor2 = prev2 ^ this2;
        let bits_in_block = 8 * size_of::<Vector>();
        let bits_difference_1: usize = xor1.as_array().iter().copied().map(|x| x.count_ones() as u64).sum::<u64>().try_into().unwrap();
        let bits_difference_2: usize = xor2.as_array().iter().copied().map(|x| x.count_ones() as u64).sum::<u64>().try_into().unwrap();
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
            fill_bytes_via_next(self, dest);
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_big_crush_basic() {
            test_big_crush(BlockRng64::new(ShiftStripeFeistelRngCore::new([0; 2 * VECTOR_SIZE])), "ShiftStripeSwap");
        }

        #[test]
        fn test_big_crush_reversed_bits() {
            test_big_crush(ReverseBits { rng: BlockRng64::new(ShiftStripeFeistelRngCore::new([0; 2 * VECTOR_SIZE])) },
                           "ShiftStripeSwap-ReversedBits")
        }

        #[test]
        fn test_big_crush_upper_half() {
            test_big_crush(HalfOutputSelector {
                source: BlockRng64::new(ShiftStripeFeistelRngCore::new([0; 2 * VECTOR_SIZE])),
                upper_half: true
            }, "ShiftStripeSwap-UpperHalf")
        }

        #[test]
        fn test_big_crush_upper_half_reversed_bits() {
            test_big_crush(ReverseBits { rng: HalfOutputSelector {
                source: BlockRng64::new(ShiftStripeFeistelRngCore::new([0; 2 * VECTOR_SIZE])),
                upper_half: true
            }}, "ShiftStripeSwap-UpperHalf-ReversedBits")
        }

        #[test]
        fn test_big_crush_lower_half() {
            test_big_crush(HalfOutputSelector {
                source: BlockRng64::new(ShiftStripeFeistelRngCore::new([0; 2 * VECTOR_SIZE])),
                upper_half: false
            }, "ShiftStripeSwap-LowerHalf")
        }

        #[test]
        fn test_big_crush_lower_half_reversed_bits() {
            test_big_crush(ReverseBits { rng: HalfOutputSelector {
                source: BlockRng64::new(ShiftStripeFeistelRngCore::new([0; 2 * VECTOR_SIZE])),
                upper_half: false
            }}, "ShiftStripeSwap-LowerHalf-ReversedBits")
        }
    }
}
