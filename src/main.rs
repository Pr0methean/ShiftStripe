use std::hash::Hasher;
use std::io::Write;
use std::ops::{Shr};
use rand::{Rng, RngCore, thread_rng};
use rand_core::block::{BlockRng64, BlockRngCore};
use rayon::prelude::*;

pub const PRIME_ROTATION_AMOUNTS: [u32; 31] = [
    2, 3, 5, 7, 11, 13, 17, 19,
    23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127,
];

// (pi * 1u128.shl(126)) computed at high precision and rounded down
pub const META_PERMUTOR: u128 = 0xc90fdaa2_2168c234_c4c6628b_80dc1cd1;

pub const STRIPE_MASKS: [u128; 8] = [
    0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
    0xcccccccccccccccccccccccccccccccc,
    0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0,
    0xff00ff00ff00ff00ff00ff00ff00ff00,
    0xffff0000ffff0000ffff0000ffff0000,
    0xffffffff00000000ffffffff00000000,
    0xffffffffffffffff0000000000000000,
    u128::MAX
];

pub const SIMPLE_ROUNDS_TO_DIFFUSE: usize = 256;
pub const FIESTEL_ROUNDS_TO_DIFFUSE: usize = 16;
pub const FEEDBACK_ROUNDS_TO_DIFFUSE: usize = 16;

pub fn shift_stripe(input: u128, permutor: u128) -> u128 {
    let mut out = input;
    for perm_byte in permutor.to_be_bytes() {
        out ^= STRIPE_MASKS[(perm_byte.shr(5)) as usize];
        out = out.rotate_right(PRIME_ROTATION_AMOUNTS[(perm_byte % 31) as usize]);
    }
    out = out.wrapping_add(META_PERMUTOR);
    out
}

pub fn unshift_stripe(input: u128, permutor: u128) -> u128 {
    let mut out = input;
    out = out.wrapping_sub(META_PERMUTOR);
    for perm_byte in permutor.to_le_bytes() {
        out = out.rotate_left(PRIME_ROTATION_AMOUNTS[(perm_byte % 31) as usize]);
        out ^= STRIPE_MASKS[(perm_byte.shr(5)) as usize];
    }
    out
}

#[test]
fn test_shift_stripe_reversible_smallnum() {
    for permutor in -128..128 {
        let permutor = permutor as u128;
        for input in -128..128 {
            let input = input as u128;
            assert_eq!(input, unshift_stripe(shift_stripe(input, permutor), permutor))
        }
    }
}

#[test]
fn test_shift_stripe_reversible_random() {
    let mut random_bytes = [0u8; 16];
    for _ in 0..100 {
        thread_rng().fill_bytes(&mut random_bytes);
        let permutor = u128::from_be_bytes(random_bytes);
        thread_rng().fill_bytes(&mut random_bytes);
        let input = u128::from_be_bytes(random_bytes);
        assert_eq!(input, unshift_stripe(shift_stripe(input, permutor), permutor))
    }
}

fn shift_stripe_feistel(input1: u128, input2: u128, mut permutor: u128, rounds: usize) -> (u128, u128) {
    let mut left = input1;
    let mut right = input2;

    for round in 0..rounds {
        let new_left = right;
        let f = shift_stripe(right, permutor);
        right = left ^ f;
        left = new_left;
        let prime_rotation = PRIME_ROTATION_AMOUNTS[(permutor.count_ones() as usize + round) % 31];
        let permuted_permutor = permutor.rotate_left(prime_rotation);
        permutor = shift_stripe(permutor, permuted_permutor);
    }
    (left, right)
}

fn shift_stripe_simple(input: u128, mut permutor: u128, rounds: usize) -> u128 {
    let (output, _) = shift_stripe_simple_feedback(input, permutor, rounds);
    output
}

fn shift_stripe_simple_feedback(input: u128, mut permutor: u128, rounds: usize) -> (u128, u128) {
    let mut output = input;
    for round in 0..rounds {
        output = shift_stripe(output, permutor);
        let prime_rotation = PRIME_ROTATION_AMOUNTS[(permutor.count_ones() as usize + round) % 31];
        let permuted_permutor = permutor.rotate_left(prime_rotation);
        permutor = shift_stripe(permutor, permuted_permutor);
    }
    (output, permutor)
}

#[derive(Copy, Clone)]
struct ShiftStripeFeistelRngCore {
    permutor: u128,
    counter: u128,
}

#[derive(Copy, Clone)]
struct ShiftStripeSimpleRngCore {
    permutor: u128,
    counter: u128,
}

impl BlockRngCore for ShiftStripeFeistelRngCore {
    type Item = u64;
    type Results = [u64; 2];

    fn generate(&mut self, results: &mut Self::Results) {
        let (result1, result2) = shift_stripe_feistel(0, self.counter, self.permutor, FIESTEL_ROUNDS_TO_DIFFUSE);
        self.counter = self.counter.wrapping_add(1);
        results[0] = compress_u128_to_u64(result1);
        results[1] = compress_u128_to_u64(result2);
    }
}

impl BlockRngCore for ShiftStripeSimpleRngCore {
    type Item = u64;
    type Results = [u64; 1];

    fn generate(&mut self, results: &mut Self::Results) {
        let result128 = shift_stripe_simple(self.counter, self.permutor, SIMPLE_ROUNDS_TO_DIFFUSE);
        self.counter = self.counter.wrapping_add(1);
        results[0] = compress_u128_to_u64(result128);
    }
}

#[derive(Copy, Clone)]
struct ShiftStripeFeedbackRngCore {
    permutor: u128,
    counter: u128,
}

#[inline]
const fn compress_u128_to_u64(input: u128) -> u64 {
    (input >> 64) as u64 ^ (input as u64)
}

impl BlockRngCore for ShiftStripeFeedbackRngCore {
    type Item = u64;
    type Results = [u64; 1];

    fn generate(&mut self, results: &mut Self::Results) {
        let (result128, permuted_permutor) = shift_stripe_simple_feedback(self.counter, self.permutor, FEEDBACK_ROUNDS_TO_DIFFUSE);
        self.counter = self.counter.wrapping_add(1);
        self.permutor ^= permuted_permutor;
        results[0] = compress_u128_to_u64(result128);
    }
}

impl ShiftStripeFeistelRngCore {
    fn new(seed_bytes: [u8; 32]) -> ShiftStripeFeistelRngCore {
        let permutor = u128::from_be_bytes(seed_bytes[0..16].try_into().unwrap());
        let counter = u128::from_be_bytes(seed_bytes[16..32].try_into().unwrap());
        ShiftStripeFeistelRngCore {
            permutor, counter
        }
    }

    fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeFeistelRngCore {
        let mut seed_bytes = [0u8; 32];
        rng.fill_bytes(&mut seed_bytes);
        Self::new(seed_bytes.into())
    }
}

impl ShiftStripeSimpleRngCore {
    fn new(seed_bytes: [u8; 32]) -> ShiftStripeSimpleRngCore {
        let permutor = u128::from_be_bytes(seed_bytes[0..16].try_into().unwrap());
        let counter = u128::from_be_bytes(seed_bytes[16..32].try_into().unwrap());
        ShiftStripeSimpleRngCore {
            permutor, counter
        }
    }

    fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeSimpleRngCore {
        let mut seed_bytes = [0u8; 32];
        rng.fill_bytes(&mut seed_bytes);
        Self::new(seed_bytes.into())
    }
}

struct ShiftStripeSponge {
    permutor: u128,
    state: [u8; 16]
}

impl ShiftStripeSponge {
    fn new(key: u128) -> ShiftStripeSponge {
        ShiftStripeSponge {
            permutor: key,
            state: [0u8; 16]
        }
    }

    fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeSponge {
        Self::new(random_u128(rng))
    }
}

fn random_u128<T: Rng>(rand: &mut T) -> u128 {
    let mut seed_bytes = [0u8; 16];
    rand.fill_bytes(&mut seed_bytes);
    u128::from_be_bytes(seed_bytes)
}

impl Hasher for ShiftStripeSponge {
    fn finish(&self) -> u64 {
        let base_result = u128::from_be_bytes(self.state);
        let (final_large_result, _) = shift_stripe_feistel(0, base_result, self.permutor, FIESTEL_ROUNDS_TO_DIFFUSE - 1);
        compress_u128_to_u64(final_large_result)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            let mut in_bytes = [0u8; 16];
            in_bytes[0..15].copy_from_slice(&self.state[1..16]);
            in_bytes[15] = *byte;
            let out = shift_stripe(u128::from_be_bytes(in_bytes), self.permutor);
            self.state = out.to_be_bytes();
        }
    }
}

#[test]
fn test_hashing_random_key() {
    test_hashing(thread_rng().gen())
}

#[cfg(test)]
use std::ops::Shl;

#[cfg(test)]
fn test_hashing(key: u128) {
    let expected_count = 1usize.shl(16) + 1usize.shl(8) + 1usize;
    let mut hashes = Vec::with_capacity(expected_count);
    let mut hasher = ShiftStripeSponge::new(key);
    hashes.push(hasher.finish()); // Hash of empty string
    for byte1 in 0..=u8::MAX {
        let mut hasher = ShiftStripeSponge::new(key);
        hasher.write(&[byte1]);
        hashes.push(hasher.finish());
        for byte2 in 0..=u8::MAX {
            let mut hasher = ShiftStripeSponge::new(key);
            hasher.write(&[byte1, byte2]);
            hashes.push(hasher.finish());
        }
    }
    hashes.sort();
    hashes.dedup();
    assert_eq!(hashes.len(), expected_count);
}

#[test]
fn test_hashing_zero_key() {
    test_hashing(0);
}

#[test]
fn test_diffusion_small_keys() {
    for permutor_short in -128..128 {
        for i in -128..128 {
            assert_eq!(vec![] as Vec<String>, check_diffusion_around(permutor_short, i));
        }
    }
}

#[test]
fn test_diffusion_small_keys_simple() {
    for permutor_short in -128..128 {
        for i in -128..128 {
            assert_eq!(vec![] as Vec<String>, check_diffusion_simple(permutor_short, i - 1, i));
        }
    }
}


#[test]
fn test_diffusion_random_key() {
    let permutor_short = random_u128(&mut thread_rng()) as i128;
    for i in -128..128 {
        assert_eq!(vec![] as Vec<String>, check_diffusion_around(permutor_short, i));
    }
}

fn check_diffusion_around(permutor_short: i128, i: i128) -> Vec<String> {
    let mut warnings = check_diffusion(permutor_short, 0, i - 1, 0, i);
    warnings.extend_from_slice(&check_diffusion(permutor_short, i - 1, 0, i, 0));

    warnings.extend_from_slice(&check_diffusion(permutor_short, i - 1, i, i, i));
    warnings.extend_from_slice(&check_diffusion(permutor_short, i, i - 1, i, i));
    warnings.extend_from_slice(&check_diffusion(permutor_short, i - 1, i - 1, i, i));
    warnings.extend_from_slice(&check_diffusion(permutor_short, i - 1, i, i, i));
    warnings
}

fn main() {
    let mut rng = BlockRng64::new(ShiftStripeFeedbackRngCore::new([0; 32]));
    let mut stdout = std::io::stdout();
    let mut write_result = Ok(());
    let mut out_buffer = [0u8; 1024];
    while write_result.is_ok() {
        rng.fill_bytes(&mut out_buffer);
        write_result = stdout.write_all(&out_buffer);
    }
}

fn check_diffusion(permutor_signed: i128, previn1: i128, previn2: i128, thisin1: i128, thisin2: i128) -> Vec<String> {
    let mut warnings = Vec::new();
    let permutor = permutor_signed as u128;
    let previn1_unsigned = previn1 as u128;
    let previn2_unsigned = previn2 as u128;
    let thisin1_unsigned = thisin1 as u128;
    let thisin2_unsigned = thisin2 as u128;
    let (prev1, prev2) = shift_stripe_feistel(previn1_unsigned, previn2_unsigned, permutor, FIESTEL_ROUNDS_TO_DIFFUSE);
    let (this1, this2) = shift_stripe_feistel(thisin1_unsigned, thisin2_unsigned, permutor, FIESTEL_ROUNDS_TO_DIFFUSE);
    let xor1 = this1 ^ prev1;
    let xor2 = this2 ^ prev2;
    let bits_difference_1 = xor1.count_ones();
    let bits_difference_2 = xor2.count_ones();
    if bits_difference_1 < 16 || bits_difference_1 > 112
        || bits_difference_2 < 16 || bits_difference_2 > 112
        || (bits_difference_1 + bits_difference_2) < 64
        || (bits_difference_1 + bits_difference_2) > 192 {
        warnings.push(format!("Warning: for permutor {} and inputs ({}, {}) and ({}, {}), outputs ({:#034x}, {:#034x}) and ({:#034x}, {:#034x}) differ by ({:#034x}, {:#034x})",
                              permutor_signed, previn1, previn2, thisin1, thisin2, prev1, prev2, this1, this2, xor1, xor2
        ));
    }
    warnings
}

fn check_diffusion_simple(permutor_signed: i128, previn: i128, thisin: i128) -> Vec<String> {
    let mut warnings = Vec::new();
    let permutor = permutor_signed as u128;
    let prev = shift_stripe_simple(previn as u128, permutor, SIMPLE_ROUNDS_TO_DIFFUSE);
    let this = shift_stripe_simple(thisin as u128, permutor, SIMPLE_ROUNDS_TO_DIFFUSE);
    let bits_difference = (prev ^ this).count_ones();
    if bits_difference < 16 || bits_difference > 112 {
        warnings.push(format!("Warning: for permutor {} and inputs {} and {}, outputs {:#034x} and {:#034x} differ by {} bits",
                              permutor_signed, previn, thisin, prev, this, bits_difference
        ));
    }
    warnings
}
