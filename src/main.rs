use std::hash::Hasher;
use std::io::Write;
use std::{io};
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

// (pi * 2^62) rounded down
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

pub const FIESTEL_ROUNDS_TO_DIFFUSE: usize = 16;

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
        let permutor = 0u128.wrapping_add_signed(permutor);
        for input in -128..128 {
            let input = 0u128.wrapping_add_signed(input);
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
        permutor ^= shift_stripe(permutor, permuted_permutor);
    }
    (left, right)
}

#[derive(Copy, Clone)]
struct ShiftStripeFeistelRngCore {
    permutor: u128,
    counter: u128,
}

impl BlockRngCore for ShiftStripeFeistelRngCore {
    type Item = u64;
    type Results = [u64; 2];

    fn generate(&mut self, results: &mut Self::Results) {
        let (result1, result2) = shift_stripe_feistel(0, self.counter, self.permutor, FIESTEL_ROUNDS_TO_DIFFUSE);
        self.counter = self.counter.wrapping_add(1);
        results[0] = (result1 >> 64) as u64 ^ (result2 & (u64::MAX as u128)) as u64;
        results[1] = (result2 >> 64) as u64 ^ (result1 & (u64::MAX as u128)) as u64;
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
        let mut result = (final_large_result >> 64) as u64;
        result ^= (final_large_result & (u64::MAX as u128)) as u64;
        result
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
    let mut hashes = Vec::with_capacity(u8::MAX as usize + 1);
    for byte in 0..=u8::MAX {
        let mut hasher = ShiftStripeSponge::new_random(&mut thread_rng());
        hasher.write(&[byte]);
        hashes.push(hasher.finish());
    }
    hashes.sort();
    hashes.dedup();
    assert_eq!(hashes.len(), u8::MAX as usize + 1);
}

#[test]
fn test_hashing_zero_key() {
    let mut hashes = Vec::with_capacity(u8::MAX as usize + 1);
    for byte in 0..=u8::MAX {
        let mut hasher = ShiftStripeSponge::new(0);
        hasher.write(&[byte]);
        hashes.push(hasher.finish());
    }
    hashes.sort();
    hashes.dedup();
    assert_eq!(hashes.len(), u8::MAX as usize + 1);
}

#[test]
fn test_diffusion_small_keys() -> Result<(), String> {
    (-128i128..128).into_par_iter()
        .map(|permutor_short| {
            for i in -128..128 {
                check_diffusion_around(permutor_short, i)?;
            }
        })
        .join_all()?
}

#[test]
fn test_diffusion_random_key() -> Result<(), String> {
    let permutor_short = random_u128(&mut thread_rng()) as i128;
    for i in -128..128 {
        check_diffusion_around(permutor_short, i)?;
    }
    Ok(())
}

fn check_diffusion_around(permutor_short: i128, i: i128) -> Result<(), String> {
    check_diffusion(permutor_short, 0, i - 1, 0, i)?;
    check_diffusion(permutor_short, i - 1, 0, i, 0)?;
    check_diffusion(permutor_short, i - 1, i, i, i)?;
    check_diffusion(permutor_short, i, i - 1, i, i)?;
    check_diffusion(permutor_short, i - 1, i - 1, i, i)?;
    check_diffusion(permutor_short, i - 1, i, i, i)?;
    Ok(())
}

fn main() -> io::Result<()> {
    let mut rng = BlockRng64::new(ShiftStripeFeistelRngCore::new_random(&mut thread_rng()));
    let mut stdout = std::io::stdout();
    let mut out_buffer = [0u8; 1024];
    loop {
        rng.fill_bytes(&mut out_buffer);
        stdout.write_all(&out_buffer)?;
    }
}

fn check_diffusion(permutor_short: i128, previn1: i128, previn2: i128, thisin1: i128, thisin2: i128) -> Result<(), String> {
    let permutor = 0u128.wrapping_add_signed(permutor_short);
    let previn1_unsigned = 0u128.wrapping_add_signed(previn1);
    let previn2_unsigned = 0u128.wrapping_add_signed(previn2);
    let thisin1_unsigned = 0u128.wrapping_add_signed(thisin1);
    let thisin2_unsigned = 0u128.wrapping_add_signed(thisin2);
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
        return Err(format!("Warning: for permutor {} and inputs ({}, {}) and ({}, {}), outputs ({:#034x}, {:#034x}) and ({:#034x}, {:#034x}) differ by ({:#034x}, {:#034x})",
                 permutor_short, previn1, previn2, thisin1, thisin2, prev1, prev2, this1, this2, xor1, xor2
        ));
    }
    Ok(())
}
