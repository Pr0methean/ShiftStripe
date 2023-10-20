#![feature(array_chunks)]
#![feature(iter_array_chunks)]

use std::collections::{BTreeMap};
use std::hash::Hasher;
use std::io::Write;
use std::iter;
use std::iter::repeat;
use std::mem::size_of;
use rand::{random, Rng, RngCore, thread_rng};
use rand_core::block::{BlockRng64, BlockRngCore};


type Unit = u64;
pub const UNITS_PER_BLOCK: usize = 2;
type Block = [Unit; UNITS_PER_BLOCK];

// (pi * 1u64.shl(62)) computed at high precision and rounded down
pub const META_PERMUTOR: Unit = 0xc90fdaa2_2168c234;
// more bits of pi
pub const SECOND_META_PERMUTOR: Unit = 0xc4c6628b_80dc1cd1;

pub const STRIPE_MASKS: [Unit; 6] = [
    0xaaaaaaaaaaaaaaaa,
    0xcccccccccccccccc,
    0xf0f0f0f0f0f0f0f0,
    0xff00ff00ff00ff00,
    0xffff0000ffff0000,
    0xffffffff00000000
];

pub const PRIME_ROTATION_AMOUNTS: [usize; 18] = [
    2, 3, 5, 7,
    11, 13, 17, 19,
    23, 29, 31, 37,
    41, 43, 47, 53,
    59, 61
];

pub fn shift_stripe(input: Unit, mut permutor: Unit, round: u32) -> Unit {
    let mut out = input;
    permutor = permutor.rotate_right((2 + round) as u32);
    let mut permutor_bytes = permutor.to_be_bytes();
    for perm_byte in permutor_bytes.into_iter() {
        out ^= STRIPE_MASKS[(perm_byte % 6) as usize];
        out ^= out.rotate_right(PRIME_ROTATION_AMOUNTS[(((perm_byte as u32 + round) / 6) % 18) as usize] as u32).wrapping_add(META_PERMUTOR );
    }
    out
}

pub const FEISTEL_ROUNDS_TO_DIFFUSE: u32 = 4;

pub fn shift_stripe_update_key(key: &mut Block, round: u32) {
    let mut new_key = Block::default();
    for unit_index in 0..UNITS_PER_BLOCK {
        let second_unit = key[(unit_index + 1) % UNITS_PER_BLOCK];
        new_key[unit_index] = shift_stripe(
            second_unit,
            key[unit_index]
                .wrapping_add(SECOND_META_PERMUTOR)
                .rotate_right((round + unit_index as u32).count_ones() + 1)
        , 0);
        if shift_stripe(second_unit, META_PERMUTOR, unit_index as u32) & 1 != 0 {
            new_key[unit_index] = new_key[unit_index].reverse_bits();
        }
    }
    *key = new_key;
}


fn shift_stripe_feistel(mut left: Block, mut right: Block, mut permutor: Block, rounds: u32) -> (Block, Block) {
    for round in 0..rounds {
        for unit_index in 0..UNITS_PER_BLOCK {
            let new_left = right[unit_index];
            let f = shift_stripe(right[unit_index], permutor[unit_index], round);
            right[unit_index] = left[unit_index] ^ f;
            left[unit_index] = new_left;
        }
        if round != rounds - 1 {
            left.rotate_right(1);
            shift_stripe_update_key(&mut permutor, round);
        }
    }
    (left, right)
}

#[derive(Copy, Clone)]
struct ShiftStripeFeistelRngCore {
    permutor: Block,
    counter: Unit,
}

impl BlockRngCore for ShiftStripeFeistelRngCore {
    type Item = Unit;
    type Results = [Unit; 2];

    fn generate(&mut self, results: &mut Self::Results) {
        let (result0, result1) = shift_stripe_feistel(
            int_to_block(self.counter.into()),
            int_to_block(self.counter.into()),
            self.permutor,
            FEISTEL_ROUNDS_TO_DIFFUSE);
        self.counter = self.counter.wrapping_add(1);
        results[0] = compress_block_to_unit(&result0);
        results[1] = compress_block_to_unit(&result1);
    }
}

#[inline]
const fn compress_u128_to_u64(input: u128) -> u64 {
    let upper_word = (input >> 64) as u64;
    let lower_word = input as u64;
    upper_word.rotate_right(31) ^ lower_word
}

impl ShiftStripeFeistelRngCore {
    fn new(seed: Block) -> ShiftStripeFeistelRngCore {
        let permutor = seed;
        let counter = compress_block_to_unit(&seed);
        ShiftStripeFeistelRngCore {
            permutor, counter
        }
    }

    fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeFeistelRngCore {
        let mut seed_bytes = [0u8; 32];
        rng.fill_bytes(&mut seed_bytes);
        Self::new(bytes_to_block(seed_bytes.into_iter()))
    }
}


fn random_block<T: Rng>(rand: &mut T) -> Block {
    let mut block = Block::default();
    rand.fill(&mut block);
    block
}


struct ShiftStripeSponge {
    permutor: Block,
    state: Block
}

impl ShiftStripeSponge {
    fn new(key: Block) -> ShiftStripeSponge {
        ShiftStripeSponge {
            permutor: key,
            state: Block::default()
        }
    }

    fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeSponge {
        ShiftStripeSponge {
            permutor: random_block(rng),
            state: random_block(rng)
        }
    }
}

fn compress_block_to_unit(block: &Block) -> Unit {
    block.iter().copied().fold(0, |x, y| shift_stripe(x, y, 0))
}

fn bytes_to_block<T: Iterator<Item=u8>>(bytes: T) -> Block {
    bytes.into_iter().array_chunks().map(Unit::from_be_bytes).collect::<Vec<_>>().try_into().unwrap()
}

fn block_to_bytes(block: Block) -> [u8; size_of::<Block>()] {
    let byte_vec: Vec<_> = block.iter().copied().flat_map(Unit::to_be_bytes).collect();
    byte_vec.try_into().unwrap()
}

fn bytes_to_unit<T: Iterator<Item=u8>>(bytes: T) -> Unit {
    Unit::from_be_bytes(bytes.into_iter().collect::<Vec<_>>().try_into().unwrap())
}

impl Hasher for ShiftStripeSponge {
    fn finish(&self) -> u64 {
        compress_block_to_unit(&self.state)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes.iter().copied() {
            let mut state_bytes = block_to_bytes(self.state);
            state_bytes.rotate_right(1);
            state_bytes[size_of::<Block>() - 1] ^= byte;
            self.state.copy_from_slice(&bytes_to_block(state_bytes.into_iter()));
            self.state[0] ^= shift_stripe(
                self.state[1],
                self.state[UNITS_PER_BLOCK - 1].wrapping_add(META_PERMUTOR),
                0
            );
        }
    }
}

#[cfg(test)]
fn test_hashing<T: Iterator<Item=Box<[u8]>>>(key: Block, inputs: T) {
    let mut hashes = BTreeMap::new();
    for input in inputs {
        let mut hasher = ShiftStripeSponge::new(key);
        hasher.write(&input);
        let hash = hasher.finish();
        hashes.insert(hash, input);
    }
    let mut hash_reverses: Vec<_> = hashes.into_iter().collect();
    hash_reverses.sort();
    let mut byte_frequencies = [[0u32; u8::MAX as usize + 1]; size_of::<Block>()];
    for (value, key) in hash_reverses.iter() {
        //print!("{:#018x} <- ", value);
        for (index, byte) in key.iter().copied().enumerate() {
            //print!("{:02x}", byte);
            byte_frequencies[index][byte as usize] += 1;
        }
        //println!();
    }
    let mut low_p_values = 0;
    for (index, byte_frequencies) in byte_frequencies.into_iter().enumerate() {
        if byte_frequencies.iter().copied().any(|x| x != byte_frequencies[0]) {
            let (stat, p) = rv::misc::x2_test(byte_frequencies.as_slice(),
                                              &[1.0/((u8::MAX as usize + 1) as f64); u8::MAX as usize + 1]);
            println!("Byte distribution for index {}: stat {}, p {:1.4}", index, stat, p);
            assert!(p >= 0.001, "p < .001; raw distribution: {:?}", byte_frequencies);
            if p < 0.05 {
                low_p_values += 1;
            }
        } else {
            println!("Bytes for index {} are equally distributed", index);
        }
    }
    assert!(low_p_values <= size_of::<Block>() / 4, "Too many low p values");
}

#[test]
fn test_hashing_zero_key() {
    test_hashing(Block::default(),
    iter::once([].into())
        .chain((0..=u8::MAX).map(|x| [x].into()))
        .chain((0..=u8::MAX).flat_map(|x| (0..=u8::MAX).map(move |y| [x, y].into())))
    );
}

#[test]
fn test_hashing_random_inputs() {
    const LEN_PER_INPUT: usize = 16;
    const INPUT_COUNT: usize = 1 << 16;
    let mut inputs = [0u8; LEN_PER_INPUT * INPUT_COUNT];
    thread_rng().fill(inputs.as_mut());
    let mut inputs: Vec<Box<[u8]>> = inputs.chunks(LEN_PER_INPUT).map(|x| x.to_owned().into_boxed_slice()).collect();
    inputs.sort();
    inputs.dedup();
    test_hashing(Block::default(),
                 inputs.into_iter());
}

#[test]
fn test_hashing_random_key() {
    test_hashing(random(),
                 iter::once([].into())
                     .chain((0..=u8::MAX).map(|x| [x].into()))
                     .chain((0..=u8::MAX).flat_map(|x| (0..=u8::MAX).map(move |y| [x, y].into()))))
}

#[test]
fn test_diffusion_small_keys() {
    for permutor_short in -128..128 {
        for i in -128..128 {
            assert_eq!(vec![] as Vec<String>, check_diffusion_around(permutor_short, i));
        }
    }
}

fn int_to_block(input: i128) -> Block {
    let mut bytes = [0u8; size_of::<Block>()];
    let first_byte = size_of::<Block>() - 16;
    bytes[first_byte..].copy_from_slice(input.to_be_bytes().as_slice());
    if input < 0 {
        let sign_extend = repeat(u8::MAX).take(first_byte).collect::<Vec<_>>();
        bytes[..first_byte].copy_from_slice(&sign_extend);
    }
    bytes_to_block(bytes.into_iter())
}

#[cfg(test)]
fn xor_blocks(block1: &Block, block2: &Block) -> Block {
    let xored_vec: Vec<_> = block1.iter().zip(block2.iter()).map(|(x, y)| x ^ y).collect();
    xored_vec.try_into().unwrap()
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

fn main() {
    let mut rng = BlockRng64::new(ShiftStripeFeistelRngCore::new([0; UNITS_PER_BLOCK]));
    let mut stdout = std::io::stdout();
    let mut write_result = Ok(());
    let mut out_buffer = [0u8; 1024];
    while write_result.is_ok() {
        rng.fill_bytes(&mut out_buffer);
        write_result = stdout.write_all(&out_buffer);
    }
}

#[cfg(test)]
fn check_diffusion(permutor_int: i128, previn1: i128, previn2: i128, thisin1: i128, thisin2: i128) -> Vec<String> {
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