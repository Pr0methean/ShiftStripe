use std::hash::Hasher;
use std::mem::size_of;
use rand::Rng;
use crate::block::{Block, block_to_bytes, bytes_to_block, compress_block_to_unit, random_block, WORDS_PER_BLOCK};
use crate::core::{META_PERMUTOR, shift_stripe};

pub struct ShiftStripeSponge {
    state: Block
}

impl ShiftStripeSponge {
    pub fn new(key: Block) -> ShiftStripeSponge {
        ShiftStripeSponge {
            state: key
        }
    }

    pub fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeSponge {
        Self::new(random_block(rng))
    }
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
                self.state[WORDS_PER_BLOCK - 1].wrapping_add(META_PERMUTOR),
                0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use core::iter::{once, repeat};
    use std::collections::BTreeMap;
    use std::hash::Hasher;
    use std::mem::size_of;
    use rand::{random, Rng, thread_rng};
    use crate::block::{Block};
    use crate::core::Word;
    use crate::hashing::ShiftStripeSponge;

    fn test_hashing<T: Iterator<Item=Box<[u8]>>>(key: Block, inputs: T) {
        // Check distribution modulo more primes than we use as rotation amounts
        const TEST_PRIMES: [u128; 14] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43, 47];

        println!("Testing hashing with key {:?}", key);
        let mut hash_reverses = BTreeMap::new();
        for input in inputs {
            let mut hasher = ShiftStripeSponge::new(key);
            hasher.write(&input);
            let hash = hasher.finish();
            hash_reverses.insert(hash, input);
        }
        let mut hash_reverses: Vec<_> = hash_reverses.into_iter().collect();
        hash_reverses.sort();
        let mut hash_mods: Vec<_> = TEST_PRIMES.iter().map(|x| vec![0u32; *x as usize]).collect();
        let mut byte_frequencies = [[0u32; u8::MAX as usize + 1]; size_of::<Block>()];
        for (value, key) in hash_reverses.into_iter() {
            for (index, byte) in key.iter().copied().enumerate() {
                byte_frequencies[index][byte as usize] += 1;
            }
            for (index, prime) in TEST_PRIMES.iter().copied().enumerate() {
                hash_mods[index][((value as u128) % prime) as usize] += 1;
            }
        }
        let mut low_p_values = 0;
        for (index, byte_frequencies) in byte_frequencies.into_iter().enumerate() {
            if byte_frequencies.iter().copied().any(|x| x != byte_frequencies[0]) {
                let (stat, p) = rv::misc::x2_test(byte_frequencies.as_slice(),
                                                  &[1.0 / ((u8::MAX as usize + 1) as f64); u8::MAX as usize + 1]);
                println!("Byte distribution for index {}: stat {}, p {:1.4}", index, stat, p);
                assert!(p >= 0.001, "p < .001; raw distribution: {:?}", byte_frequencies);
                if p < 0.05 {
                    low_p_values += 1;
                }
            } else {
                println!("Bytes for index {} are equally distributed", index);
            }
        }
        for (index, prime) in TEST_PRIMES.iter().copied().enumerate() {
            let count_per_mod = (Word::MAX as u128 + 1) / prime;
            let leftover = (Word::MAX as u128 + 1) - prime * count_per_mod;
            let prob_per_mod = (count_per_mod as f64) / ((Word::MAX as u128 + 1) as f64);
            let prob_per_mod_with_left = (count_per_mod + 1) as f64 / ((Word::MAX as u128 + 1) as f64);
            let mut probabilities: Vec<_> = repeat(prob_per_mod).take(prime as usize).collect();
            probabilities[0..(leftover as usize)].fill(prob_per_mod_with_left);
            let sum_of_probs: f64 = probabilities.iter().copied().sum();
            debug_assert!(sum_of_probs >= 1.0 - 1.0e-9);
            debug_assert!(sum_of_probs <= 1.0 + 1.0e-9);
            let (stat, p) = rv::misc::x2_test(&hash_mods[index], &probabilities);
            println!("Modulo-{} distribution: stat {}, p {:1.4}", prime, stat, p);
            assert!(p >= 0.001, "p < .001; raw distribution: {:?}", hash_mods[index]);
            if p < 0.05 {
                low_p_values += 1;
            }
        }
        assert!(low_p_values <= (size_of::<Block>() + TEST_PRIMES.len()) / 4,
                "Too many low p values");
    }

    #[test]
    fn test_hashing_zero_key() {
        test_hashing(Block::default(),
                     once([].into())
                         .chain((0..=u8::MAX).map(|x| [x].into()))
                         .chain((0..=u8::MAX).flat_map(|x| (0..=u8::MAX).map(move |y| [x, y].into())))
        );
    }

    #[test]
    fn test_hashing_random_inputs() {
        const LEN_PER_INPUT: usize = size_of::<Block>();
        const INPUT_COUNT: usize = 1 << 16;
        let mut inputs = vec![0u8; LEN_PER_INPUT * INPUT_COUNT];
        thread_rng().fill(inputs.as_mut_slice());
        let mut inputs: Vec<Box<[u8]>> = inputs.chunks(LEN_PER_INPUT).map(|x| x.to_owned().into_boxed_slice()).collect();
        inputs.sort();
        inputs.dedup();
        test_hashing(Block::default(),
                     inputs.into_iter());
    }

    #[test]
    fn test_hashing_random_inputs_and_random_key() {
        const LEN_PER_INPUT: usize = size_of::<Block>();
        const INPUT_COUNT: usize = 1 << 16;
        let mut inputs = vec![0u8; LEN_PER_INPUT * INPUT_COUNT];
        thread_rng().fill(inputs.as_mut_slice());
        let mut inputs: Vec<Box<[u8]>> = inputs.chunks(LEN_PER_INPUT).map(|x| x.to_owned().into_boxed_slice()).collect();
        inputs.sort();
        inputs.dedup();
        test_hashing(random(),
                     inputs.into_iter());
    }

    #[test]
    fn test_hashing_random_key() {
        test_hashing(random(),
                     once([].into())
                         .chain((0..=u8::MAX).map(|x| [x].into()))
                         .chain((0..=u8::MAX).flat_map(|x| (0..=u8::MAX).map(move |y| [x, y].into()))))
    }
}
