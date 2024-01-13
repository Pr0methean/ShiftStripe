use core::hash::Hasher;
use rand::{Rng};
use crate::block::{compress_block_to_unit, random_block};
use crate::core::{META_PERMUTOR, shift_stripe, shuffle_lanes, Vector, VECTOR_SIZE, Word};

#[derive(Clone, Debug)]
pub struct ShiftStripeSponge {
    first_state: Vector,
    second_state: Vector
}

impl ShiftStripeSponge {
    pub fn new(seed: [Word; 2 * VECTOR_SIZE]) -> ShiftStripeSponge {
        let mut chunks = seed.array_chunks::<VECTOR_SIZE>().copied();
        ShiftStripeSponge {
            first_state: Vector::from_array(chunks.next().unwrap()),
            second_state: Vector::from_array(chunks.next().unwrap()),
        }
    }

    pub fn from_rng<T: Rng>(rng: &mut T) -> ShiftStripeSponge {
        Self::new(random_block(rng))
    }
}

impl Hasher for ShiftStripeSponge {
    #[inline]
    fn finish(&self) -> u64 {
        compress_block_to_unit(&self.first_state) ^ compress_block_to_unit(&self.second_state)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes.iter().copied() {
            let temp_state = self.first_state.clone();
            self.first_state[VECTOR_SIZE - 1] ^= META_PERMUTOR.wrapping_mul(byte.into());
            shift_stripe(&mut self.first_state, self.second_state);
            shuffle_lanes(self.second_state);
            self.second_state ^= temp_state;
        }
    }
}

#[cfg(test)]
mod tests {
    use core::iter::{once, repeat};
    use std::collections::BTreeMap;
    use core::hash::Hasher;
    use core::mem::{size_of};
    use rand::{Rng, thread_rng};
    use crate::block::random_block;
    use crate::block::DefaultArray;
    use crate::core::{VECTOR_SIZE, Word};
    use crate::hashing::ShiftStripeSponge;

    fn test_hashing<T: Iterator<Item=Box<[u8]>>>(key: [Word; 2 * VECTOR_SIZE], inputs: T) {
        // Check distribution modulo more primes than we use as rotation amounts
        const TEST_PRIMES: [u128; 14] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43, 47];

        println!("Testing hashing with key {:016x?}", key);
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
        let histogram_size = (u8::MAX as usize + 1) * size_of::<u64>();
        let mut byte_frequencies = vec![0u32; histogram_size];
        for (hash, _input) in hash_reverses.into_iter() {
            // println!("{:02x?} -> {:016x}", _input, hash);
            for (index, byte) in hash.to_be_bytes().into_iter().enumerate() {
                byte_frequencies[(index << 8) + byte as usize] += 1;
            }
            for (index, prime) in TEST_PRIMES.iter().copied().enumerate() {
                hash_mods[index][((hash as u128) % prime) as usize] += 1;
            }
        }
        let expected_freqs = vec![1.0 / histogram_size as f64; histogram_size];
        if byte_frequencies.iter().copied().any(|x| x != byte_frequencies[0]) {
            let (stat, p) = rv::misc::x2_test(&byte_frequencies, &expected_freqs);
            println!("Distribution: stat {}, p {:1.4}", stat, p);
            if p < 0.001 {
                // try to find a culprit byte
                for byte_index in 0..size_of::<u64>() {
                    let nth_byte = &(byte_frequencies[(byte_index << 8)..((byte_index + 1) << 8)]);
                    let (stat, p) = rv::misc::x2_test(&nth_byte, &(expected_freqs[0..(1 << 8)]));
                    assert!(p >= 0.0001, "stat {}, p {} < .0001 for byte {}; raw distribution: {:?}",
                            stat, p, byte_index, nth_byte);
                }
                panic!("p < .001; raw distribution: {:?}", byte_frequencies);
            }
            assert!(p >= 0.0001, "p < .0001; raw distribution: {:?}", byte_frequencies);
        } else {
            println!("Bytes and moduli are equally distributed");
        }
        const POSSIBLE_WORDS: u128 = Word::MAX as u128 + 1;
        for (index, prime) in TEST_PRIMES.iter().copied().enumerate() {
            let count_per_mod = POSSIBLE_WORDS / prime;
            let leftover = POSSIBLE_WORDS - prime * count_per_mod;
            let prob_per_mod = (count_per_mod as f64) / (POSSIBLE_WORDS as f64);
            let prob_per_mod_with_left = (count_per_mod + 1) as f64 / (POSSIBLE_WORDS as f64);
            let mut probabilities: Vec<_> = repeat(prob_per_mod).take(prime as usize).collect();
            probabilities[0..(leftover as usize)].fill(prob_per_mod_with_left);
            let sum_of_probs: f64 = probabilities.iter().copied().sum();
            debug_assert!(sum_of_probs >= 1.0 - 1.0e-9);
            debug_assert!(sum_of_probs <= 1.0 + 1.0e-9);
            let (stat, p) = rv::misc::x2_test(&hash_mods[index], &probabilities);
            println!("Modulo-{} distribution: stat {}, p {:1.4}", prime, stat, p);
            assert!(p >= 0.0001, "p < .0001; raw distribution: {:?}", hash_mods[index]);
        }
    }

    #[test]
    fn test_hashing_zero_key() {
        test_hashing(DefaultArray::default().0,
                             once([].into())
                         .chain((0..=u8::MAX).map(|x| [x].into()))
                         .chain((0..=u8::MAX).flat_map(|x| (0..=u8::MAX).map(move |y| [x, y].into())))
        );
    }

    #[test]
    fn test_hashing_random_inputs() {
        const LEN_PER_INPUT: usize = size_of::<[Word; 2]>();
        const INPUT_COUNT: usize = 1 << 16;
        let mut inputs = vec![0u8; LEN_PER_INPUT * INPUT_COUNT];
        thread_rng().fill(inputs.as_mut_slice());
        let mut inputs: Vec<Box<[u8]>> = inputs.chunks(LEN_PER_INPUT).map(|x| x.to_owned().into_boxed_slice()).collect();
        inputs.sort();
        inputs.dedup();
        test_hashing(DefaultArray::default().0, inputs.into_iter());
    }

    #[test]
    fn test_hashing_random_inputs_and_random_key() {
        const LEN_PER_INPUT: usize = size_of::<[Word; 2]>();
        const INPUT_COUNT: usize = 1 << 16;
        let mut inputs = vec![0u8; LEN_PER_INPUT * INPUT_COUNT];
        thread_rng().fill(inputs.as_mut_slice());
        let mut inputs: Vec<Box<[u8]>> = inputs.chunks(LEN_PER_INPUT).map(|x| x.to_owned().into_boxed_slice()).collect();
        inputs.sort();
        inputs.dedup();
        test_hashing(random_block(&mut thread_rng()), inputs.into_iter());
    }

    #[test]
    fn test_hashing_random_key() {
        let key = random_block(&mut thread_rng());
        log::info!("Using key {:02x?}", key);
        test_hashing(key,
                     once([].into())
                         .chain((0..=u8::MAX).map(|x| [x].into()))
                         .chain((0..=u8::MAX).flat_map(|x| (0..=u8::MAX).map(move |y| [x, y].into()))))
    }
}
