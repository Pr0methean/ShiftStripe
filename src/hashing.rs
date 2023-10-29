use core::hash::Hasher;
use core::mem::size_of;
use rand::{Rng};
use crate::block::{block_to_bytes, bytes_to_block, compress_block_to_unit, random_block};
use crate::core::{META_PERMUTOR, shift_stripe, Word};

#[derive(Clone, Debug)]
pub struct ShiftStripeSponge<const WORDS_PER_BLOCK: usize> {
    state: [Word; WORDS_PER_BLOCK]
}

impl <const WORDS_PER_BLOCK: usize> ShiftStripeSponge<WORDS_PER_BLOCK> {
    pub fn new(key: [Word; WORDS_PER_BLOCK]) -> ShiftStripeSponge<WORDS_PER_BLOCK> {
        ShiftStripeSponge {
            state: key
        }
    }
}

impl <const WORDS_PER_BLOCK: usize> ShiftStripeSponge<WORDS_PER_BLOCK> {
    pub fn new_random<T: Rng>(rng: &mut T) -> ShiftStripeSponge<WORDS_PER_BLOCK> {
        Self::new(random_block(rng))
    }
}

impl <const WORDS_PER_BLOCK: usize> Hasher for ShiftStripeSponge<WORDS_PER_BLOCK>
    where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: {
    fn finish(&self) -> u64 {
        compress_block_to_unit(&self.state)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes.iter().copied() {
            let mut state_bytes = block_to_bytes(self.state);
            state_bytes.rotate_right(1);
            state_bytes[size_of::<[Word; WORDS_PER_BLOCK]>() - 1] ^= byte;
            self.state.copy_from_slice(&bytes_to_block::<_, WORDS_PER_BLOCK>(state_bytes.into_iter()));
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
    use core::hash::Hasher;
    use core::mem::{size_of};
    use rand::{Rng, thread_rng};
    use crate::block::random_block;
    use crate::block::DefaultArray;
    use crate::core::Word;
    use crate::hashing::ShiftStripeSponge;

    fn test_hashing<T: Iterator<Item=Box<[u8]>>, const WORDS_PER_BLOCK: usize>(key: [Word; WORDS_PER_BLOCK], inputs: T)
    where ShiftStripeSponge<WORDS_PER_BLOCK> : Hasher,
        [(); (u8::MAX as usize + 1) * size_of::<[Word; WORDS_PER_BLOCK]>()]: ,
        [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: {
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
        let histogram_size = (u8::MAX as usize + 1) * size_of::<u64>();
        let mut byte_frequencies = vec![0u32; histogram_size];
        for (value, _) in hash_reverses.into_iter() {
            for (index, byte) in value.to_be_bytes().into_iter().enumerate() {
                byte_frequencies[(index << 8) + byte as usize] += 1;
            }
            for (index, prime) in TEST_PRIMES.iter().copied().enumerate() {
                hash_mods[index][((value as u128) % prime) as usize] += 1;
            }
        }
        let expected_freqs = vec![1.0 / histogram_size as f64; histogram_size];
        let (stat, p) = rv::misc::x2_test(&byte_frequencies, &expected_freqs);
        println!("Distribution: stat {}, p {:1.4}", stat, p);
        assert!(p >= 0.01, "p < .001; raw distribution: {:?}", byte_frequencies);
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
        }
        if byte_frequencies.iter().copied().any(|x| x != byte_frequencies[0]) {
            let (stat, p) = rv::misc::x2_test(&byte_frequencies, &expected_freqs);
            println!("Distribution: stat {}, p {:1.4}", stat, p);
            assert!(p >= 0.01, "p < .001; raw distribution: {:?}", byte_frequencies);
        } else {
            println!("Bytes and moduli are equally distributed");
        }
    }

    fn test_hashing_zero_key<const WORDS_PER_BLOCK: usize>()
        where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: ,
            [(); (u8::MAX as usize + 1) * size_of::<[Word; WORDS_PER_BLOCK]>()]: {
        test_hashing::<_, WORDS_PER_BLOCK>(DefaultArray::default().0,
                             once([].into())
                         .chain((0..=u8::MAX).map(|x| [x].into()))
                         .chain((0..=u8::MAX).flat_map(|x| (0..=u8::MAX).map(move |y| [x, y].into())))
        );
    }

    fn test_hashing_random_inputs<const WORDS_PER_BLOCK: usize>()
        where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]:,
        [(); (u8::MAX as usize + 1) * size_of::<[Word; WORDS_PER_BLOCK]>()]:{
        const LEN_PER_INPUT: usize = size_of::<[Word; 2]>();
        const INPUT_COUNT: usize = 1 << 16;
        let mut inputs = vec![0u8; LEN_PER_INPUT * INPUT_COUNT];
        thread_rng().fill(inputs.as_mut_slice());
        let mut inputs: Vec<Box<[u8]>> = inputs.chunks(LEN_PER_INPUT).map(|x| x.to_owned().into_boxed_slice()).collect();
        inputs.sort();
        inputs.dedup();
        test_hashing::<_, WORDS_PER_BLOCK>(DefaultArray::default().0,
                             inputs.into_iter());
    }

    fn test_hashing_random_inputs_and_random_key<const WORDS_PER_BLOCK: usize>()
            where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: ,
            [(); (u8::MAX as usize + 1) * size_of::<[Word; WORDS_PER_BLOCK]>()]: {
        const LEN_PER_INPUT: usize = size_of::<[Word; 2]>();
        const INPUT_COUNT: usize = 1 << 16;
        let mut inputs = vec![0u8; LEN_PER_INPUT * INPUT_COUNT];
        thread_rng().fill(inputs.as_mut_slice());
        let mut inputs: Vec<Box<[u8]>> = inputs.chunks(LEN_PER_INPUT).map(|x| x.to_owned().into_boxed_slice()).collect();
        inputs.sort();
        inputs.dedup();
        test_hashing::<_, WORDS_PER_BLOCK>(random_block(&mut thread_rng()),
                     inputs.into_iter());
    }

    fn test_hashing_random_key<const WORDS_PER_BLOCK: usize>()
        where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: ,
        [(); (u8::MAX as usize + 1) * size_of::<[Word; WORDS_PER_BLOCK]>()]: {
        let key = random_block(&mut thread_rng());
        log::info!("Using key {:02x?}", key);
        test_hashing::<_, WORDS_PER_BLOCK>(key,
                     once([].into())
                         .chain((0..=u8::MAX).map(|x| [x].into()))
                         .chain((0..=u8::MAX).flat_map(|x| (0..=u8::MAX).map(move |y| [x, y].into()))))
    }

    macro_rules! parameterize_hashing_test {
        ($func: ident, $num_blocks: expr) => {
            paste::item!{
                #[test]
                fn [< test_hashing_ $num_blocks _blocks_ $func >] () {
                    $func::< $num_blocks >();
                }
            }
        }
    }

    macro_rules! hashing_test_suite {
        ($num_blocks: expr) => {
            parameterize_hashing_test!(test_hashing_zero_key, $num_blocks);
            parameterize_hashing_test!(test_hashing_random_inputs, $num_blocks);
            parameterize_hashing_test!(test_hashing_random_key, $num_blocks);
            parameterize_hashing_test!(test_hashing_random_inputs_and_random_key, $num_blocks);
        };
    }

    hashing_test_suite!(2);
    hashing_test_suite!(3);
    hashing_test_suite!(4);
    hashing_test_suite!(5);
    hashing_test_suite!(6);
    hashing_test_suite!(7);
    hashing_test_suite!(8);
}
