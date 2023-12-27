use std::iter::repeat;
use core::mem::size_of;
use rand::{Fill, Rng};
use rand::distributions::{Distribution, Standard};
use rand_core::Error;
use crate::core::{shift_stripe, Word};


#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct DefaultArray<T: Default, const N: usize>(pub(crate) [T; N]);

impl <T, const N: usize> Default for DefaultArray<T, N> where T: Default + Copy {
    fn default() -> Self {
        DefaultArray([T::default(); N])
    }
}

impl <T, const N: usize> AsMut<[T]> for DefaultArray<T, N> where T: Default {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl <T, const N: usize> AsRef<[T]> for DefaultArray<T, N> where T: Default {
    fn as_ref(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl <T, const N: usize> Fill for DefaultArray<T, N> where T: Default, Standard: Distribution<T> {
    fn try_fill<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(), Error> {
        for element in self.0.iter_mut() {
            *element = rng.gen();
        }
        Ok(())
    }
}

pub fn compress_block_to_unit<const WORDS_PER_BLOCK: usize>(block: &[Word; WORDS_PER_BLOCK]) -> Word {
    block.iter().copied().fold(0, |x, y| shift_stripe(x, y))
}

pub fn bytes_to_block<T: Iterator<Item=u8>, const WORDS_PER_BLOCK: usize>(bytes: T) -> [Word; WORDS_PER_BLOCK] {
    bytes.into_iter().array_chunks().map(Word::from_be_bytes).collect::<Vec<_>>().try_into().unwrap()
}

pub fn block_to_bytes<const WORDS_PER_BLOCK: usize>(block: [Word; WORDS_PER_BLOCK]) -> [u8; size_of::<[Word; WORDS_PER_BLOCK]>()] {
    let byte_vec: Vec<_> = block.iter().copied().flat_map(Word::to_be_bytes).collect();
    byte_vec.try_into().unwrap()
}

pub fn random_block<T: Rng, const WORDS_PER_BLOCK: usize>(rand: &mut T) -> [Word; WORDS_PER_BLOCK] {
    let mut block = DefaultArray::default();
    rand.fill(&mut block);
    block.0
}

pub fn int_to_block<const WORDS_PER_BLOCK: usize>(input: i128) -> [Word; WORDS_PER_BLOCK]
    where [(); size_of::<[Word; WORDS_PER_BLOCK]>()]: {
    let mut bytes = [0u8; size_of::<[Word; WORDS_PER_BLOCK]>()];
    let first_byte = size_of::<[Word; WORDS_PER_BLOCK]>() - size_of::<u128>();
    bytes[first_byte..].copy_from_slice(input.to_be_bytes().as_slice());
    if input < 0 {
        let sign_extend = repeat(u8::MAX).take(first_byte).collect::<Vec<_>>();
        bytes[..first_byte].copy_from_slice(&sign_extend);
    }
    bytes_to_block(bytes.into_iter())
}

#[cfg(test)]
pub(crate) fn xor_blocks<const WORDS_PER_BLOCK: usize>(block1: &[Word; WORDS_PER_BLOCK], block2: &[Word; WORDS_PER_BLOCK]) -> [Word; WORDS_PER_BLOCK] {
    let xored_vec: Vec<_> = block1.iter().zip(block2.iter()).map(|(x, y)| x ^ y).collect();
    xored_vec.try_into().unwrap()
}