use std::iter::repeat;
use core::mem::size_of;
use std::ops::BitXor;
use rand::{Fill, Rng};
use rand::distributions::{Distribution, Standard};
use rand_core::Error;
use crate::core::{Vector, Word};


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
    #[inline]
    fn try_fill<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(), Error> {
        for element in self.0.iter_mut() {
            *element = rng.gen();
        }
        Ok(())
    }
}

#[inline]
pub fn compress_block_to_unit(block: &Vector) -> Word {
    block.as_array().iter().copied().fold(0, Word::bitxor)
}

#[inline]
pub fn bytes_to_vector<T: Iterator<Item=u8>>(bytes: T) -> Vector {
    bytes.into_iter().array_chunks().map(Word::from_be_bytes).collect()
}

#[inline]
pub fn block_to_bytes(block: Vector) -> [u8; size_of::<Vector>()] {
    let byte_vec: Vec<_> = block.iter().copied().flat_map(Word::to_be_bytes).collect();
    byte_vec.try_into().unwrap()
}

#[inline]
pub fn random_block<T: Rng, const WORDS_PER_BLOCK: usize>(rand: &mut T) -> [Word; WORDS_PER_BLOCK] {
    let mut block = DefaultArray::default();
    rand.fill(&mut block);
    block.0
}

#[inline]
pub fn int_to_vector(input: i128) -> Vector {
    let mut bytes = [0u8; size_of::<Vector>()];
    let first_byte = size_of::<Vector>() - size_of::<i128>();
    bytes[first_byte..].copy_from_slice(input.to_be_bytes().as_slice());
    if input < 0 {
        let sign_extend = repeat(u8::MAX).take(first_byte).collect::<Vec<_>>();
        bytes[..first_byte].copy_from_slice(&sign_extend);
    }
    bytes_to_vector(bytes.into_iter())
}

#[inline]
pub(crate) fn xor_blocks<const WORDS_PER_BLOCK: usize>(block1: &[Word; WORDS_PER_BLOCK], block2: &[Word; WORDS_PER_BLOCK]) -> [Word; WORDS_PER_BLOCK] {
    let xored_vec: Vec<_> = block1.iter().zip(block2.iter()).map(|(x, y)| x ^ y).collect();
    xored_vec.try_into().unwrap()
}