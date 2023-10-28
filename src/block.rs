use std::iter::repeat;
use std::mem::size_of;
use rand::Rng;
use crate::core::{shift_stripe, Word};

// Must be at least 2
pub const WORDS_PER_BLOCK: usize = 2;

pub type Block = [Word; WORDS_PER_BLOCK];

pub fn compress_block_to_unit(block: &Block) -> Word {
    block.iter().copied().fold(0, |x, y| shift_stripe(x, y, 0))
}

pub fn bytes_to_block<T: Iterator<Item=u8>>(bytes: T) -> Block {
    bytes.into_iter().array_chunks().map(Word::from_be_bytes).collect::<Vec<_>>().try_into().unwrap()
}

pub fn block_to_bytes(block: Block) -> [u8; size_of::<Block>()] {
    let byte_vec: Vec<_> = block.iter().copied().flat_map(Word::to_be_bytes).collect();
    byte_vec.try_into().unwrap()
}

pub fn random_block<T: Rng>(rand: &mut T) -> Block {
    let mut block = Block::default();
    rand.fill(&mut block);
    block
}

pub fn int_to_block(input: i128) -> Block {
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
pub(crate) fn xor_blocks(block1: &Block, block2: &Block) -> Block {
    let xored_vec: Vec<_> = block1.iter().zip(block2.iter()).map(|(x, y)| x ^ y).collect();
    xored_vec.try_into().unwrap()
}