pub type Word = u64;

// (pi * 1.shl(62)) computed at high precision and rounded down
pub const META_PERMUTOR: Word = 0xc90fdaa2_2168c234;

pub const STRIPE_MASKS: [Word; 8] = [
    0xaaaaaaaaaaaaaaaa,
    0xcccccccccccccccc,
    0xf0f0f0f0f0f0f0f0,
    0xff00ff00ff00ff00,
    0xffff0000ffff0000,
    0xffffffff00000000,
    Word::MAX,
    0x6996966996696996
];

// This array is padded to speed up copying; this is its real length.
pub const NUM_PRIMES: usize = 11;
pub const PRIME_ROTATION_AMOUNTS: [u8; 16] = [
    2, 3, 5, 7,
    11, 13, 17, 19,
    23, 29, 31,
    0, 0, 0, 0, 0
];

pub fn shift_stripe(input: Word, mut permutor: Word, round: u32) -> Word {
    let mut out = input;
    permutor = permutor.rotate_right(round.wrapping_add(2));
    let permutor_bytes = permutor.to_be_bytes();
    for perm_byte in permutor_bytes.into_iter() {
        let rotation_selector = round as usize + perm_byte as usize;
        out ^= STRIPE_MASKS[(perm_byte & 7) as usize];
        out ^= out.rotate_right(PRIME_ROTATION_AMOUNTS[rotation_selector % NUM_PRIMES] as u32)
            .wrapping_add(META_PERMUTOR );
        out = [out, out.swap_bytes(), out.reverse_bits()][(perm_byte % 3) as usize];
    }
    out
}
