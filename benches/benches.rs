#![feature(test)]
#![feature(generic_const_exprs)]
#[cfg(test)]
mod bench {
    extern crate test;

    use std::collections::hash_map::DefaultHasher;
    use test::Bencher;
    use rand::{thread_rng};
    use rand_core::block::BlockRng64;
    use rand_core::RngCore;
    use shift_stripe::hashing::ShiftStripeSponge;
    use shift_stripe::prng::ShiftStripeFeistelRngCore;

    #[bench]
    fn benchmark_prng(b: &mut Bencher) {
        let mut rng = BlockRng64::new(ShiftStripeFeistelRngCore::from_rng(&mut thread_rng()));
        b.iter(|| rng.next_u64());
    }

    #[bench]
    fn benchmark_prng_baseline(b: &mut Bencher) {
        let rng = &mut thread_rng();
        b.iter(|| rng.next_u64());
    }
    #[bench]
    fn benchmark_hash(b: &mut Bencher) {
        use rand::Rng;
        use core::hash::Hasher;

        let mut input = [0u64; 1024];
        thread_rng().fill(&mut input);
        let mut input = input.into_iter().map(test::black_box).cycle();
        let mut hasher = ShiftStripeSponge::from_rng(&mut thread_rng());
        b.iter(|| hasher.write(&input.next().unwrap().to_be_bytes()));
        std::hint::black_box(hasher.finish());
    }

    #[bench]
    fn benchmark_hash_baseline(b: &mut Bencher) {
        use rand::Rng;
        use core::hash::Hasher;

        let mut input = [0u64; 1024];
        thread_rng().fill(&mut input);
        let mut input = input.into_iter().map(test::black_box).cycle();
        let mut hasher = DefaultHasher::new();
        b.iter(|| hasher.write(&input.next().unwrap().to_be_bytes()));
        std::hint::black_box(hasher.finish());
    }
}
