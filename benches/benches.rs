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

    macro_rules! prng_bench {
        ($num_blocks: expr) => {
            paste::item! {
                #[bench]
                fn [<benchmark_prng_ $num_blocks _blocks>] (b: &mut Bencher) {
                let mut rng = BlockRng64::new(ShiftStripeFeistelRngCore::<$num_blocks>::new_random(&mut thread_rng()));
                    b.iter(|| rng.next_u64());
                }
            }
        }
    }

    #[bench]
    fn benchmark_prng_baseline(b: &mut Bencher) {
        let rng = &mut thread_rng();
        b.iter(|| rng.next_u64());
    }

    prng_bench!(02);
    prng_bench!(03);
    prng_bench!(04);
    prng_bench!(05);
    prng_bench!(06);
    prng_bench!(07);
    prng_bench!(08);
    prng_bench!(09);
    prng_bench!(10);
    prng_bench!(11);
    prng_bench!(12);
    prng_bench!(13);
    prng_bench!(14);
    prng_bench!(15);
    prng_bench!(16);

    macro_rules! hash_bench {
        ($num_blocks: expr) => {
            paste::item! {
                #[bench]
                fn [<benchmark_hash_ $num_blocks _blocks>] (b: &mut Bencher) {
                    use rand::Rng;
                    use core::hash::Hasher;

                    let mut input = [0u64; 1024];
                    rand::thread_rng().fill(&mut input);
                    let mut input = input.into_iter().map(test::black_box).cycle();
                    let mut hasher = ShiftStripeSponge::<$num_blocks>::new_random(&mut rand::thread_rng());
                    b.iter(|| hasher.write(&input.next().unwrap().to_be_bytes()));
                    std::hint::black_box(hasher.finish());
                }
            }
        }
    }

    #[bench]
    fn benchmark_hash_baseline(b: &mut Bencher) {
        use rand::Rng;
        use core::hash::Hasher;

        let mut input = [0u64; 1024];
        rand::thread_rng().fill(&mut input);
        let mut input = input.into_iter().map(test::black_box).cycle();
        let mut hasher = DefaultHasher::new();
        b.iter(|| hasher.write(&input.next().unwrap().to_be_bytes()));
        std::hint::black_box(hasher.finish());
    }

    hash_bench!(2);
    hash_bench!(3);
    hash_bench!(4);
    hash_bench!(5);
    hash_bench!(6);
    hash_bench!(7);
    hash_bench!(8);
}
