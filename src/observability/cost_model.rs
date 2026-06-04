pub fn rough_token_estimate(bytes: usize) -> u64 {
    (bytes as u64).div_ceil(4)
}
