use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FastPathStats {
    pub rule_id: String,
    pub hits: u64,
    pub misses: u64,
}

impl FastPathStats {
    pub fn hit_rate(&self) -> f64 {
        let denominator = self.hits + self.misses;
        if denominator == 0 {
            0.0
        } else {
            self.hits as f64 / denominator as f64
        }
    }
}
