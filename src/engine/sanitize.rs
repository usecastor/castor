//! Text stream sanitization and runaway-repetition circuit breaker.
//!
//! Provides stateful runaway-repetition detection across streamed LLM deltas,
//! combining single-character thresholding with linear-time Z-algorithm
//! periodicity scanning over arbitrary pattern lengths.

/// Characters that appear in long consecutive runs in model output (git-diff
/// '+' hunks, code, URLs, JSON, math).
const CODE_REPEAT_CHARS: &[char] = &[
    '+', '.', '/', '\\', '<', '>', '|', ':', ';', '(', ')', '[', ']', '{', '}', '\'', '"', '!',
    '?', ',', '~', '^', '&', '%', '$', '@',
];

/// Whitespace / standard markdown divider characters.
const DIVIDER_CHARS: &[char] = &['-', '=', '*', '#', ' ', '\t', '\n', '_'];

const CODE_LIMIT: usize = 1000;
const DIVIDER_LIMIT: usize = 250;
const DEFAULT_LIMIT: usize = 100;
pub const PATTERN_REPEAT_COUNT: usize = 40;
/// Minimum consecutive repeats required even for long sentences/paragraphs.
pub const MIN_PATTERN_REPEATS: usize = 6;
/// Minimum repeated span (characters) targeted by the dynamic repeat rule.
/// Short fragments clamp up to `PATTERN_REPEAT_COUNT` (40); longer sentences
/// and blocks trigger proportionally earlier (`ceil(768 / unit_chars)`).
const MIN_PATTERN_SPAN_CHARS: usize = 768;
/// Maximum rolling tail retained for arbitrary-period suffix periodicity scans.
const ROLLING_WINDOW_BYTES: usize = 16_384;

/// A detected runaway-repetition event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repetition {
    /// "character" or "pattern".
    pub r#type: String,
    /// The repeated unit.
    pub pattern: String,
    /// The run length at detection time.
    pub count: usize,
}

/// Stateful detector for runaway repetition in a streamed text stream.
///
/// Feed chunks via [`RepetitionDetector::feed`]; it tracks the trailing
/// single-character run and a rolling tail window using linear-time Z-algorithm
/// periodicity scanning across arbitrary pattern lengths.
#[derive(Debug, Default)]
pub struct RepetitionDetector {
    last_char: char,
    char_repeat_count: usize,
    rolling: String,
    rev_buf: Vec<u8>,
    z_buf: Vec<usize>,
}

impl RepetitionDetector {
    pub fn new() -> Self {
        Self::default()
    }

    fn limit_for(ch: char) -> usize {
        if CODE_REPEAT_CHARS.contains(&ch) {
            CODE_LIMIT
        } else if DIVIDER_CHARS.contains(&ch) {
            DIVIDER_LIMIT
        } else {
            DEFAULT_LIMIT
        }
    }

    /// Dynamic consecutive-repeat threshold for a repeating pattern of `unit_chars`
    /// codepoints: stays at `PATTERN_REPEAT_COUNT` (40) for short fragments and scales
    /// down to `MIN_PATTERN_REPEATS` (6) as unit length grows.
    pub fn required_repeats(unit_chars: usize) -> usize {
        MIN_PATTERN_SPAN_CHARS
            .div_ceil(unit_chars.max(1))
            .clamp(MIN_PATTERN_REPEATS, PATTERN_REPEAT_COUNT)
    }

    /// Feed a chunk of streamed text. Returns a [`Repetition`] when
    /// degenerate repetition is detected, or `None` otherwise.
    pub fn feed(&mut self, text: &str) -> Option<Repetition> {
        if text.is_empty() {
            return None;
        }

        // 1. Single-character consecutive repetition.
        for ch in text.chars() {
            if ch == self.last_char {
                self.char_repeat_count += 1;
                if self.char_repeat_count >= Self::limit_for(ch) {
                    return Some(Repetition {
                        r#type: "character".to_string(),
                        pattern: ch.to_string(),
                        count: self.char_repeat_count,
                    });
                }
            } else {
                self.last_char = ch;
                self.char_repeat_count = 1;
            }
        }

        // 2. Arbitrary-period multi-character suffix repetition (Z-algorithm over
        //    the reversed rolling tail; single O(N) pass across all unit lengths).
        self.rolling.push_str(text);
        if self.rolling.len() > ROLLING_WINDOW_BYTES {
            let mut cut = self.rolling.len() - ROLLING_WINDOW_BYTES;
            while cut < self.rolling.len() && !self.rolling.is_char_boundary(cut) {
                cut += 1;
            }
            self.rolling.drain(..cut);
        }

        self.detect_periodic_suffix()
    }

    fn detect_periodic_suffix(&mut self) -> Option<Repetition> {
        let bytes = self.rolling.as_bytes();
        let n = bytes.len();
        if n < 2 * MIN_PATTERN_REPEATS {
            return None;
        }

        self.rev_buf.clear();
        self.rev_buf.extend(bytes.iter().rev().copied());
        self.z_buf.resize(n, 0);
        self.z_buf[0] = n;
        let (mut l, mut r) = (0usize, 0usize);
        for i in 1..n {
            let mut z = if i < r {
                (r - i).min(self.z_buf[i - l])
            } else {
                0
            };
            while i + z < n && self.rev_buf[z] == self.rev_buf[i + z] {
                z += 1;
            }
            self.z_buf[i] = z;
            if i + z > r {
                l = i;
                r = i + z;
            }
        }

        let max_period = n / MIN_PATTERN_REPEATS;
        for p in 2..=max_period {
            if self.z_buf[p] < p * (MIN_PATTERN_REPEATS - 1) {
                continue;
            }
            let unit_start = n - p;
            if !self.rolling.is_char_boundary(unit_start) {
                continue;
            }
            let unit = &self.rolling[unit_start..];
            // Single-character runs are handled by step 1 with char-class limits
            // (CODE_LIMIT/DIVIDER_LIMIT/DEFAULT_LIMIT), so skip homogeneous units here.
            let mut chars = unit.chars();
            let Some(first) = chars.next() else {
                continue;
            };
            if !chars.any(|c| c != first) {
                continue;
            }
            let is_pure = unit
                .chars()
                .all(|c| CODE_REPEAT_CHARS.contains(&c) || DIVIDER_CHARS.contains(&c));
            if is_pure {
                continue;
            }
            let unit_chars = unit.chars().count();
            let count = 1 + self.z_buf[p] / p;
            if count >= Self::required_repeats(unit_chars) {
                return Some(Repetition {
                    r#type: "pattern".to_string(),
                    pattern: unit.to_string(),
                    count,
                });
            }
        }

        None
    }
}
