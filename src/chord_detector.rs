//! Chord Detector
//!
//! Real-time detection of musical chords from 12-bin chromagrams.
//!
//! Ported and modified from C++ version by Adam Stark, Queen Mary University of London.
//! https://github.com/adamstark/Chord-Detector-and-Chromagram

use std::fmt::Display;
use thiserror::Error;

const SEMITONES: usize = 12;

/// Number of chord qualities
const NUM_CHORD_KINDS: usize = 10;

/// Total number of chords (root × quality)
const NUM_CHORDS: usize = SEMITONES * NUM_CHORD_KINDS;

/// Supported chord qualities in the same order as `CHORD_SPECS` and `CHORD_INTERVALS`
const CHORD_KINDS: [ChordKind; NUM_CHORD_KINDS] = [
    ChordKind::Major,
    ChordKind::Minor,
    ChordKind::PowerFifth,
    ChordKind::DominantSeventh,
    ChordKind::MajorSeventh,
    ChordKind::MinorSeventh,
    ChordKind::Diminished,
    ChordKind::Augmented,
    ChordKind::SuspendedSecond,
    ChordKind::SuspendedFourth,
];

/// (quality, bias, number_of_intervals)
const CHORD_SPECS: &[(ChordKind, f32, f32)] = &[
    (ChordKind::Major,           1.06, 3.0),
    (ChordKind::Minor,           1.06, 3.0),
    (ChordKind::PowerFifth,      1.005, 2.0),
    (ChordKind::DominantSeventh, 1.06, 4.0),
    (ChordKind::MajorSeventh,    1.00, 4.0),
    (ChordKind::MinorSeventh,    1.06, 4.0),
    (ChordKind::Diminished,      1.05, 3.0),
    (ChordKind::Augmented,       1.055, 3.0),
    (ChordKind::SuspendedSecond, 1.0, 3.0),
    (ChordKind::SuspendedFourth, 1.0, 3.0),
];

/// Intervals (in semitones) matching `CHORD_SPECS` order
const CHORD_INTERVALS: [&[usize]; NUM_CHORD_KINDS] = [
    &[0, 4, 7],
    &[0, 3, 7],
    &[0, 7],
    &[0, 4, 7, 10],
    &[0, 4, 7, 11],
    &[0, 3, 7, 10],
    &[0, 3, 6],
    &[0, 4, 8],
    &[0, 2, 7],
    &[0, 5, 7],
];

/// A single chromagram: energy for each of the 12 semitones
type Chromagram = [f32; SEMITONES];

/// Precomputed chord profile + inverse normalizer
#[derive(Copy, Clone)]
struct PrecalcProfile {
    weights: Chromagram,
    inv_norm: f32,
}

/// Represents a musical chord detected from an audio signal.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Chord {
    /// The root note of the detected chord.
    pub root: NoteName,
    /// The quality (e.g., Major, Minor) of the detected chord.
    pub quality: ChordKind,
    /// A confidence score for the detection, where lower values indicate a better match.
    pub confidence: f32,
}

/// Supported chord qualities
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChordKind {
    /// Major chord (e.g., C-E-G)
    Major,
    /// Minor chord (e.g., C-Eb-G)
    Minor,
    /// Power chord (e.g., C-G)
    PowerFifth,
    /// Dominant seventh chord (e.g., C-E-G-Bb)
    DominantSeventh,
    /// Major seventh chord (e.g., C-E-G-B)
    MajorSeventh,
    /// Minor seventh chord (e.g., C-Eb-G-Bb)
    MinorSeventh,
    /// Diminished chord (e.g., C-Eb-Gb)
    Diminished,
    /// Augmented chord (e.g., C-E-G#)
    Augmented,
    /// Suspended second chord (e.g., C-D-G)
    SuspendedSecond,
    /// Suspended fourth chord (e.g., C-F-G)
    SuspendedFourth,
}

impl Display for ChordKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Twelve chromatic pitch classes
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NoteName {
    /// C
    C,
    /// C sharp / D flat
    Cs,
    /// D
    D,
    /// D sharp / E flat
    Ds,
    /// E
    E,
    /// F
    F,
    /// F sharp / G flat
    Fs,
    /// G
    G,
    /// G sharp / A flat
    Gs,
    /// A
    A,
    /// A sharp / B flat
    As,
    /// B
    B,
    /// Unknown note name
    Unknown,
}

impl NoteName {
    const fn from_idx(idx: usize) -> NoteName {
        match idx {
            0 => NoteName::C,
            1 => NoteName::Cs,
            2 => NoteName::D,
            3 => NoteName::Ds,
            4 => NoteName::E,
            5 => NoteName::F,
            6 => NoteName::Fs,
            7 => NoteName::G,
            8 => NoteName::Gs,
            9 => NoteName::A,
            10 => NoteName::As,
            11 => NoteName::B,
            _ => NoteName::Unknown,
        }
    }
}

/// Errors when detecting chords
#[derive(Debug, Error)]
pub enum ChordError {
    /// The chromagram provided did not have the expected number of semitones.
    #[error("expected a {expected}-bin chromagram, got {got}")]
    InvalidLength {
        /// The expected number of semitones (12).
        expected: usize,
        /// The actual number of semitones provided.
        got: usize,
    },

    /// An invalid argument was provided to a detection function.
    #[error("invalid argument `{arg}`: {msg}")]
    InvalidArgument {
        /// The name of the invalid argument.
        arg: &'static str,
        /// A description of the invalid argument.
        msg: String,
    },
}

/// Builder for `ChordDetector` to customize bleed factor
pub struct ChordDetectorBuilder {
    bleed: f32,
}

impl ChordDetectorBuilder {
    /// Create a new builder with default bleed = 0.157
    pub fn new() -> Self {
        ChordDetectorBuilder { bleed: 0.157 }
    }

    /// Set the bleed suppression factor (0.0..1.0)
    pub fn bleed(mut self, value: f32) -> Self {
        self.bleed = value;
        self
    }

    /// Build the `ChordDetector`
    pub fn build(self) -> ChordDetector {
        ChordDetector::with_bleed(self.bleed)
    }
}

impl Default for ChordDetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Main chord detector
pub struct ChordDetector {
    bleed: f32,
    chroma_work: Chromagram,
    profiles: [PrecalcProfile; NUM_CHORDS],
    scores: [f32; NUM_CHORDS],
    idx_scores: Vec<(usize, f32)>,
}

impl ChordDetector {
    /// Return a builder to customize bleed suppression factor
    pub fn builder() -> ChordDetectorBuilder {
        ChordDetectorBuilder::new()
    }

    /// Create a detector with default bleed = 0.157
    pub fn new() -> Self {
        Self::with_bleed(0.157)
    }

    /// Create a detector with a custom bleed suppression factor
    fn with_bleed(bleed: f32) -> Self {
        // Precompute profiles
        let mut profiles = [PrecalcProfile {
            weights: [0.0; SEMITONES],
            inv_norm: 0.0,
        }; NUM_CHORDS];

        for (kind_idx, &(_kind, bias, interval_count)) in CHORD_SPECS.iter().enumerate() {
            let inv = 1.0 / (((SEMITONES as f32) - interval_count) * bias);
            let intervals = CHORD_INTERVALS[kind_idx];
            for root in 0..SEMITONES {
                let base = kind_idx * SEMITONES + root;
                profiles[base].inv_norm = inv;
                for &off in intervals {
                    let note = (root + off) % SEMITONES;
                    profiles[base].weights[note] = 1.0;
                }
            }
        }

        ChordDetector {
            bleed,
            chroma_work: [0.0; SEMITONES],
            profiles,
            scores: [0.0; NUM_CHORDS],
            idx_scores: Vec::with_capacity(NUM_CHORDS),
        }
    }

    /// Detect the single best chord from a chromagram slice.
    ///
    /// Returns `Err(ChordError::InvalidLength)` if `chroma.len() != SEMITONES`.
    pub fn detect_chord(&mut self, chroma: &[f32]) -> Result<Chord, ChordError> {
        let mut list = self.top_k(chroma, 1)?;
        Ok(list.remove(0))
    }

    /// Detect the top `k` chords from a chromagram slice.
    ///
    /// Returns:
    /// - `Err(InvalidLength)` if `chroma.len() != SEMITONES`.
    /// - `Err(InvalidArgument)` if `k == 0`.
    pub fn top_k(&mut self, chroma: &[f32], k: usize) -> Result<Vec<Chord>, ChordError> {
        if chroma.len() != SEMITONES {
            return Err(ChordError::InvalidLength {
                expected: SEMITONES,
                got: chroma.len(),
            });
        }
        if k == 0 {
            return Err(ChordError::InvalidArgument {
                arg: "k",
                msg: "must be >= 1".to_string(),
            });
        }
        let choices = k.min(NUM_CHORDS);
        self.classify_chroma(chroma, choices)
    }

    /// Core pipeline returning exactly `choices` chords.
    fn classify_chroma(
        &mut self,
        chroma: &[f32],
        choices: usize,
    ) -> Result<Vec<Chord>, ChordError> {
        // 1) bleed suppression
        self.chroma_work.copy_from_slice(chroma);
        for i in 0..SEMITONES {
            let bleed_amt = self.bleed * self.chroma_work[i];
            let target = (i + SEMITONES - 5) % SEMITONES; // shift down a perfect fourth
            let reduced = (self.chroma_work[target] - bleed_amt).max(0.0);
            self.chroma_work[target] = reduced;
        }

        // 2) score each profile
        for (i, p) in self.profiles.iter().enumerate() {
            self.scores[i] = score_chord(&self.chroma_work, p);
        }

        // 3) pick top k
        self.idx_scores.clear();
        for (i, &s) in self.scores.iter().enumerate() {
            self.idx_scores.push((i, s));
        }

        // place the `choices` smallest scores in front
        self.idx_scores
            .select_nth_unstable_by(choices, |a, b| a.1.partial_cmp(&b.1).unwrap());
        // sort those front elements
        self.idx_scores[..choices]
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut result = Vec::with_capacity(choices);
        for &(idx, score) in &self.idx_scores[..choices] {
            let kind_idx = idx / SEMITONES;
            let root_idx = idx % SEMITONES;
            result.push(Chord {
                root: NoteName::from_idx(root_idx),
                quality: CHORD_KINDS[kind_idx],
                confidence: score,
            });
        }
        Ok(result)
    }
}

/// Score a single chord profile against the chromagram
#[inline(always)]
fn score_chord(chroma: &Chromagram, p: &PrecalcProfile) -> f32 {
    let mut acc = 0.0;
    (0..SEMITONES).for_each(|i| {
        let miss = 1.0 - p.weights[i];
        let c = chroma[i];
        acc += miss * (c * c);
    });
    acc.sqrt() * p.inv_norm
}

impl Default for ChordDetector {
    fn default() -> Self {
        ChordDetector::new()
    }
}
