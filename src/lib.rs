//! # chord_detector
//!
//! A unified crate for real‐time audio analysis: compute 12‐bin chromagrams and
//! detect musical chords with minimal latency.
//!
//! ## Example
//! ```rust
//! use chord_detector::{Chromagram, ChordDetector};
//!
//! fn run() -> Result<(), Box<dyn std::error::Error>> {
//!     // 1) Build a chromagram pipeline
//!     let mut chroma = Chromagram::builder()
//!         .frame_size(1024)
//!         .sampling_rate(48_000)
//!         .build()?;
//!
//!     // 2) Build a chord detector
//!     let mut detector = ChordDetector::builder()
//!         .bleed(0.15)
//!         .build();
//!
//!     // 3) In your audio loop:
//!     let audio_frame: Vec<f32> = vec![0.0; 1024]; // fill with actual samples
//!     if let Some(chroma_bins) = chroma.next(&audio_frame)? {
//!         let chord = detector.detect_chord(&chroma_bins)?;
//!         println!(
//!             "Detected {:?} {:?} chord with confidence {:.3}",
//!             chord.root,
//!             chord.quality,
//!             chord.confidence
//!         );
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//! - `chromagram` (default): enables FFT‐based chromagram via `rustfft`

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rust_2018_idioms)]
#![deny(clippy::all)]

/// High‐level chord detector API.
pub use chord_detector::{
    Chord, ChordDetector, ChordDetectorBuilder, ChordError, ChordKind, NoteName,
};

/// Streaming chromagram extractor.
pub use chromagram::{Chromagram, ChromagramBuilder, ChromagramError};

/// Chromagram computation module.
pub mod chromagram;

/// Chord detection module.
pub mod chord_detector;
