# chord_detector

A Rust crate for real-time audio analysis: compute 12-bin chromagrams and detect musical chords with minimal latency.

## Features
- Streaming chromagram extraction via a builder API
- Real-time chord detection from 12-bin chromagrams
- Customizable bleed suppression for chord matching
- Zero-allocation in the hot path after initialization



## Credits

This library uses and modifies work from:

- **Chord Detection Algorithm**: Based on "Real-Time Chord Recognition For Live Performance" by A. M. Stark and M. D. Plumbley, ICMC 2009, Montreal. Expanded in Adam Stark's PhD thesis: "Musicians and Machines: Bridging the Semantic Gap in Live Performance", Queen Mary University of London, 2011.
    - Original implementation: [https://github.com/adamstark/Chord-Detector-and-Chromagram](https://github.com/adamstark/Chord-Detector-and-Chromagram)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
chord_detector = "0.1.0"
```

## Quick Start
```rust
use chord_detector::{Chromagram, ChordDetector};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // 1) Build a chromagram pipeline
    let mut chroma = Chromagram::builder()
        .frame_size(1024)
        .sampling_rate(48_000)
        .build()?;

    // 2) Build a chord detector
    let mut detector = ChordDetector::builder()
        .bleed(0.15)
        .build();

    // 3) In your audio loop:
    let audio_frame: Vec<f32> = vec![0.0; 1024]; // fill with actual samples
    if let Some(chroma_bins) = chroma.next(&audio_frame)? {
        let chord = detector.detect_chord(&chroma_bins)?;
        println!(
            "Detected {} {} chord with confidence {:.3}",
            chord.root,
            chord.kind,
            chord.confidence
        );
    }

    Ok(())
}
```

## API Reference

### Chromagram

Compute 12-bin chromagrams from a stream of audio frames.

#### ChromagramBuilder

- **`ChromagramBuilder::new() -> Self`**

- **`.frame_size(usize) -> Self`**
    - Set the frame size for audio processing

- **`.sampling_rate(usize) -> Self`**
    - Set the sampling rate of the audio

- **`.downsample_factor(usize) -> Self`**
    - Set the downsample factor for processing

- **`.num_harmonics(usize) -> Self`**
    - Set the number of harmonics to consider for chord detection

- **`.num_octaves(usize) -> Self`**
    - Set the number of octaves to consider for chord detection

- **`.search_width(usize) -> Self`**
    - Set the search width for finding spectral peaks

- **`.build() -> Result<Chromagram, ChromagramError>`**
    - Finalize and create the Chromagram

#### Chromagram

- **`Chromagram::builder() -> ChromagramBuilder`**
    - Start customizing with a builder

- **`chromagram.next(frame: &[f32]) -> Result<Option<[f32; 12]>, ChromagramError>`**
    - Returns `Ok(None)` until enough data accumulates (half FFT buffer)
    - Returns `Ok(Some(chroma))` when a new chromagram is ready

### ChordDetector

Match 12-bin chromagrams to chord profiles.

#### ChordDetectorBuilder

- **`ChordDetectorBuilder::new() -> Self`**
    - Create a new builder with default bleed = 0.157

- **`.bleed(f32) -> Self`**
    - Set the bleed suppression factor (0.0..1.0)

- **`.build() -> ChordDetector`**
    - Build the `ChordDetector`

#### ChordDetector

- **`ChordDetector::builder() -> ChordDetectorBuilder`**
    - Return a builder to customize bleed suppression factor

- **`ChordDetector::new() -> ChordDetector`**
    - Create a detector with default bleed = 0.157

- **`detect_chord(chroma: &[f32]) -> Result<Chord, ChordError>`**
    - Detect the single best chord from a chromagram slice.
    - Returns `Err(ChordError::InvalidLength)` if `chroma.len() != SEMITONES`.

- **`top_k(chroma: &[f32], k: usize) -> Result<Vec<Chord>, ChordError>`**
    - Detect the top `k` chords from a chromagram slice.
    - Returns:
        - `Err(InvalidLength)` if `chroma.len() != SEMITONES`.
        - `Err(InvalidArgument)` if `k == 0`.

## Data Types
```rust
pub enum NoteName {
    C, Cs, D, Ds, E, F, Fs, G, Gs, A, As, B, Unknown
}

pub enum ChordKind {
    Major,
    Minor,
    Power,
    DominantSeventh,
    MajorSeventh,
    MinorSeventh,
    Diminished,
    Augmented,
    SuspendedSecond,
    SuspendedFourth,
}

pub struct Chord {
    pub root: NoteName,
    pub kind: ChordKind,
    pub confidence: f32, // lower is a better match
}

pub enum ChromagramError { /* frame size & config errors */ }
pub enum ChordError { /* invalid length & argument errors */ }
```

## Dependencies

- **rustfft**: Fast Fourier Transform implementation
- **thiserror**: Error handling utilities

## License

This crate is licensed under **GNU General Public License v3.0**.
See the LICENSE file for full text.

## Version History
#### 0.1.0 — 2025-07-24

- Initial release

- Streaming chromagram builder and real-time chord detector

- Customizable builder APIs for both modules

## Testing with Audio Files

The integration tests use the [lewton](https://crates.io/crates/lewton) crate to load `.ogg` files from the `tests/chord-samples` directory. Ensure the test files have been created with `tests/generate_chords.py` and run tests with `cargo test`.
