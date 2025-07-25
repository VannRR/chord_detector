//! Chromagram
//!
//! Stream‐based 12-bin chromagram computation for real-time audio.
//!
//! Ported and modified from C++ version by Adam Stark, Queen Mary University of London.
//! https://github.com/adamstark/Chord-Detector-and-Chromagram

use std::{
    f32::consts::PI,
    sync::Arc,
};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use thiserror::Error;

const SEMITONES: usize = 12;
const BUFFER_SIZE: usize = 8192;
const CHROMA_INTERVAL: usize = BUFFER_SIZE / 2;

#[allow(clippy::large_const_arrays)]
const HAMMING_WINDOW: [f32; BUFFER_SIZE] = make_hamming_window();

/// Errors returned by the Chromagram pipeline.
#[derive(Debug, Error)]
pub enum ChromagramError {
    /// Frame received was not of the expected size.
    #[error("expected frame of length {expected}, got {got}")]
    InvalidFrameSize {
        /// The expected size of the audio frame.
        expected: usize,
        /// The actual size of the received audio frame.
        got: usize,
    },

    /// An error occurred during the configuration of the Chromagram.
    #[error("configuration error: {0}")]
    Configuration(String),
}

/// Builder for a Chromagram pipeline.
pub struct ChromagramBuilder {
    frame_size: usize,
    sampling_rate: usize,
    downsample_factor: usize,
    num_harmonics: usize,
    num_octaves: usize,
    search_width: usize,
}

impl ChromagramBuilder {
    /// Start with default parameters:
    /// frame_size = 1024, sampling_rate = 44_100,
    /// downsample_factor = 4, num_harmonics = 2,
    /// num_octaves = 2, search_width = 2.
    pub fn new() -> Self {
        ChromagramBuilder {
            frame_size: 1024,
            sampling_rate: 44_100,
            downsample_factor: 4,
            num_harmonics: 2,
            num_octaves: 2,
            search_width: 3,
        }
    }

    /// Set the frame size for audio processing.
    pub fn frame_size(mut self, size: usize) -> Self {
        self.frame_size = size;
        self
    }

    /// Set the sampling rate of the audio.
    pub fn sampling_rate(mut self, rate: usize) -> Self {
        self.sampling_rate = rate;
        self
    }

    /// Set the downsample factor for processing.
    pub fn downsample_factor(mut self, factor: usize) -> Self {
        self.downsample_factor = factor;
        self
    }

    /// Set the number of harmonics to consider for chord detection.
    pub fn num_harmonics(mut self, n: usize) -> Self {
        self.num_harmonics = n;
        self
    }

    /// Set the number of octaves to consider for chord detection.
    pub fn num_octaves(mut self, n: usize) -> Self {
        self.num_octaves = n;
        self
    }

    /// Set the search width for finding spectral peaks.
    pub fn search_width(mut self, w: usize) -> Self {
        self.search_width = w;
        self
    }

    /// Finalize and create the Chromagram.
    pub fn build(self) -> Result<Chromagram, ChromagramError> {
        if BUFFER_SIZE % self.downsample_factor != 0 {
            return Err(ChromagramError::Configuration(
                "BUFFER_SIZE must be divisible by downsample_factor".into(),
            ));
        }
        if self.frame_size == 0 {
            return Err(ChromagramError::Configuration("frame_size cannot be zero".into()));
        }

        // Prepare FFT plan once
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(BUFFER_SIZE);

        // Precompute pitch-class reference frequencies (C = 130.81Hz)
        let mut note_frequencies = [0.0; SEMITONES];
        let reference = 130.8127f32;
        for (i, freq) in note_frequencies.iter_mut().enumerate() {
            *freq = reference * 2f32.powf(i as f32 / 12.0);
        }

        Ok(Chromagram {
            buffer: vec![0.0; BUFFER_SIZE],
            head: 0,
            filtered: vec![0.0; self.frame_size / self.downsample_factor],
            fft_buffer: vec![Complex { re: 0.0, im: 0.0 }; BUFFER_SIZE],
            magnitude: vec![0.0; (BUFFER_SIZE / 2) + 1],
            chroma: [0.0; SEMITONES],
            sampling_rate: self.sampling_rate,
            frame_size: self.frame_size,
            downsample_factor: self.downsample_factor,
            num_harmonics: self.num_harmonics,
            num_octaves: self.num_octaves,
            search_width: self.search_width,
            samples_since_last: 0,
            fft,
            note_frequencies,
        })
    }
}

impl Default for ChromagramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming chromagram calculator.
pub struct Chromagram {
    buffer: Vec<f32>,
    head: usize,
    filtered: Vec<f32>,
    fft_buffer: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
    chroma: [f32; SEMITONES],
    sampling_rate: usize,
    frame_size: usize,
    downsample_factor: usize,
    num_harmonics: usize,
    num_octaves: usize,
    search_width: usize,
    samples_since_last: usize,
    fft: Arc<dyn Fft<f32>>,
    note_frequencies: [f32; SEMITONES],
}

impl Chromagram {
    /// Start customizing with a builder.
    pub fn builder() -> ChromagramBuilder {
        ChromagramBuilder::new()
    }

    /// Push one audio frame in. Returns `Ok(None)` until enough data accumulates,
    /// then `Ok(Some(chroma))` when a new chromagram is ready.
    pub fn next(&mut self, frame: &[f32]) -> Result<Option<[f32; SEMITONES]>, ChromagramError> {
        if frame.len() != self.frame_size {
            return Err(ChromagramError::InvalidFrameSize {
                expected: self.frame_size,
                got: frame.len(),
            });
        }

        self.downsample_frame(frame);

        // Write filtered samples into circular buffer
        for &s in &self.filtered {
            self.buffer[self.head] = s;
            self.head = (self.head + 1) % BUFFER_SIZE;
        }

        self.samples_since_last += self.frame_size;
        if self.samples_since_last < CHROMA_INTERVAL {
            return Ok(None);
        }
        self.samples_since_last -= CHROMA_INTERVAL;

        self.compute_spectrum();
        self.compute_chromagram();
        Ok(Some(self.chroma))
    }

    #[inline]
    fn downsample_frame(&mut self, input: &[f32]) {
        let (b0, b1, b2) = (0.2929, 0.5858, 0.2929);
        let (a1, a2) = (-0.0, 0.1716);
        let mut x1 = 0.0;
        let mut x2 = 0.0;
        let mut y1 = 0.0;
        let mut y2 = 0.0;
        let mut out = 0;

        for (i, &x0) in input.iter().enumerate() {
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1; x1 = x0;
            y2 = y1; y1 = y0;

            if i % self.downsample_factor == 0 {
                self.filtered[out] = y0;
                out += 1;
            }
        }
    }

    #[inline]
    fn compute_spectrum(&mut self) {
        // Unwrap buffer into FFT input with Hamming
        let start = (self.head + BUFFER_SIZE - CHROMA_INTERVAL) % BUFFER_SIZE;
        (0..BUFFER_SIZE).for_each(|i| {
            let sample = self.buffer[(start + i) % BUFFER_SIZE];
            self.fft_buffer[i].re = sample * HAMMING_WINDOW[i];
            self.fft_buffer[i].im = 0.0;
        });

        self.fft.process(&mut self.fft_buffer);

        for (i, mag) in self.magnitude.iter_mut().enumerate() {
            let c = &self.fft_buffer[i];
            *mag = (c.re * c.re + c.im * c.im).sqrt();
        }
    }

    #[inline]
    fn compute_chromagram(&mut self) {
        let bin_width = (self.sampling_rate as f32 / self.downsample_factor as f32)
            / BUFFER_SIZE as f32;
        let max_bin = self.magnitude.len() - 1;

        for n in 0..SEMITONES {
            let mut c_sum = 0.0;
            for octave in 1..=self.num_octaves {
                let mut note_sum = 0.0;
                for harm in 1..=self.num_harmonics {
                    let freq = self.note_frequencies[n] * octave as f32 * harm as f32;
                    let center = (freq / bin_width).round() as usize;
                    let lo = center.saturating_sub(self.search_width * harm);
                    let hi = (center + self.search_width * harm).min(max_bin);

                    let peak = self.magnitude[lo..=hi]
                        .iter()
                        .cloned()
                        .fold(0.0_f32, f32::max);
                    note_sum += peak / harm as f32;
                }
                c_sum += note_sum;
            }
            self.chroma[n] = c_sum;
        }
    }
}

/// Approximate cosine for window generation.
const fn cos_const(mut x: f32) -> f32 {
    let two_pi = 2.0 * PI;
    while x < -PI { x += two_pi; }
    while x > PI  { x -= two_pi; }
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    1.0 - x2 * 0.5 + x4 * (1.0 / 24.0)
        - x6 * (1.0 / 720.0)
        + x8 * (1.0 / 40320.0)
}

/// Compile‐time Hamming window.
const fn make_hamming_window() -> [f32; BUFFER_SIZE] {
    let mut w = [0.0; BUFFER_SIZE];
    let mut n = 0;
    while n < BUFFER_SIZE {
        let phase = 2.0 * PI * n as f32 / (BUFFER_SIZE as f32 - 1.0);
        w[n] = 0.54 - 0.46 * cos_const(phase);
        n += 1;
    }
    w
}
