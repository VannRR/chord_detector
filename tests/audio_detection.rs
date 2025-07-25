//! Integration tests for pitch, chord, and analyzer detection using real audio files.

use chord_detector::{ChordDetectorBuilder, ChordKind, NoteName};
use lazy_static::lazy_static;
use lewton::inside_ogg::OggStreamReader;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use walkdir::WalkDir;

fn load_audio_mono_f32(path: &PathBuf) -> Vec<f32> {
    let file = File::open(path).expect("failed to open .ogg file");
    let mut ogg = OggStreamReader::new(file).expect("invalid Ogg/Vorbis file");
    let channels = ogg.ident_hdr.audio_channels as usize;
    let mut out = Vec::new();

    while let Some(pcm) = ogg.read_dec_packet_itl().expect("decode error") {
        for frame in pcm.chunks(channels) {
            let sum: f32 = frame.iter().map(|&s| (s as f32) / (i16::MAX as f32)).sum();

            let sample_mono = sum / (channels as f32);
            out.push(sample_mono);
        }
    }

    out
}

/// Helper to split audio into frames
fn frames_from_samples(samples: &[f32], frame_len: usize) -> Vec<&[f32]> {
    samples.chunks(frame_len).collect()
}

/// Holds parsed info from filenames like "C3_maj.ogg"
#[derive(Debug, Clone)]
struct TestFileInfo {
    filename: String,
    path: PathBuf,
    root: NoteName,
    quality: ChordKind,
}

fn get_kind(str: &str) -> ChordKind {
    match str {
        "maj" => ChordKind::Major,
        "min" => ChordKind::Minor,
        "power" => ChordKind::PowerFifth,
        "7" => ChordKind::DominantSeventh,
        "maj7" => ChordKind::MajorSeventh,
        "m7" => ChordKind::MinorSeventh,
        "dim" => ChordKind::Diminished,
        "aug" => ChordKind::Augmented,
        "sus2" => ChordKind::SuspendedSecond,
        "sus4" => ChordKind::SuspendedFourth,
        other => {
            panic!("unrecognized chord kind: `{other}`");
        }
    }
}

impl TestFileInfo {
    /// Try to parse `path` into its musical intent
    fn from_path(path: PathBuf) -> Option<Self> {
        let stem = path.file_stem()?.to_str()?;
        let mut parts = stem.split('-');
        let root_oct = parts.next()?;
        let kind = parts.collect::<Vec<_>>().join("-");

        // Parse root, e.g. "C#" or "Db"
        let note_name = &root_oct[..root_oct.len() - 1];

        let root = match note_name {
            "C" => NoteName::C,
            "C#" => NoteName::Cs,
            "D" => NoteName::D,
            "D#" => NoteName::Ds,
            "E" => NoteName::E,
            "F" => NoteName::F,
            "F#" => NoteName::Fs,
            "G" => NoteName::G,
            "G#" => NoteName::Gs,
            "A" => NoteName::A,
            "A#" => NoteName::As,
            "B" => NoteName::B,
            other => {
                eprintln!("unrecognized note name: `{other}`");
                return None;
            }
        };

        Some(TestFileInfo {
            filename: path.file_name().unwrap().to_str().unwrap().to_string(),
            path,
            root,
            quality: get_kind(&kind),
        })
    }
}

/// Gather all .ogg files under `tests/audio`
fn collect_test_files(base: &str) -> Vec<TestFileInfo> {
    WalkDir::new(base)
        .into_iter()
        .filter_map(Result::ok)
        .map(|e| e.path().to_path_buf())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("ogg"))
        .filter_map(TestFileInfo::from_path)
        .collect()
}

// Point at `tests/audio` in the workspace
const AUDIO_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/chord-samples");

lazy_static! {
    static ref TEST_FILES: Vec<TestFileInfo> = collect_test_files(AUDIO_DIR);
}

#[derive(Default, Debug)]
struct Counters {
    major: usize,
    minor: usize,
    power: usize,
    dominant_seventh: usize,
    major_seventh: usize,
    minor_seventh: usize,
    diminished: usize,
    augmented: usize,
    suspended_second: usize,
    suspended_fourth: usize,
}

impl Counters {
    fn increment(&mut self, kind: ChordKind) {
        match kind {
            ChordKind::Major => self.major += 1,
            ChordKind::Minor => self.minor += 1,
            ChordKind::PowerFifth => self.power += 1,
            ChordKind::DominantSeventh => self.dominant_seventh += 1,
            ChordKind::MajorSeventh => self.major_seventh += 1,
            ChordKind::MinorSeventh => self.minor_seventh += 1,
            ChordKind::Diminished => self.diminished += 1,
            ChordKind::Augmented => self.augmented += 1,
            ChordKind::SuspendedSecond => self.suspended_second += 1,
            ChordKind::SuspendedFourth => self.suspended_fourth += 1,
        }
    }
}

#[test]
fn test_chord_detector() {
    let sample_rate = 44_100;
    let frame_len = 4_096;

    let failures = Arc::new(Mutex::new(Vec::<(ChordKind, String)>::new()));
    let counters = Arc::new(Mutex::new(Counters::default()));

    TEST_FILES.par_iter().for_each(|tf| {
        let path = PathBuf::from("tests/audio").join(&tf.path);
        let samples = load_audio_mono_f32(&path);
        let frames = frames_from_samples(&samples, frame_len);
        let mut detector = ChordDetectorBuilder::new().build();
        let mut chromagram = chord_detector::ChromagramBuilder::new()
            .sampling_rate(sample_rate)
            .frame_size(frame_len)
            .build()
            .unwrap();

        let mut passed = false;
        let mut last_chord = None;

        for frame in frames {
            if frame.len() != frame_len {
                continue;
            }
            let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
            if rms < 0.01 {
                continue;
            }


            let chroma = chromagram.next(frame).unwrap();
            if chroma.is_none() {
                continue;
            }

            let chord = detector.detect_chord(&chroma.unwrap()).unwrap();
            last_chord = Some(chord);

            if chord.quality == tf.quality && chord.root == tf.root {
                passed = true;
                break;
            }
        }

        if !passed {
            let actual = last_chord.unwrap();
            {
                let mut ctrs = counters.lock().unwrap();
                ctrs.increment(tf.quality);
            }
            let msg = format!(
                "file:{}\nexpected = root: {:?}, quality: {:?}\nactual = root: {:?}, quality: {:?}\n",
                tf.filename, tf.root, tf.quality, actual.root, actual.quality
            );
            failures.lock().unwrap().push((tf.quality, msg));
        }
    });

    let mut failures = Arc::try_unwrap(failures).unwrap().into_inner().unwrap();
    let counters = Arc::try_unwrap(counters).unwrap().into_inner().unwrap();

    if !failures.is_empty() {
        failures.sort_by_key(|(k, _)| *k);

        panic!(
            "{} chord tests failed:\n Major={}  Minor={}  Power={}  \
             Dom7={}  Maj7={}  Min7={}  Dim={}  Aug={}  Sus2={}  Sus4={}\n\n{}",
            failures.len(),
            counters.major,
            counters.minor,
            counters.power,
            counters.dominant_seventh,
            counters.major_seventh,
            counters.minor_seventh,
            counters.diminished,
            counters.augmented,
            counters.suspended_second,
            counters.suspended_fourth,
            "wow",
            //failures
            //    .into_iter()
             //   .map(|(_, m)| m)
             //   .collect::<Vec<_>>()
             //   .join("\n")
        );
    }
}
