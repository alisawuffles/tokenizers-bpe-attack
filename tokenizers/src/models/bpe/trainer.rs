#![allow(clippy::map_entry)]

use super::{Pair, WithFirstLastIterator, Word, BPE};
use crate::parallelism::*;
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::progress::{ProgressBar, ProgressStyle};
// use regex_syntax::ast::print;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::Read;

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: HashSet<usize>,
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

struct Config {
    min_frequency: u64,
    vocab_size: usize,
    show_progress: bool,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
}

/// A `BpeTrainerBuilder` can be used to create a `BpeTrainer` with a custom
/// configuration.
pub struct BpeTrainerBuilder {
    config: Config,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                min_frequency: 0,
                vocab_size: 30000,
                show_progress: true,
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: HashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,
            },
        }
    }
}

impl BpeTrainerBuilder {
    /// Constructs a new `BpeTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.config.initial_alphabet = alphabet;
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }
    /// Set max_token_length
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            words: HashMap::new(),
        }
    }
}

/// In charge of training a `BPE` model
///
/// # Examples
///
/// ```
/// use tokenizers::tokenizer::Trainer;
/// use tokenizers::models::bpe::{BPE, BpeTrainer};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// let mut model = BPE::default();
/// let special_tokens = trainer.train(&mut model).unwrap();
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct BpeTrainer {
    /// The minimum frequency a pair must have to produce a merge operation
    pub min_frequency: u64,
    /// The target vocabulary size
    pub vocab_size: usize,
    /// Whether to show progress while training
    pub show_progress: bool,
    /// A list of special tokens that the model should know of
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// The initial alphabet we want absolutely to include. This allows to cover
    /// some characters that are not necessarily in the training set
    pub initial_alphabet: HashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to caracterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// An optional parameter to limit the max length of any single token
    pub max_token_length: Option<usize>,

    words: HashMap<String, u64>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl BpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(&self, w2id: &mut HashMap<String, u32>, id2w: &mut Vec<String>) {
        for token in &self.special_tokens {
            if !w2id.contains_key(&token.content) {
                id2w.push(token.content.to_owned());
                w2id.insert(token.content.to_owned(), (id2w.len() - 1) as u32);
            }
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn compute_alphabet(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        // Compute the alphabet from seen words
        let mut alphabet: HashMap<char, usize> = HashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                // println!("{}", c);
                alphabet
                    .entry(c)
                    .and_modify(|cnt| *cnt += *count as usize)
                    .or_insert(*count as usize);
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            alphabet
                .entry(*c)
                .and_modify(|cnt| *cnt = std::usize::MAX)
                .or_insert(std::usize::MAX);
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // Compute the number of chars to remove from the alphabet
        // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
        // will be removed
        let to_remove = self
            .limit_alphabet
            .map(|limit| {
                if alphabet.len() > limit {
                    alphabet.len() - limit
                } else {
                    0
                }
            })
            .unwrap_or(0);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet (sorted for determinism)
        kept.sort_unstable_by_key(|k| (*k.0) as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s, (id2w.len() - 1) as u32);
            }
        });
    }

    /// Tokenize words and add subwords to the vocabulary when relevant
    fn tokenize_words(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
        p: &Option<ProgressBar>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                if w2id.contains_key(&s) {
                    // Found the initial char in the authorized alphabet

                    // Add the `continuing_subword_prefix` if relevant
                    if !is_first {
                        if let Some(prefix) = &self.continuing_subword_prefix {
                            s = format!("{}{}", prefix, s);
                        }
                    }
                    // Add the `end_of_word_suffix` if relevant
                    if is_last {
                        if let Some(suffix) = &self.end_of_word_suffix {
                            s = format!("{}{}", s, suffix);
                        }
                    }

                    // Insert the new formed string if necessary
                    if !w2id.contains_key(&s) {
                        id2w.push(s.clone());
                        w2id.insert(s.clone(), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&s], 1); // We do not care about the len here
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }

        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
        p: &Option<ProgressBar>,
    ) -> (HashMap<Pair, i64>, HashMap<Pair, HashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = HashMap::new();
                let mut where_to_update: HashMap<Pair, HashSet<usize>> = HashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    if !pair_counts.contains_key(&cur_pair) {
                        pair_counts.insert(cur_pair, 0);
                    }

                    // Then update counts
                    let count = counts[i];
                    where_to_update
                        .entry(cur_pair)
                        .and_modify(|h| {
                            h.insert(i);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(i);
                            h
                        });
                    *pair_counts.get_mut(&cur_pair).unwrap() += count as i64;
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (HashMap::new(), HashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        pair_counts.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    for (k, v) in wtu {
                        where_to_update
                            .entry(k)
                            .and_modify(|set| *set = set.union(&v).copied().collect())
                            .or_insert(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    pub fn do_merge_and_count(
        &self,
        word_counts: &HashMap<String, u64>,  // these are counts of whitespace-delimited words
        model: &mut BPE,
    ) -> Result<Vec<AddedToken>> {
        // These are mappings between tokens and indices
        let mut word_to_id: HashMap<String, u32> = HashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<String> = Vec::with_capacity(self.vocab_size);
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        // Read file merges.txt
        let mut file = std::fs::File::open("merges.txt").unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let mut lines = contents.lines();
        let mut merge_order: Vec<(String, String)> = Vec::new();
        // Remove the first line
        lines.next();
        // Loop over the remaining lines
        for line in lines {
            // Line is left and right half separated by a space
            let mut split = line.split(" ");
            // Add the merge to merge_order
            merge_order.push((split.next().unwrap().to_string(), split.next().unwrap().to_string()));
        }

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary (internally modifies word_to_id and id_to_word)
        //
        println!("Step 1: Add special tokens");
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. Compute the initial alphabet (internally modifies word_to_id and id_to_word)
        //
        println!("Step 2: Compute alphabet");
        self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);
        
        // Print the length of word_to_id
        println!("Length of word_to_id: {}", word_to_id.len());

        // Print 5 elements in word_to_id
        println!("Printing 5 elements in word_to_id");
        let mut count = 0;
        for (word, id) in &word_to_id {
            println!("Word: {}, ID: {}", word, id);
            count += 1;
            if count == 10 {
                break;
            }
        }

        //
        // 3. Tokenize words: turn real words into tokens based on the initial alphabet
        //
        println!("Step 3: Tokenize words");
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &progress);
        self.finalize_progress(&progress, words.len());

        //
        // 4. Count pairs in words
        //
        println!("Step 4: Count pairs in words");
        self.update_progress(&progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);
        // Insert them in the queue
        // let mut queue = BinaryHeap::with_capacity(pair_counts.len());
        // let mut queue: BTreeMap<Pair, Merge> = BTreeMap::new();
        let mut queue: HashMap<Pair, Merge> = HashMap::new();
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            // add to queue only if pair is in merge_order
            let pair_str = (id_to_word[pair.0 as usize].clone(), id_to_word[pair.1 as usize].clone());
            if merge_order.contains(&pair_str) && count > 0 {
                let merge = Merge {
                    pair,
                    count: count as u64,
                    pos,
                };
                // add the merge to the queue
                queue.insert(pair, merge);
            }
        });
        self.finalize_progress(&progress, words.len());
        println!("Length of queue: {}", queue.len());

        // Make a copy of pair_counts called initial_pair_counts
        let mut old_pair_counts: HashMap<Pair, i64> = pair_counts.clone();
        let mut all_pair_counts: Vec<HashMap<Pair, i64>> = Vec::new();
        all_pair_counts.push(pair_counts.clone());

        // Print 5 elements of pair_counts
        println!("Size of pair_counts: {}", pair_counts.len());
        println!("Printing 5 elements of pair_counts");
        let mut count = 0;
        for (key, value) in &pair_counts {
            println!("Key: ({}, {}), Value: {}", id_to_word[key.0 as usize], id_to_word[key.1 as usize], value);
            count += 1;
            if count == 5 {
                break;
            }
        }

        //
        // 5. Do the first max_merges merges
        //
        println!("Step 5: Apply merges");
        let max_merges = 30000;
        self.update_progress(&progress, max_merges, "Compute merges");
        let mut merges: Vec<(Pair, u32)> = vec![];

        for (left, right) in &merge_order {
            // print the merge we are applying
            // println!("-------");
            // println!("Applying merge: {} + {}", left, right);

            if all_pair_counts.len() == max_merges {
                break;
            }

            // If left or right are not in word_to_id (the current vocabulary)
            // This can happen when the base vocabularies are different
            if !word_to_id.contains_key(left) || !word_to_id.contains_key(right) {
                all_pair_counts.push(HashMap::new());
                continue;
            }

            // Convert left and right into a Pair using word_to_id
            let left_id = word_to_id.get(left).unwrap();
            let right_id = word_to_id.get(right).unwrap();
            let override_pair = (*left_id, *right_id);

            if queue.is_empty() {
                break;
            }
            
            // Get the pair corresponding to override_merge from queue
            if !queue.contains_key(&override_pair) {
                // Push a new empty HashMap to all_pair_counts
                all_pair_counts.push(HashMap::new());
                continue;
            }
            
            let mut top: Merge = queue.remove(&override_pair).unwrap();
            top.count = pair_counts[&top.pair] as u64;
            // If this merge does not exist in data at all
            if top.count < 1 || self.min_frequency > top.count {
                all_pair_counts.push(HashMap::new());
                continue;
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].to_owned();

            // Build new token
            if let Some(prefix) = &self.continuing_subword_prefix {
                if part_b.starts_with(prefix) {
                    let prefix_byte_len = prefix.chars().map(|c| c.len_utf8()).sum();
                    part_b = part_b[prefix_byte_len..].to_string();
                }
            }
            let new_token = format!("{}{}", part_a, part_b);
            // implement sentencepiece-like merge.
            // if this code were to be merged, integrate a way in the python bindings to communicate this variable
            // default should be 0/None to maintain previous behavior. 16 is the spm default.

            // Insert new token if it does not already exist
            let new_token_id = word_to_id
                .get(&new_token)
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if word_to_id.get(&new_token).is_none() {
                id_to_word.push(new_token.clone());
                word_to_id.insert(new_token.clone(), new_token_id);
            }
            merges.push((top.pair, new_token_id));

            // Merge the new pair in every word
            let changes = top
                .pos
                .maybe_par_iter()
                .flat_map(|&i| {
                    let word = &words[i] as *const _ as *mut Word;
                    // We can merge each of these words in parallel here because each position
                    // can be there only once (HashSet). So this is safe.
                    unsafe {
                        // let word: &mut Word = &mut (*word);
                        (*word)
                            .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                            .into_iter()
                            .map(|c| (c, i))
                            .collect::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();

            // Introduce new formed pairs
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i64;
                pair_counts
                    .entry(pair)
                    .and_modify(|c| *c += count)
                    .or_insert(count);
                if change > 0 {
                    where_to_update
                        .entry(pair)
                        .and_modify(|h| {
                            h.insert(iw);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(iw);
                            h
                        });
                }
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                let pair_str = (id_to_word[pair.0 as usize].clone(), id_to_word[pair.1 as usize].clone());
                if merge_order.contains(&pair_str) && count > 0 {
                    queue.insert(pair, Merge {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            });

            // Detect differences from old_pair_counts and create pair_counts_diffs
            // println!("Check for differences");
            pair_counts.remove(&top.pair);  // remove current merge from pair_counts (this isn't done automatically)
            let mut pair_counts_diff: HashMap<Pair, i64> = HashMap::new();
            let keys: HashSet<&Pair> = pair_counts.keys().chain(old_pair_counts.keys()).collect();
            for k in keys {
                // if the value of k in pair_counts is different from the value of k in old_pair_counts, add it to pair_counts_diff
                if pair_counts.get(k).unwrap_or(&0) != old_pair_counts.get(k).unwrap_or(&0) {
                    pair_counts_diff.insert(*k, *pair_counts.get(k).unwrap_or(&0));
                }
            }

            old_pair_counts = pair_counts.clone();
            all_pair_counts.push(pair_counts_diff);

            if let Some(p) = &progress {
                p.inc(1);
            }
        }

        self.finalize_progress(&progress, merges.len());

        // assert that all_pair_counts has no more than max_merges elements, otherwise print the values
        assert!(all_pair_counts.len() <= max_merges, "all_pair_counts has {} elements", all_pair_counts.len());

        // Print 5 elements of pair_counts
        // println!("Size of pair_counts: {}", pair_counts.len());
        // println!("Printing 5 elements of pair_counts");
        // let mut count = 0;
        // for (key, value) in &pair_counts {
        //     println!("Key: ({}, {}), Value: {}", id_to_word[key.0 as usize], id_to_word[key.1 as usize], value);
        //     count += 1;
        //     if count == 5 {
        //         break;
        //     }
        // }

        // Transfer new vocab & options to model
        model.vocab = word_to_id;
        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        model.merges = merges
            .into_iter()
            .enumerate()
            .map(|(i, (pair, new_token_id))| (pair, (i as u32, new_token_id)))
            .collect();
        model.pair_counts = all_pair_counts;

        if let Some(prefix) = &self.continuing_subword_prefix {
            model.continuing_subword_prefix = Some(prefix.to_owned());
        } else {
            model.continuing_subword_prefix = None;
        }
        if let Some(suffix) = &self.end_of_word_suffix {
            model.end_of_word_suffix = Some(suffix.to_owned());
        } else {
            model.end_of_word_suffix = None;
        }

        Ok(self.special_tokens.clone())
    }

    pub fn do_train(
        &self,
        word_counts: &HashMap<String, u64>,  // these are counts of whitespace-delimited words
        model: &mut BPE,
    ) -> Result<Vec<AddedToken>> {
        // check if the file tokenzier_override.txt exists
        let file = std::fs::File::open("merges.txt");
        // If the file exists
        if file.is_ok() {
            println!("Calling do_merge_and_count()");
            return self.do_merge_and_count(word_counts, model);
        } else {
            println!("Calling do_train_original()");
            return self.do_train_original(word_counts, model);
        }
    }

    pub fn do_train_original(
        &self,
        word_counts: &HashMap<String, u64>,  // these are counts of whitespace-delimited words
        model: &mut BPE,
    ) -> Result<Vec<AddedToken>> {
        // These are mappings between tokens and indices
        let mut word_to_id: HashMap<String, u32> = HashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<String> = Vec::with_capacity(self.vocab_size);
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary (internally modifies word_to_id and id_to_word)
        //
        println!("Step 1: Add special tokens");
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. Compute the initial alphabet (internally modifies word_to_id and id_to_word)
        //
        println!("Step 2: Compute alphabet");
        self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);
        
        // Print the length of word_to_id
        println!("Length of word_to_id: {}", word_to_id.len());

        // Print 5 random elements in word_to_id
        println!("Printing 5 random elements in word_to_id");
        let mut count = 0;
        for (word, id) in &word_to_id {
            println!("Word: {}, ID: {}", word, id);
            count += 1;
            if count == 10 {
                break;
            }
        }

        //
        // 3. Tokenize words: turn real words into tokens based on the initial alphabet
        //
        println!("Step 3: Tokenize words");
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &progress);
        self.finalize_progress(&progress, words.len());

        //
        // 4. Count pairs in words
        //
        println!("Step 4: Count pairs");
        self.update_progress(&progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);
        // Insert them in the queue
        let mut queue = BinaryHeap::with_capacity(pair_counts.len());
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        });
        self.finalize_progress(&progress, words.len());

        //
        // 5. Do merges
        //
        println!("Step 5: Do merges");
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        let mut merges: Vec<(Pair, u32)> = vec![];
        loop {
            // Stop as soon as we have a big enough vocabulary
            if word_to_id.len() >= self.vocab_size {
                break;
            }

            if queue.is_empty() {
                break;
            }

            let mut top: Merge = queue.pop().unwrap();
            if top.count != pair_counts[&top.pair] as u64 {
                top.count = pair_counts[&top.pair] as u64;
                queue.push(top);
                continue;
            }

            if top.count < 1 || self.min_frequency > top.count {
                break;
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].to_owned();

            // Build new token
            if let Some(prefix) = &self.continuing_subword_prefix {
                if part_b.starts_with(prefix) {
                    let prefix_byte_len = prefix.chars().map(|c| c.len_utf8()).sum();
                    part_b = part_b[prefix_byte_len..].to_string();
                }
            }
            let new_token = format!("{}{}", part_a, part_b);
            // implement sentencepiece-like merge.
            // if this code were to be merged, integrate a way in the python bindings to communicate this variable
            // default should be 0/None to maintain previous behavior. 16 is the spm default.

            // Insert new token if it does not already exist
            let new_token_id = word_to_id
                .get(&new_token)
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if word_to_id.get(&new_token).is_none() {
                id_to_word.push(new_token.clone());
                word_to_id.insert(new_token.clone(), new_token_id);
            }
            merges.push((top.pair, new_token_id));

            // Merge the new pair in every words
            let changes = top
                .pos
                .maybe_par_iter()
                .flat_map(|&i| {
                    let word = &words[i] as *const _ as *mut Word;
                    // We can merge each of these words in parallel here because each position
                    // can be there only once (HashSet). So this is safe.
                    unsafe {
                        // let word: &mut Word = &mut (*word);
                        (*word)
                            .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                            .into_iter()
                            .map(|c| (c, i))
                            .collect::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();

            // Introduce new formed pairs
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i64;
                pair_counts
                    .entry(pair)
                    .and_modify(|c| *c += count)
                    .or_insert(count);
                if change > 0 {
                    where_to_update
                        .entry(pair)
                        .and_modify(|h| {
                            h.insert(iw);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(iw);
                            h
                        });
                }
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Merge {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            });

            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        self.finalize_progress(&progress, merges.len());

        // Transfer new vocab & options to model
        model.vocab = word_to_id;
        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        model.merges = merges
            .into_iter()
            .enumerate()
            .map(|(i, (pair, new_token_id))| (pair, (i as u32, new_token_id)))
            .collect();
        // model.pair_counts = initial_pair_counts;

        if let Some(prefix) = &self.continuing_subword_prefix {
            model.continuing_subword_prefix = Some(prefix.to_owned());
        } else {
            model.continuing_subword_prefix = None;
        }
        if let Some(suffix) = &self.end_of_word_suffix {
            model.end_of_word_suffix = Some(suffix.to_owned());
        } else {
            model.end_of_word_suffix = None;
        }

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for BpeTrainer {
    type Model = BPE;

    /// Train a BPE model
    fn train(&self, model: &mut BPE) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }
}
