// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow_schema::ArrowError;
use csv_core::{ReadRecordResult, Reader};

/// The estimated length of a field in bytes
const AVERAGE_FIELD_SIZE: usize = 8;

/// The minimum amount of data in a single read
const MIN_CAPACITY: usize = 1024;

/// The number of field ends buffered while discarding an oversized record
const SKIP_ENDS_CAPACITY: usize = 16;

/// [`RecordDecoder`] provides a push-based interface to decoder [`StringRecords`]
#[derive(Debug)]
pub struct RecordDecoder {
    delimiter: Reader,

    /// The expected number of fields per row
    num_columns: usize,

    /// The current line number
    line_number: usize,

    /// Offsets delimiting field start positions
    offsets: Vec<usize>,

    /// The current offset into `self.offsets`
    ///
    /// We track this independently of Vec to avoid re-zeroing memory
    offsets_len: usize,

    /// The number of fields read for the current record
    current_field: usize,

    /// The number of rows buffered
    num_rows: usize,

    /// Decoded field data
    data: Vec<u8>,

    /// Offsets into data
    ///
    /// We track this independently of Vec to avoid re-zeroing memory
    data_len: usize,

    /// Whether rows with less than expected columns are considered valid
    ///
    /// Default value is false
    /// When enabled fills in missing columns with null
    truncated_rows: bool,

    /// Whether rows with more than the expected number of columns are accepted
    /// with the extra trailing fields discarded
    ignore_extra_columns: bool,

    /// The value of `data_len` at the start of the current record
    ///
    /// `csv_core` writes field end offsets relative to the start of the record, so this is
    /// the base that must be added to them to obtain a position within `data`
    record_data_start: usize,

    /// Whether the remainder of an oversized record is currently being discarded
    ///
    /// See the `ReadRecordResult::OutputEndsFull` handling in [`Self::decode`]
    skipping: bool,

    /// Reusable output buffers used while discarding an oversized record
    ///
    /// Only allocated once an oversized record is actually encountered
    scratch_data: Vec<u8>,
    scratch_ends: Vec<usize>,

    /// The number of rows padded because they had fewer fields than expected
    ///
    /// Only incremented when `truncated_rows` is enabled. Cumulative over the
    /// lifetime of this decoder, i.e. not reset by [`Self::flush`], but reset by
    /// [`Self::clear`] along with the buffered rows it counted
    truncated_row_count: usize,

    /// The number of rows that had trailing fields discarded because they had more
    /// fields than expected
    ///
    /// Only incremented when `ignore_extra_columns` is enabled. Has the same
    /// cumulative semantics as `truncated_row_count`
    extra_column_row_count: usize,
}

impl RecordDecoder {
    pub fn new(
        delimiter: Reader,
        num_columns: usize,
        truncated_rows: bool,
        ignore_extra_columns: bool,
    ) -> Self {
        Self {
            delimiter,
            num_columns,
            line_number: 1,
            offsets: vec![],
            offsets_len: 1, // The first offset is always 0
            current_field: 0,
            data_len: 0,
            data: vec![],
            num_rows: 0,
            truncated_rows,
            // A zero column schema has no offsets to write into, so every record
            // reports `OutputEndsFull` and there is nothing to truncate to.
            // Keep the previous error behavior rather than skipping every record,
            // which would produce rows that cannot be flushed.
            ignore_extra_columns: ignore_extra_columns && num_columns > 0,
            record_data_start: 0,
            skipping: false,
            scratch_data: vec![],
            scratch_ends: vec![],
            truncated_row_count: 0,
            extra_column_row_count: 0,
        }
    }

    /// Discards fields after `num_columns` from the current record
    fn truncate_current_record(&mut self) {
        debug_assert!(self.current_field >= self.num_columns);
        // Called exactly once per oversized record: either when the record ends with too
        // many fields, or when it exhausts the offsets buffer, after which the remainder
        // is drained by the `skipping` path rather than truncated again.
        //
        // Note `current_field == num_columns` is only possible on the latter path, which
        // `csv_core` reports only for a record that has yet to be terminated, so it always
        // has at least one further field
        self.extra_column_row_count += 1;
        self.offsets_len -= self.current_field - self.num_columns;
        self.current_field = self.num_columns;
        self.data_len = self.record_data_start + self.offsets[self.offsets_len - 1];
    }

    /// Decodes records from `input` returning the number of records and bytes read
    ///
    /// Note: this expects to be called with an empty `input` to signal EOF
    pub fn decode(&mut self, input: &[u8], to_read: usize) -> Result<(usize, usize), ArrowError> {
        if to_read == 0 {
            return Ok((0, 0));
        }

        // Reserve sufficient capacity in offsets
        self.offsets
            .resize(self.offsets_len + to_read * self.num_columns, 0);

        // The current offset into `input`
        let mut input_offset = 0;

        // The number of rows decoded in this pass
        let mut read = 0;

        loop {
            if self.skipping {
                let (result, bytes_read, _, _) = self.delimiter.read_record(
                    &input[input_offset..],
                    &mut self.scratch_data,
                    &mut self.scratch_ends,
                );
                input_offset += bytes_read;

                match result {
                    ReadRecordResult::End | ReadRecordResult::InputEmpty => {
                        return Ok((read, input_offset));
                    }
                    ReadRecordResult::OutputFull | ReadRecordResult::OutputEndsFull => continue,
                    ReadRecordResult::Record => {
                        self.skipping = false;
                        read += 1;
                        self.current_field = 0;
                        self.line_number += 1;
                        self.num_rows += 1;
                        self.record_data_start = self.data_len;

                        if read == to_read || input.len() == input_offset {
                            return Ok((read, input_offset));
                        }
                        continue;
                    }
                }
            }

            // Reserve necessary space in output data based on best estimate
            let remaining_rows = to_read - read;
            let capacity = remaining_rows * self.num_columns * AVERAGE_FIELD_SIZE;
            let estimated_data = capacity.max(MIN_CAPACITY);
            self.data.resize(self.data_len + estimated_data, 0);

            // Try to read a record
            loop {
                let (result, bytes_read, bytes_written, end_positions) =
                    self.delimiter.read_record(
                        &input[input_offset..],
                        &mut self.data[self.data_len..],
                        &mut self.offsets[self.offsets_len..],
                    );

                self.current_field += end_positions;
                self.offsets_len += end_positions;
                input_offset += bytes_read;
                self.data_len += bytes_written;

                match result {
                    ReadRecordResult::End | ReadRecordResult::InputEmpty => {
                        // Reached end of input
                        return Ok((read, input_offset));
                    }
                    // Need to allocate more capacity
                    ReadRecordResult::OutputFull => break,
                    ReadRecordResult::OutputEndsFull => {
                        if self.ignore_extra_columns {
                            // The record has more fields than the entire remaining offsets
                            // buffer can hold. Keep the first `num_columns` and discard the
                            // rest of the record into the scratch buffers, so that offsets
                            // does not have to grow to fit fields that are thrown away.
                            //
                            // Note `csv_core` only reports `OutputEndsFull` if there is
                            // input left over, so a record that ends exactly here yields
                            // `Record` instead and is truncated below. Conversely, when fed
                            // small chunks the result is `InputEmpty` and this branch is
                            // never reached, in which case the extra fields simply
                            // accumulate in offsets until the record ends.
                            self.truncate_current_record();
                            self.scratch_data.resize(MIN_CAPACITY, 0);
                            self.scratch_ends.resize(SKIP_ENDS_CAPACITY, 0);
                            self.skipping = true;
                            break;
                        } else {
                            return Err(ArrowError::CsvError(format!(
                                "incorrect number of fields for line {}, expected {} got more than {}",
                                self.line_number, self.num_columns, self.current_field
                            )));
                        }
                    }
                    ReadRecordResult::Record => {
                        if self.current_field != self.num_columns {
                            if self.truncated_rows && self.current_field < self.num_columns {
                                // If the number of fields is less than expected, pad with nulls
                                let fill_count = self.num_columns - self.current_field;
                                let fill_value = self.offsets[self.offsets_len - 1];
                                self.offsets[self.offsets_len..self.offsets_len + fill_count]
                                    .fill(fill_value);
                                self.offsets_len += fill_count;
                                self.truncated_row_count += 1;
                            } else if self.ignore_extra_columns
                                && self.current_field > self.num_columns
                            {
                                self.truncate_current_record();
                            } else {
                                return Err(ArrowError::CsvError(format!(
                                    "incorrect number of fields for line {}, expected {} got {}",
                                    self.line_number, self.num_columns, self.current_field
                                )));
                            }
                        }
                        read += 1;
                        self.current_field = 0;
                        self.line_number += 1;
                        self.num_rows += 1;
                        self.record_data_start = self.data_len;

                        if read == to_read {
                            // Read sufficient rows
                            return Ok((read, input_offset));
                        }

                        if input.len() == input_offset {
                            // Input exhausted, need to read more
                            // Without this read_record will interpret the empty input
                            // byte array as indicating the end of the file
                            return Ok((read, input_offset));
                        }
                    }
                }
            }
        }
    }

    /// Returns the current number of buffered records
    pub fn len(&self) -> usize {
        self.num_rows
    }

    /// Returns true if the decoder is empty
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Returns the number of rows padded because they had fewer fields than expected
    ///
    /// Cumulative across [`Self::flush`] calls, reset by [`Self::clear`]
    pub fn truncated_row_count(&self) -> usize {
        self.truncated_row_count
    }

    /// Returns the number of rows that had trailing fields discarded because they had
    /// more fields than expected
    ///
    /// Cumulative across [`Self::flush`] calls, reset by [`Self::clear`]
    pub fn extra_column_row_count(&self) -> usize {
        self.extra_column_row_count
    }

    /// Clears the current contents of the decoder
    pub fn clear(&mut self) {
        // Preserve an in-progress record while discarding complete buffered rows
        //
        // `record_data_start` marks the start of the current record, and every field
        // decoded for it so far occupies the tail of `offsets`
        debug_assert!(self.record_data_start <= self.data_len);
        debug_assert!(self.offsets_len > self.current_field);

        let current_data_len = self.data_len - self.record_data_start;
        self.data
            .copy_within(self.record_data_start..self.data_len, 0);
        self.data_len = current_data_len;
        self.record_data_start = 0;

        let current_offsets_start = self.offsets_len - self.current_field;
        self.offsets
            .copy_within(current_offsets_start..self.offsets_len, 1);
        self.offsets_len = self.current_field + 1;
        self.num_rows = 0;
        // The rows counted so far are being discarded along with the buffered data,
        // so they must not be reported as padded or truncated
        self.truncated_row_count = 0;
        self.extra_column_row_count = 0;
    }

    /// Flushes the current contents of the reader
    pub fn flush(&mut self) -> Result<StringRecords<'_>, ArrowError> {
        // `current_field == 0` alone does not imply a record boundary: bytes of the
        // first field of the next record may already have been consumed, in which
        // case `data_len` has advanced past the start of the current record. Flushing
        // then would drop those bytes while `csv_core` still counts them, corrupting
        // the record's field end offsets
        if self.current_field != 0 || self.data_len != self.record_data_start {
            return Err(ArrowError::CsvError(
                "Cannot flush part way through record".to_string(),
            ));
        }

        // csv_core::Reader writes end offsets relative to the start of the row
        // Therefore scan through and offset these based on the cumulative row offsets
        let mut row_offset: usize = 0;
        self.offsets[1..self.offsets_len]
            .chunks_exact_mut(self.num_columns)
            .try_for_each(|row| -> Result<(), ArrowError> {
                let offset = row_offset;
                row.iter_mut().try_for_each(|x| -> Result<(), ArrowError> {
                    *x = x.checked_add(offset).ok_or_else(|| {
                        ArrowError::CsvError(
                            "CSV record offsets overflowed usize while flushing".to_string(),
                        )
                    })?;
                    row_offset = *x;
                    Ok(())
                })
            })?;

        // Need to truncate data to the actual amount of data read
        let data = std::str::from_utf8(&self.data[..self.data_len]).map_err(|e| {
            let valid_up_to = e.valid_up_to();

            // We can't use binary search because of empty fields
            let idx = self.offsets[..self.offsets_len]
                .iter()
                .rposition(|x| *x <= valid_up_to)
                .unwrap();

            let field = idx % self.num_columns + 1;
            let line_offset = self.line_number - self.num_rows;
            let line = line_offset + idx / self.num_columns;

            ArrowError::CsvError(format!(
                "Encountered invalid UTF-8 data for line {line} and field {field}"
            ))
        })?;

        let offsets = &self.offsets[..self.offsets_len];
        let num_rows = self.num_rows;

        // Reset state
        // `truncated_row_count` and `extra_column_row_count` are deliberately left alone
        // so that they accumulate across the batches produced by a single decoder
        self.offsets_len = 1;
        self.data_len = 0;
        self.num_rows = 0;
        self.record_data_start = 0;

        Ok(StringRecords {
            num_rows,
            num_columns: self.num_columns,
            offsets,
            data,
        })
    }
}

/// A collection of parsed, UTF-8 CSV records
#[derive(Debug)]
pub struct StringRecords<'a> {
    num_columns: usize,
    num_rows: usize,
    offsets: &'a [usize],
    data: &'a str,
}

impl<'a> StringRecords<'a> {
    fn get(&self, index: usize) -> StringRecord<'a> {
        let field_idx = index * self.num_columns;
        StringRecord {
            data: self.data,
            offsets: &self.offsets[field_idx..field_idx + self.num_columns + 1],
        }
    }

    pub fn len(&self) -> usize {
        self.num_rows
    }

    pub fn iter(&self) -> impl Iterator<Item = StringRecord<'a>> + '_ {
        (0..self.num_rows).map(|x| self.get(x))
    }
}

/// A single parsed, UTF-8 CSV record
#[derive(Debug, Clone, Copy)]
pub struct StringRecord<'a> {
    data: &'a str,
    offsets: &'a [usize],
}

impl<'a> StringRecord<'a> {
    pub fn get(&self, index: usize) -> &'a str {
        let end = self.offsets[index + 1];
        let start = self.offsets[index];

        // SAFETY:
        // Parsing produces offsets at valid byte boundaries
        unsafe { self.data.get_unchecked(start..end) }
    }
}

impl std::fmt::Display for StringRecord<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let num_fields = self.offsets.len() - 1;
        write!(f, "[")?;
        for i in 0..num_fields {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", self.get(i))?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::reader::records::RecordDecoder;
    use csv_core::Reader;
    use std::io::{BufRead, BufReader, Cursor};

    #[test]
    fn test_basic() {
        let csv = [
            "foo,bar,baz",
            "a,b,c",
            "12,3,5",
            "\"asda\"\"asas\",\"sdffsnsd\", as",
        ]
        .join("\n");

        let mut expected = vec![
            vec!["foo", "bar", "baz"],
            vec!["a", "b", "c"],
            vec!["12", "3", "5"],
            vec!["asda\"asas", "sdffsnsd", " as"],
        ]
        .into_iter();

        let mut reader = BufReader::with_capacity(3, Cursor::new(csv.as_bytes()));
        let mut decoder = RecordDecoder::new(Reader::new(), 3, false, false);

        loop {
            let to_read = 3;
            let mut read = 0;
            loop {
                let buf = reader.fill_buf().unwrap();
                let (records, bytes) = decoder.decode(buf, to_read - read).unwrap();

                reader.consume(bytes);
                read += records;

                if read == to_read || bytes == 0 {
                    break;
                }
            }
            if read == 0 {
                break;
            }

            let b = decoder.flush().unwrap();
            b.iter().zip(&mut expected).for_each(|(record, expected)| {
                let actual = (0..3)
                    .map(|field_idx| record.get(field_idx))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected)
            });
        }
        assert!(expected.next().is_none());
    }

    #[test]
    fn test_invalid_fields() {
        let csv = "a,b\nb,c\na\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, false);
        let err = decoder.decode(csv.as_bytes(), 4).unwrap_err().to_string();

        let expected = "Csv error: incorrect number of fields for line 3, expected 2 got 1";

        assert_eq!(err, expected);

        // Test with initial skip
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, false);
        let (skipped, bytes) = decoder.decode(csv.as_bytes(), 1).unwrap();
        assert_eq!(skipped, 1);
        decoder.clear();

        let remaining = &csv.as_bytes()[bytes..];
        let err = decoder.decode(remaining, 3).unwrap_err().to_string();
        assert_eq!(err, expected);
    }

    #[test]
    fn test_skip_insufficient_rows() {
        let csv = "a\nv\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 1, false, false);
        let (read, bytes) = decoder.decode(csv.as_bytes(), 3).unwrap();
        assert_eq!(read, 2);
        assert_eq!(bytes, csv.len());
    }

    #[test]
    fn test_truncated_rows() {
        let csv = "a,b\nv\n,1\n,2\n,3\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, true, false);
        let (read, bytes) = decoder.decode(csv.as_bytes(), 5).unwrap();
        assert_eq!(read, 5);
        assert_eq!(bytes, csv.len());
        // Only "v" is short, the rows starting with a delimiter have both fields
        assert_eq!(decoder.truncated_row_count(), 1);
    }

    #[test]
    fn test_truncated_row_count_not_reset_by_flush() {
        let csv = "a\nb,2\nc\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, true, false);

        let (read, _) = decoder.decode(csv.as_bytes(), 2).unwrap();
        assert_eq!(read, 2);
        assert_eq!(decoder.truncated_row_count(), 1);
        decoder.flush().unwrap();
        assert_eq!(decoder.truncated_row_count(), 1);

        let (read, _) = decoder.decode(&csv.as_bytes()[6..], 1).unwrap();
        assert_eq!(read, 1);
        assert_eq!(decoder.truncated_row_count(), 2);
        decoder.flush().unwrap();
        assert_eq!(decoder.truncated_row_count(), 2);
    }

    #[test]
    fn test_truncated_row_count_reset_by_clear() {
        let csv = "a\nb,2\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, true, false);

        let (read, _) = decoder.decode(csv.as_bytes(), 2).unwrap();
        assert_eq!(read, 2);
        assert_eq!(decoder.truncated_row_count(), 1);

        // The rows are discarded, so the padding done to them is discarded too
        decoder.clear();
        assert_eq!(decoder.truncated_row_count(), 0);
    }

    /// Regression test for an overflow path found by the `arrow-csv`
    /// cargo-fuzz harness being prototyped for #5332. Stages the
    /// `RecordDecoder` state directly so that rebasing the second row's
    /// end offset overflows `usize`. With the previous `*x += offset` this
    /// panicked with `attempt to add with overflow`; the patched code
    /// surfaces the condition as `ArrowError::CsvError`.
    #[test]
    fn test_flush_offset_overflow_returns_csv_error() {
        let mut decoder = RecordDecoder::new(Reader::new(), 1, false, false);
        decoder.offsets = vec![0, usize::MAX, 1];
        decoder.offsets_len = 3;
        decoder.num_rows = 2;
        let err = decoder.flush().unwrap_err();
        assert_eq!(
            err.to_string(),
            "Csv error: CSV record offsets overflowed usize while flushing"
        );
    }

    /// Flushing with bytes of the next record's first field already consumed would
    /// silently drop those bytes while `csv_core` still counts them, corrupting the
    /// field end offsets of the record they belong to
    #[test]
    fn test_flush_partial_first_field_errors() {
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, false);
        // "1" is a partial first field of the second record: no field end has been
        // written yet, so `current_field` is still 0
        let (read, bytes) = decoder.decode(b"a,b\n1", 3).unwrap();
        assert_eq!((read, bytes), (1, 5));

        let err = decoder.flush().unwrap_err().to_string();
        assert_eq!(err, "Csv error: Cannot flush part way through record");

        // Completing the record makes both it and the buffered row flushable
        let (read, bytes) = decoder.decode(b",2\n", 2).unwrap();
        assert_eq!((read, bytes), (1, 3));
        let records = decoder.flush().unwrap();
        let actual = records
            .iter()
            .map(|record| vec![record.get(0), record.get(1)])
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn test_ignore_extra_columns() {
        let csv = "a,\"b,c\",extra\n1,2,x,y,z\n3,4\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);
        let (read, bytes) = decoder.decode(csv.as_bytes(), 3).unwrap();
        assert_eq!((read, bytes), (3, csv.len()));

        let records = decoder.flush().unwrap();
        let actual = records
            .iter()
            .map(|record| vec![record.get(0), record.get(1)])
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![vec!["a", "b,c"], vec!["1", "2"], vec!["3", "4"]]
        );
    }

    #[test]
    fn test_ignore_extra_columns_byte_by_byte() {
        let csv = b"a,b,c,d,e,f\n1,2,3\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);
        // Fill the ends buffer while input remains, entering skip mode part way
        // through the first oversized record
        let (records, bytes) = decoder.decode(&csv[..9], 2).unwrap();
        assert_eq!((records, bytes), (0, 9));
        assert!(decoder.skipping);

        let mut offset = bytes;
        let mut read = 0;

        while offset < csv.len() {
            let (records, bytes) = decoder.decode(&csv[offset..offset + 1], 2 - read).unwrap();
            assert_eq!(bytes, 1);
            offset += bytes;
            read += records;
        }
        let (records, bytes) = decoder.decode(&[], 2 - read).unwrap();
        assert_eq!(bytes, 0);
        read += records;
        assert_eq!(read, 2);

        let records = decoder.flush().unwrap();
        let actual = records
            .iter()
            .map(|record| vec![record.get(0), record.get(1)])
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn test_ignore_extra_columns_output_ends_full() {
        let extras = (0..200)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let csv = format!("a,b,{extras}\nc,d\n");
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);
        let (read, bytes) = decoder.decode(csv.as_bytes(), 4).unwrap();
        assert_eq!((read, bytes), (2, csv.len()));

        let records = decoder.flush().unwrap();
        let actual = records
            .iter()
            .map(|record| vec![record.get(0), record.get(1)])
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![vec!["a", "b"], vec!["c", "d"]]);
    }

    #[test]
    fn test_ignore_extra_columns_output_ends_full_without_trailing_newline() {
        let extras = vec!["extra"; 100].join(",");
        let csv = format!("a,b,{extras}");
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);
        let (read, bytes) = decoder.decode(csv.as_bytes(), 1).unwrap();
        assert_eq!((read, bytes), (0, csv.len()));
        let (read, bytes) = decoder.decode(&[], 1).unwrap();
        assert_eq!((read, bytes), (1, 0));

        let records = decoder.flush().unwrap();
        let record = records.iter().next().unwrap();
        assert_eq!((record.get(0), record.get(1)), ("a", "b"));
    }

    #[test]
    fn test_ignore_extra_columns_across_batches() {
        let csv = b"first,row,ignored\nsecond,value,also_ignored\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);

        let (read, bytes) = decoder.decode(csv, 1).unwrap();
        assert_eq!(read, 1);
        let records = decoder.flush().unwrap();
        let first = records.iter().next().unwrap();
        assert_eq!((first.get(0), first.get(1)), ("first", "row"));

        let (read, remaining_bytes) = decoder.decode(&csv[bytes..], 1).unwrap();
        assert_eq!((read, remaining_bytes), (1, csv.len() - bytes));
        let records = decoder.flush().unwrap();
        let second = records.iter().next().unwrap();
        assert_eq!((second.get(0), second.get(1)), ("second", "value"));
    }

    #[test]
    fn test_ignore_extra_columns_composes_with_truncated_rows() {
        let csv = "a\nb,c,extra\nd,e\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, true, true);
        let (read, bytes) = decoder.decode(csv.as_bytes(), 3).unwrap();
        assert_eq!((read, bytes), (3, csv.len()));
        assert_eq!(decoder.truncated_row_count(), 1);

        let records = decoder.flush().unwrap();
        let actual = records
            .iter()
            .map(|record| vec![record.get(0), record.get(1)])
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![vec!["a", ""], vec!["b", "c"], vec!["d", "e"]]);
    }

    #[test]
    fn test_extra_columns_errors_when_disabled() {
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, false);
        let err = decoder.decode(b"a,b,c\n", 2).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Csv error: incorrect number of fields for line 1, expected 2 got 3"
        );

        let extras = vec!["extra"; 20].join(",");
        let csv = format!("a,b,{extras}\n");
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, false);
        let err = decoder.decode(csv.as_bytes(), 2).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Csv error: incorrect number of fields for line 1, expected 2 got more than 4"
        );
    }

    #[test]
    fn test_clear_preserves_partial_oversized_record() {
        let csv = b"a,b,c,d,e\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);

        for byte in csv {
            let (_, read) = decoder.decode(std::slice::from_ref(byte), 1).unwrap();
            assert_eq!(read, 1);
            decoder.clear();
        }

        assert!(!decoder.skipping);
        assert_eq!(decoder.current_field, 0);
    }

    /// A zero column schema cannot represent any record, so `ignore_extra_columns`
    /// must not turn the error into an unflushable stream of empty rows
    #[test]
    fn test_ignore_extra_columns_zero_columns_still_errors() {
        let mut decoder = RecordDecoder::new(Reader::new(), 0, false, true);
        let err = decoder.decode(b"a,b\n", 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Csv error: incorrect number of fields for line 1, expected 0 got more than 0"
        );
    }

    #[test]
    fn test_extra_column_row_count() {
        let csv = "a,b\nc,d,e\nf,g\nh,i,j,k\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);
        let (read, bytes) = decoder.decode(csv.as_bytes(), 4).unwrap();
        assert_eq!((read, bytes), (4, csv.len()));
        // Only the second and fourth rows have extra fields
        assert_eq!(decoder.extra_column_row_count(), 2);
    }

    #[test]
    fn test_extra_column_row_count_counts_oversized_record_once() {
        // Long enough to exhaust the offsets buffer and take the `skipping` path
        let extras = (0..200)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let csv = format!("a,b,{extras}\nc,d\n");
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);
        let (read, bytes) = decoder.decode(csv.as_bytes(), 4).unwrap();
        assert_eq!((read, bytes), (2, csv.len()));
        assert_eq!(decoder.extra_column_row_count(), 1);
    }

    #[test]
    fn test_extra_column_row_count_not_reset_by_flush() {
        let csv = "a,b,c\nd,e\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);

        let (read, _) = decoder.decode(csv.as_bytes(), 2).unwrap();
        assert_eq!(read, 2);
        assert_eq!(decoder.extra_column_row_count(), 1);
        decoder.flush().unwrap();
        assert_eq!(decoder.extra_column_row_count(), 1);

        let (read, _) = decoder.decode(b"f,g,h\n", 1).unwrap();
        assert_eq!(read, 1);
        assert_eq!(decoder.extra_column_row_count(), 2);
        decoder.flush().unwrap();
        assert_eq!(decoder.extra_column_row_count(), 2);
    }

    #[test]
    fn test_extra_column_row_count_reset_by_clear() {
        let csv = "a,b,c\nd,e\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, true);

        let (read, _) = decoder.decode(csv.as_bytes(), 2).unwrap();
        assert_eq!(read, 2);
        assert_eq!(decoder.extra_column_row_count(), 1);

        // The rows are discarded, so the fields discarded from them are too
        decoder.clear();
        assert_eq!(decoder.extra_column_row_count(), 0);
    }

    #[test]
    fn test_extra_column_row_count_zero_when_disabled() {
        let csv = "a,b\nc,d\n";
        let mut decoder = RecordDecoder::new(Reader::new(), 2, false, false);
        let (read, _) = decoder.decode(csv.as_bytes(), 2).unwrap();
        assert_eq!(read, 2);
        assert_eq!(decoder.extra_column_row_count(), 0);
    }
}
