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

//! [`object_store`] integration for the async Avro writer.

use crate::writer::async_writer::AsyncFileWriter;
use arrow_schema::ArrowError;
use bytes::Bytes;
use futures::future::BoxFuture;
use object_store::ObjectStore;
use object_store::buffered::BufWriter;
use object_store::path::Path;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// An [`AsyncFileWriter`] that writes Avro output to an [`ObjectStore`] location
/// via multipart upload.
///
/// This is the Avro counterpart of `parquet::arrow::async_writer::ParquetObjectWriter`
/// and is intended to be paired with [`AsyncAvroWriter`](crate::writer::AsyncAvroWriter)
/// (or the lower-level [`AsyncWriter`](crate::writer::AsyncWriter)) to stream Avro
/// Object Container Files directly to S3, GCS, Azure, or any other [`ObjectStore`]
/// implementation, without buffering the entire file in memory.
///
/// # Example
///
/// ```
/// # use arrow_array::{ArrayRef, Int64Array, RecordBatch};
/// # use arrow_avro::reader::ReaderBuilder;
/// # use arrow_avro::writer::AsyncAvroWriter;
/// # use arrow_avro::writer::async_writer::AvroObjectWriter;
/// # use object_store::memory::InMemory;
/// # use object_store::path::Path;
/// # use object_store::{ObjectStore, ObjectStoreExt};
/// # use std::io::Cursor;
/// # use std::sync::Arc;
/// #
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let store = Arc::new(InMemory::new());
///
/// let col = Arc::new(Int64Array::from_iter_values([1, 2, 3])) as ArrayRef;
/// let batch = RecordBatch::try_from_iter([("col", col)]).unwrap();
///
/// let object_store_writer = AvroObjectWriter::new(store.clone(), Path::from("out.avro"));
/// let mut writer =
///     AsyncAvroWriter::new(object_store_writer, batch.schema().as_ref().clone()).await.unwrap();
/// writer.write(&batch).await.unwrap();
/// writer.finish().await.unwrap();
///
/// let bytes = store.get(&Path::from("out.avro")).await.unwrap().bytes().await.unwrap();
/// let mut reader = ReaderBuilder::new().build(Cursor::new(bytes)).unwrap();
/// let read = reader.next().unwrap().unwrap();
/// assert_eq!(read.num_rows(), 3);
/// # }
/// ```
#[derive(Debug)]
pub struct AvroObjectWriter {
    w: BufWriter,
}

impl AvroObjectWriter {
    /// Create a new [`AvroObjectWriter`] that writes to the specified path in the given store.
    ///
    /// To configure the writer behavior (buffer size, multipart upload concurrency,
    /// content type, ...), build an [`object_store::buffered::BufWriter`] explicitly
    /// and use [`Self::from_buf_writer`].
    pub fn new(store: Arc<dyn ObjectStore>, path: Path) -> Self {
        Self::from_buf_writer(BufWriter::new(store, path))
    }

    /// Construct a new [`AvroObjectWriter`] from an existing [`BufWriter`].
    pub fn from_buf_writer(w: BufWriter) -> Self {
        Self { w }
    }

    /// Consume the writer and return the underlying [`BufWriter`].
    pub fn into_inner(self) -> BufWriter {
        self.w
    }
}

impl From<BufWriter> for AvroObjectWriter {
    fn from(w: BufWriter) -> Self {
        Self::from_buf_writer(w)
    }
}

impl AsyncFileWriter for AvroObjectWriter {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), ArrowError>> {
        Box::pin(async move {
            self.w.put(bs).await.map_err(|e| {
                ArrowError::ExternalError(Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            })
        })
    }

    fn complete(&mut self) -> BoxFuture<'_, Result<(), ArrowError>> {
        Box::pin(async move {
            self.w.shutdown().await.map_err(|e| {
                ArrowError::IoError(format!("Error finishing object store upload: {e}"), e)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::ReaderBuilder;
    use crate::writer::format::AvroOcfFormat;
    use crate::writer::{AsyncAvroWriter, WriterBuilder};
    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;
    use std::io::Cursor;

    #[tokio::test(flavor = "current_thread")]
    async fn roundtrip_via_object_store() -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = Path::from("roundtrip.avro");

        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef],
        )?;

        let sink = AvroObjectWriter::new(Arc::clone(&store), path.clone());
        let mut writer = AsyncAvroWriter::new(sink, schema).await?;
        writer.write(&batch).await?;
        writer.finish().await?;

        let bytes = store.get(&path).await?.bytes().await?;
        let mut reader = ReaderBuilder::new().build(Cursor::new(bytes))?;
        let out = reader.next().unwrap()?;
        assert_eq!(out, batch);
        Ok(())
    }

    /// Multiple batches accumulated into one OCF block must still round-trip
    /// when the sink is an object store (validates that block accumulation
    /// works correctly with the staged-multipart-upload code path).
    #[tokio::test(flavor = "current_thread")]
    async fn roundtrip_object_store_with_block_accumulation()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = Path::from("multi-batch.avro");

        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batches: Vec<RecordBatch> = (0..16)
            .map(|i| {
                RecordBatch::try_new(
                    Arc::new(schema.clone()),
                    vec![Arc::new(Int64Array::from(vec![i, i + 1, i + 2])) as ArrayRef],
                )
                .unwrap()
            })
            .collect();

        let sink = AvroObjectWriter::new(Arc::clone(&store), path.clone());
        let mut writer = WriterBuilder::new(schema)
            .build_async::<_, AvroOcfFormat>(sink)
            .await?;
        for b in &batches {
            writer.write(b).await?;
        }
        writer.finish().await?;

        let bytes = store.get(&path).await?.bytes().await?;
        let reader = ReaderBuilder::new().build(Cursor::new(bytes))?;
        let total: usize = reader
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(total, 16 * 3);
        Ok(())
    }
}
