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

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_avro::errors::AvroError;
use arrow_avro::reader::{AsyncAvroFileReader, AsyncFileReader, SpawnedReader};
use arrow_avro::writer::{AsyncAvroWriter, AsyncFileWriter};
use arrow_schema::ArrowError;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, TryStreamExt};
use object_store::buffered::BufWriter;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use std::error::Error;
use std::ops::Range;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// This example demonstrates reading and writing Avro files against object storage
/// via the [`object_store`] crate. `arrow-avro` does not ship a dedicated
/// `object_store` integration type; instead, implement [`AsyncFileReader`] /
/// [`AsyncFileWriter`] directly, as shown below.
///
/// # Example Overview
///
/// 1. Writes an Avro Object Container File to an [`ObjectStore`] with
///    [`ObjectStoreWriter`], a minimal [`AsyncFileWriter`] implementation on top
///    of an [`ObjectStore`]
///
/// 2. Reads it back with [`ObjectStoreReader`], a minimal [`AsyncFileReader`]
///    implementation on top of an [`ObjectStore`]
///
/// 3. Reads it again with the reader wrapped in a [`SpawnedReader`], which
///    performs all I/O on a separate tokio runtime so that the runtime
///    decoding Avro is not also driving the I/O
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = Path::from("example.avro");

    // 1. Write an Avro Object Container File directly to the store with an
    // `AsyncFileWriter` implemented on `ObjectStore`.
    let col = Arc::new(Int64Array::from_iter_values([1, 2, 3])) as ArrayRef;
    let batch = RecordBatch::try_from_iter([("col", col)])?;

    let sink = ObjectStoreWriter::new(Arc::clone(&store), path.clone());
    let mut writer = AsyncAvroWriter::new(sink, batch.schema().as_ref().clone()).await?;
    writer.write(&batch).await?;
    writer.finish().await?;

    // 2. Read it back with an `AsyncFileReader` implemented on `ObjectStore`.
    // The builder requires the file size, which can be obtained via `head`
    let file_size = store.head(&path).await?.size;

    let reader = ObjectStoreReader::new(Arc::clone(&store), path.clone());
    let stream = AsyncAvroFileReader::builder(reader, file_size, 1024)
        .try_build()
        .await?;
    let read: Vec<RecordBatch> = stream.try_collect().await?;
    assert_eq!(read, vec![batch.clone()]);
    println!("read {} rows", read[0].num_rows());

    // 3. Read again, performing the I/O on a dedicated runtime
    let io_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("failed to build I/O runtime");

    let reader = ObjectStoreReader::new(Arc::clone(&store), path);
    let reader = SpawnedReader::new(reader, io_runtime.handle().clone());
    let stream = AsyncAvroFileReader::builder(reader, file_size, 1024)
        .try_build()
        .await?;
    let read: Vec<RecordBatch> = stream.try_collect().await?;
    assert_eq!(read, vec![batch]);
    println!("read {} rows via dedicated I/O runtime", read[0].num_rows());

    io_runtime.shutdown_background();
    Ok(())
}

/// An [`AsyncFileWriter`] for a location in an [`ObjectStore`], writing via
/// multipart upload. This mirrors the example on the [`AsyncFileWriter`] trait
/// documentation.
#[derive(Debug)]
struct ObjectStoreWriter {
    w: BufWriter,
}

impl ObjectStoreWriter {
    fn new(store: Arc<dyn ObjectStore>, path: Path) -> Self {
        Self {
            w: BufWriter::new(store, path),
        }
    }
}

impl AsyncFileWriter for ObjectStoreWriter {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), ArrowError>> {
        async move {
            self.w.put(bs).await.map_err(|e| {
                ArrowError::ExternalError(Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            })
        }
        .boxed()
    }

    fn complete(&mut self) -> BoxFuture<'_, Result<(), ArrowError>> {
        async move {
            self.w.shutdown().await.map_err(|e| {
                ArrowError::IoError(format!("Error finishing object store upload: {e}"), e)
            })
        }
        .boxed()
    }

    fn abort(&mut self) -> BoxFuture<'_, Result<(), ArrowError>> {
        async move {
            self.w.abort().await.map_err(|e| {
                ArrowError::ExternalError(Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            })
        }
        .boxed()
    }
}

/// An [`AsyncFileReader`] for a location in an [`ObjectStore`]
///
/// This mirrors the example on the [`AsyncFileReader`] trait documentation.
#[derive(Clone, Debug)]
struct ObjectStoreReader {
    store: Arc<dyn ObjectStore>,
    path: Path,
}

impl ObjectStoreReader {
    fn new(store: Arc<dyn ObjectStore>, path: Path) -> Self {
        Self { store, path }
    }
}

impl AsyncFileReader for ObjectStoreReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes, AvroError>> {
        async move {
            self.store
                .get_range(&self.path, range)
                .await
                .map_err(|e| AvroError::General(e.to_string()))
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, Result<Vec<Bytes>, AvroError>> {
        async move {
            self.store
                .get_ranges(&self.path, &ranges)
                .await
                .map_err(|e| AvroError::General(e.to_string()))
        }
        .boxed()
    }
}
