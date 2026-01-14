# Vector Database for Object Storage

Concept implementation of a vector database for storing objects with image and location embeddings.

## What It Does

Stores objects with:
- **Object ID** - unique identifier
- **Coordinates** - (x, y, z) position relative to map
- **Image reference** - path to object image
- **Image embedding** - vector representation of object appearance
- **Location embedding** - vector representation of location/semantic context

## Key Features

- **Similarity search by image** - find objects that look similar
- **Similarity search by location** - find objects in similar places
- **Persistent storage** - data saved to disk automatically
- **Simple API** - easy to integrate with LLM calls

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py
```

## Usage Example

```python
from vector_db import ObjectRecord, ObjectVectorDB
import numpy as np

# Initialize database
db = ObjectVectorDB()

# Create an object
obj = ObjectRecord(
    object_id="mug_01",
    object_xyz=(1.0, 2.0, 0.0),
    object_image_ref="images/mug_01.png",
    object_embedding=[0.1, 0.2, ...],  # your image embedding vector
    location_embedding=[0.3, 0.4, ...]  # your location embedding vector
)

# Store object
db.upsert(obj)

# Query similar objects by image
results = db.query_by_image_embedding(query_embedding, n_results=5)

# Query similar objects by location
results = db.query_by_location_embedding(query_embedding, n_results=5)
```

## Files

- `vector_db.py` - main database implementation
- `demo.py` - presentation demo script
- `example_usage.py` - basic usage example

