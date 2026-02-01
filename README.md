# Vector Database for Objects, Scenes & People

Concept implementation of vector databases for storing:
- **Objects & Scenes** (together - scenes contain objects)
- **People** (separate - specialized storage)

Supports the agent flow with Object & Scene Finding and PeopleFinding capabilities.

## What It Does

**Stores objects with:**
- **Object ID** - unique identifier
- **Coordinates** - (x, y, z) position relative to map
- **Image reference** - path to object image
- **Image embedding** - vector representation of object appearance
- **Location embedding** - vector representation of location/semantic context
- **Scene ID** (optional) - links object to parent scene (scenes contain objects)

**Stores scenes with:**
- **Scene ID** - unique identifier
- **SLAM Coordinates** - (x, y, z) position from SLAM system
- **Scene image reference** - path to scene image
- **Scene embedding** - vector representation from Scene Detection

**Stores people with (separate database):**
- **Person ID** - unique identifier (or FaceID)
- **People Coordinates** - (x, y, z) position with timeframe
- **Face embedding** - from FaceRecognition (for FaceID lookup)
- **Pose embedding** - from PoseRecognition (optional)
- **Chat history reference** - link to ChatHistory

## Key Features

**Object Features:**
- **Similarity search by image** - find objects that look similar
- **Similarity search by location** - find objects in similar places
- **Query objects by scene** - get all objects in a scene (scenes contain objects)

**Scene Features:**
- **Similarity search by scene embedding** - find similar scenes (from Scene Detection)
- **Spatial search by SLAM coordinates** - find scenes near specific locations

**People Features (separate database):**
- **Face recognition search** - find people by face embedding (FaceRecognition → FaceID)
- **Pose recognition search** - find people by pose embedding (PoseRecognition)
- **FaceID lookup** - get person by FaceID

**General:**
- **Persistent storage** - data saved to disk automatically
- **Simple API** - easy to integrate with LLM calls and agent flow
- **Separate databases** - Objects/Scenes together, People separate (as per design)

## Data Schema (concept)

### Object Schema

| Field                | Type      | Description                              |
|----------------------|-----------|------------------------------------------|
| object_id            | string    | Unique object identifier                 |
| object_xyz           | float[3]  | (x, y, z) coords relative to map         |
| object_image_ref     | string    | Path/URI to object image                 |
| object_embedding     | vector    | Object image embedding                   |
| location_embedding   | vector    | Location / semantic embedding            |
| scene_id             | string?   | (Optional) Parent scene ID (scenes contain objects) |

### Scene Schema

| Field                | Type      | Description                              |
|----------------------|-----------|------------------------------------------|
| scene_id             | string    | Unique scene identifier                  |
| scene_xyz            | float[3]  | (x, y, z) SLAM coordinates               |
| scene_image_ref      | string    | Path/URI to scene image                  |
| scene_embedding      | vector    | Scene embedding from Scene Detection    |

### Person Schema (People Database)

| Field                | Type      | Description                              |
|----------------------|-----------|------------------------------------------|
| person_id            | string    | Unique person identifier (or FaceID)     |
| people_xyz           | float[3]  | (x, y, z) People Coordinate              |
| face_embedding       | vector?   | Face embedding from FaceRecognition     |
| pose_embedding       | vector?   | Pose embedding from PoseRecognition     |
| timeframe            | string?   | When detected (People Coordinate / Timeframe) |
| chat_history_ref     | string?   | Link to ChatHistory                      |

## Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

## Usage Examples

### Object Storage & Query

```python
from vector_db import ObjectRecord, ObjectVectorDB
import numpy as np

# initialize DB
db = ObjectVectorDB()

# create object (optionally link to a scene)
obj = ObjectRecord(
    object_id="mug_01",
    object_xyz=(1.0, 2.0, 0.0),
    object_image_ref="images/mug_01.png",
    object_embedding=[0.1, 0.2, ...],  # your image embedding vector
    location_embedding=[0.3, 0.4, ...],  # your location embedding vector
    scene_id="kitchen_01"  # optional: link to parent scene
)

# store object
db.upsert(obj)

# query similar objects by image
results = db.query_by_image_embedding(query_embedding, n_results=5)

# query similar objects by location
results = db.query_by_location_embedding(query_embedding, n_results=5)

# get all objects in a scene (scenes contain objects)
objects_in_scene = db.get_objects_by_scene("kitchen_01")
```

### Scene Storage & Query

```python
from vector_db import SceneRecord, ObjectVectorDB
import numpy as np

# initialize DB (same instance handles both objects and scenes)
db = ObjectVectorDB()

# create scene (from Scene Detection → Scene Embedding → SLAM Coordinate)
scene = SceneRecord(
    scene_id="kitchen_01",
    scene_xyz=(2.5, 3.1, 0.0),  # SLAM coordinates
    scene_image_ref="images/kitchen_01.png",
    scene_embedding=[0.5, 0.6, ...]  # scene embedding from Scene Detection
)

# store scene
db.upsert_scene(scene)

# query similar scenes by scene embedding
results = db.query_by_scene_embedding(query_scene_embedding, n_results=5)

# find scenes near SLAM coordinates
results = db.find_scenes_by_slam_coords(
    query_xyz=(2.4, 3.0, 0.0),
    radius=1.0,  # search radius in meters
    n_results=5
)
```

### Object & Scene Finding (Agent Flow Integration)

```python
# From Object&SceneFinding component in agent flow:

# Find objects
object_results = db.query_by_image_embedding(object_query_embedding, n_results=5)

# Find scenes
scene_results = db.query_by_scene_embedding(scene_query_embedding, n_results=5)

# Get objects in a scene (scenes contain objects)
objects_in_scene = db.get_objects_by_scene("kitchen_01")
```

### People Storage & Query (Separate Database)

```python
from vector_db import PersonRecord, PeopleVectorDB
import numpy as np

# initialize People DB (separate from objects/scenes)
people_db = PeopleVectorDB()

# create person (from People Detection → FaceRecognition/PoseRecognition)
person = PersonRecord(
    person_id="person_01",  # or face_id from FaceRecognition
    people_xyz=(3.0, 4.0, 0.0),  # People Coordinate
    face_embedding=[0.7, 0.8, ...],  # from FaceRecognition
    pose_embedding=[0.9, 1.0, ...],  # from PoseRecognition (optional)
    timeframe="2024-01-01T10:30:00",
    chat_history_ref="chat/person_01.json"  # link to ChatHistory
)

# store person
people_db.upsert(person)

# query by face embedding (FaceRecognition → FaceID)
face_results = people_db.query_by_face_embedding(face_query_embedding, n_results=5)

# query by pose embedding (PoseRecognition)
pose_results = people_db.query_by_pose_embedding(pose_query_embedding, n_results=5)

# get person by FaceID
person_info = people_db.get_person_by_face_id("face_123")
```

### PeopleFinding (Agent Flow Integration)

```python
# From PeopleFinding component in agent flow:

# Find people by face (FaceRecognition → FaceID)
people = people_db.query_by_face_embedding(face_embedding, n_results=5)

# Find people by pose
people_by_pose = people_db.query_by_pose_embedding(pose_embedding, n_results=5)

# Get person coordinates for robot navigation
person = people_db.get_person("person_01")
if person:
    coords = person['people_xyz']
    # Use for robot navigation
```

## Database Architecture

The system uses **two separate databases** as per design:

1. **ObjectVectorDB** (`chroma_db/`)
   - Stores objects and scenes together
   - Scenes act as collections containing objects
   - Objects can reference their parent scene via `scene_id`

2. **PeopleVectorDB** (`chroma_people_db/`)
   - Separate database for people storage
   - Specialized storage with minimal cross-referencing
   - Handles FaceRecognition, PoseRecognition, and FaceID lookup

## Files

- `vector_db.py` - main database implementation (both ObjectVectorDB and PeopleVectorDB)
- `demo.py` - presentation demo script
- `example_usage.py` - basic usage example
- `agent_integration_example.py` - integration examples for agent flow

