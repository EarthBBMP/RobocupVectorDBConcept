# Vector Database for Objects, Scenes & People

Vector databases for storing objects, scenes, and people. Objects and scenes are stored together, people are stored separately.

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

**Stores people with:**
- **Person ID** - unique identifier or FaceID
- **People Coordinates** - (x, y, z) position with timeframe
- **Face embedding** - for face recognition
- **Pose embedding** - for pose recognition (optional)
- **Chat history reference** - link to chat history

## Key Features

**Object Features:**
- Similarity search by image
- Similarity search by location
- Query objects by scene

**Scene Features:**
- Similarity search by scene embedding
- Spatial search by SLAM coordinates

**People Features:**
- Face recognition search
- Pose recognition search
- FaceID lookup

**General:**
- Persistent storage
- Simple API
- Separate databases for objects/scenes and people

## Data Schema (concept)

### Object Schema

| Field                | Type      | Description                              |
|----------------------|-----------|------------------------------------------|
| object_id            | string    | Unique object identifier                 |
| object_xyz           | float[3]  | (x, y, z) coords relative to map         |
| object_image_ref     | string    | Path/URI to object image                 |
| object_embedding     | vector    | Object image embedding                   |
| location_embedding   | vector    | Location / semantic embedding            |
| scene_id             | string    | Optional parent scene ID |

### Scene Schema

| Field                | Type      | Description                              |
|----------------------|-----------|------------------------------------------|
| scene_id             | string    | Unique scene identifier                  |
| scene_xyz            | float[3]  | (x, y, z) SLAM coordinates               |
| scene_image_ref      | string    | Path/URI to scene image                  |
| scene_embedding      | vector    | Scene embedding |

### Person Schema

| Field                | Type      | Description                              |
|----------------------|-----------|------------------------------------------|
| person_id            | string    | Unique person identifier (or FaceID)     |
| people_xyz           | float[3]  | (x, y, z) People Coordinate              |
| face_embedding       | vector    | Face embedding |
| pose_embedding       | vector    | Pose embedding |
| timeframe            | string    | When detected |
| chat_history_ref     | string    | Link to chat history |

## Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

## Usage Examples

### Object Storage & Query

```python
from object_db import ObjectRecord, ObjectVectorDB

db = ObjectVectorDB()

obj = ObjectRecord(
    object_id="mug_01",
    object_xyz=(1.0, 2.0, 0.0),
    object_image_ref="images/mug_01.png",
    object_embedding=[0.1, 0.2, ...],
    location_embedding=[0.3, 0.4, ...],
    scene_id="kitchen_01"
)

db.upsert(obj)

results = db.query_by_image_embedding(query_embedding, n_results=5)
results = db.query_by_location_embedding(query_embedding, n_results=5)
objects_in_scene = db.get_objects_by_scene("kitchen_01")
```

### Scene Storage & Query

```python
from object_db import SceneRecord, ObjectVectorDB

db = ObjectVectorDB()

scene = SceneRecord(
    scene_id="kitchen_01",
    scene_xyz=(2.5, 3.1, 0.0),
    scene_image_ref="images/kitchen_01.png",
    scene_embedding=[0.5, 0.6, ...]
)

db.upsert_scene(scene)

results = db.query_by_scene_embedding(query_scene_embedding, n_results=5)
results = db.find_scenes_by_slam_coords(
    query_xyz=(2.4, 3.0, 0.0),
    radius=1.0,
    n_results=5
)
```

### People Storage & Query

```python
from people_db import PersonRecord, PeopleVectorDB

people_db = PeopleVectorDB()

person = PersonRecord(
    person_id="person_01",
    people_xyz=(3.0, 4.0, 0.0),
    face_embedding=[0.7, 0.8, ...],
    pose_embedding=[0.9, 1.0, ...],
    timeframe="2024-01-01T10:30:00",
    chat_history_ref="chat/person_01.json"
)

people_db.upsert(person)

face_results = people_db.query_by_face_embedding(face_query_embedding, n_results=5)
pose_results = people_db.query_by_pose_embedding(pose_query_embedding, n_results=5)
person_info = people_db.get_person_by_face_id("face_123")
```

## Database Architecture

Two separate databases:

1. **ObjectVectorDB** (`chroma_db/`)
   - Stores objects and scenes together
   - Objects can reference their parent scene via `scene_id`

2. **PeopleVectorDB** (`chroma_people_db/`)
   - Separate database for people storage
   - Handles face recognition, pose recognition, and FaceID lookup

## Files

- `object_db.py` - Object and Scene database
- `people_db.py` - People database
- `vector_db.py` - Compatibility layer (re-exports from object_db and people_db)
- `demo.py` - Demo script
- `example_usage.py` - Basic usage example
- `agent_integration_example.py` - Agent integration examples

