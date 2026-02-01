# Step-by-Step Run Guide

This guide shows you how to run the database system step by step.

## 📋 Prerequisites

Make sure you have Python 3.8+ installed.

## 🚀 Step 1: Install Dependencies

Open your terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

This will install:
- `chromadb` - Vector database
- `numpy` - For vector operations

**Expected output:**
```
Successfully installed chromadb-0.4.24 numpy-1.26.0
```

---

## 🎯 Step 2: Run Basic Example (Quick Start)

This is the simplest example to test if everything works:

```bash
python example_usage.py
```

**What it does:**
- Creates 2 objects
- Stores them in the database
- Queries by image embedding
- Queries by location embedding

**Expected output:**
```
Nearest objects by IMAGE embedding:
{'object_id': 'mug_01', 'object_xyz': [1.0, 2.0, 0.0], ...}
{'object_id': 'bottle_01', 'object_xyz': [3.5, -1.2, 0.0], ...}

Nearest objects by LOCATION embedding:
...
```

---

## 🎬 Step 3: Run Full Demo (Recommended)

This shows all features including objects, scenes, and people:

```bash
python demo.py
```

**What it does:**
- **Part 1:** Object storage & queries
- **Part 2:** Scene storage & queries  
- **Part 3:** Object & Scene Finding integration
- **Part 4:** People storage & queries (separate DB)
- **Part 5:** People Finding integration

**Expected output:**
```
============================================================
Vector Database Demo - Objects, Scenes & People
============================================================

✓ Object/Scene DB initialized
✓ People DB initialized (separate database)

============================================================
PART 1: OBJECT STORAGE & QUERY
============================================================

1. Adding objects to database
  ✓ Created scene: kitchen_01
  ✓ Added objects: mug_01, bottle_01, cup_01

2. Query: Find objects similar to a query image
  Found 3 similar objects:
    1. mug_01 (distance: 0.8234)
       Location: [1.0, 2.0, 0.0]
       ...

[... continues with all parts ...]

============================================================
✓ Demo Complete!
============================================================
```

**This is the best file to show your senior!** It demonstrates everything.

---

## 🤖 Step 4: Run Agent Integration Example

This shows how to integrate with your agent/robot system:

```bash
python agent_integration_example.py
```

**What it does:**
- Simulates Vision component detecting objects/scenes/people
- Simulates Object&SceneFinding component querying
- Simulates PeopleFinding component querying
- Simulates ControlRobot getting coordinates

**Expected output:**
```
============================================================
Agent & Robot Integration Example
============================================================

1. Vision → Object Detection → Storing objects
✓ Stored object: mug_01 at (1.0, 2.0, 0.0)

2. Vision → Scene Detection → Storing scenes
✓ Stored scene: kitchen_01 at SLAM (2.5, 3.1, 0.0)

3. Vision → People Detection → Storing people (separate DB)
✓ Stored person: person_01 at (4.0, 5.0, 0.0)

4. Object&SceneFinding → Querying ObjectVectorDB
   Found 3 objects
   Found 3 scenes

5. PeopleFinding → Querying PeopleVectorDB
   Found 3 people by face
   Found 3 people by pose
   Found person by FaceID: person_01

6. ControlRobot → Getting coordinates for navigation
   Object 'mug_01' is at: (1.0, 2.0, 0.0)
   Robot can navigate to: Go to object (1.0, 2.0)
   Scene 'kitchen_01' is at SLAM: (2.5, 3.1, 0.0)
   Person 'person_01' is at: (4.0, 5.0, 0.0)

============================================================
✓ Integration example complete!
============================================================
```

---

## 📁 Step 5: Check Database Files

After running, you'll see database folders created:

```
Robocup/
├── chroma_db/          # Object & Scene database
│   ├── chroma.sqlite3
│   └── ...
└── chroma_people_db/    # People database (separate)
    ├── chroma.sqlite3
    └── ...
```

These folders contain your persistent data. You can delete them to start fresh.

---

## 🔄 Step 6: Run Again (Data Persists)

Run the demo again:

```bash
python demo.py
```

**Notice:** The data from the previous run is still there! The databases persist to disk automatically.

To start fresh, delete the database folders:
```bash
# Windows PowerShell
Remove-Item -Recurse -Force chroma_db, chroma_people_db

# Windows CMD
rmdir /s /q chroma_db chroma_people_db

# Linux/Mac
rm -rf chroma_db chroma_people_db
```

---

## 🎓 Understanding the Output

### Object Queries
- **distance**: Lower = more similar (cosine similarity)
- **object_xyz**: 3D coordinates [x, y, z]
- **scene_id**: Which scene contains this object (if linked)

### Scene Queries
- **scene_xyz**: SLAM coordinates
- **distance**: Scene similarity score

### People Queries
- **people_xyz**: Person coordinates with timeframe
- **distance**: Face/pose similarity score
- **chat_history_ref**: Link to chat history (if available)

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'chromadb'"
**Solution:** Run `pip install -r requirements.txt` again

### Error: "ModuleNotFoundError: No module named 'object_db'"
**Solution:** Make sure you're in the project folder. Check that `object_db.py` exists.

### Database locked error
**Solution:** Close any other Python processes using the database, or delete the database folders to start fresh.

### Import errors
**Solution:** Make sure all files are in the same folder:
- `object_db.py`
- `people_db.py`
- `vector_db.py`
- `demo.py`
- `example_usage.py`
- `agent_integration_example.py`

---

## 📝 Quick Reference

| File | Purpose | When to Run |
|------|---------|-------------|
| `example_usage.py` | Basic test | First time setup |
| `demo.py` | Full demonstration | **Show to senior** |
| `agent_integration_example.py` | Integration example | Understand agent flow |

---

## ✅ Checklist

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Ran `example_usage.py` successfully
- [ ] Ran `demo.py` and saw all parts
- [ ] Ran `agent_integration_example.py`
- [ ] Understood the separate databases (objects/scenes vs people)
- [ ] Ready to show your senior! 🎉

---

## 🎯 For Your Senior

**Recommended order to show:**
1. **`demo.py`** - Shows complete functionality
2. **`agent_integration_example.py`** - Shows how it integrates with agent flow
3. **Code files** - `object_db.py` and `people_db.py` (separate as requested)

**Key points to highlight:**
- ✅ Two separate databases (objects/scenes together, people separate)
- ✅ Scenes contain objects (via `scene_id`)
- ✅ People database is specialized with minimal cross-referencing
- ✅ Ready for agent integration
