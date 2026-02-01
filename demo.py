from __future__ import annotations

import numpy as np
from object_db import ObjectRecord, SceneRecord, ObjectVectorDB
from people_db import PersonRecord, PeopleVectorDB


def random_vector(dim: int) -> list[float]:
    # generate random normalized vector
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v.tolist()


def main() -> None:
    print("=" * 60)
    print("Vector Database Demo - Objects, Scenes & People")
    print("=" * 60)
    print()

    object_db = ObjectVectorDB(persist_directory="chroma_db")
    people_db = PeopleVectorDB(persist_directory="chroma_people_db")
    print("Object/Scene DB initialized")
    print("People DB initialized\n")

    print("=" * 60)
    print("PART 1: OBJECT STORAGE & QUERY")
    print("=" * 60)
    print()

    print("1. Adding objects to database")
    scene1 = SceneRecord(
        scene_id="kitchen_01",
        scene_xyz=(2.0, 2.5, 0.0),
        scene_image_ref="images/kitchen_01.png",
        scene_embedding=random_vector(256),
    )
    object_db.upsert_scene(scene1)
    print(f"  Created scene: {scene1.scene_id}\n")
    
    obj1 = ObjectRecord(
        object_id="mug_01",
        object_xyz=(1.0, 2.0, 0.0),
        object_image_ref="images/mug_01.png",
        object_embedding=random_vector(128),
        location_embedding=random_vector(64),
        scene_id="kitchen_01",
    )
    obj2 = ObjectRecord(
        object_id="bottle_01",
        object_xyz=(3.5, -1.2, 0.0),
        object_image_ref="images/bottle_01.png",
        object_embedding=random_vector(128),
        location_embedding=random_vector(64),
        scene_id="kitchen_01",
    )
    obj3 = ObjectRecord(
        object_id="cup_01",
        object_xyz=(0.5, 1.8, 0.0),
        object_image_ref="images/cup_01.png",
        object_embedding=random_vector(128),
        location_embedding=random_vector(64),
        scene_id="kitchen_01",
    )

    object_db.upsert(obj1)
    object_db.upsert(obj2)
    object_db.upsert(obj3)
    print(f"  Added objects: {obj1.object_id}, {obj2.object_id}, {obj3.object_id}\n")

    print("2. Query: Find objects similar to a query image")
    query_img_emb = random_vector(128)
    image_hits = object_db.query_by_image_embedding(query_img_emb, n_results=3)
    print(f"  Found {len(image_hits)} similar objects:")
    for i, hit in enumerate(image_hits, 1):
        print(f"    {i}. {hit['object_id']} (distance: {hit['distance']:.4f})")
        print(f"       Location: {hit['object_xyz']}")
        print(f"       Image: {hit['object_image_ref']}")
    print()

    print("3. Query: Find objects in similar locations")
    query_loc_emb = random_vector(64)
    location_hits = object_db.query_by_location_embedding(query_loc_emb, n_results=3)
    print(f"  Found {len(location_hits)} objects in similar locations:")
    for i, hit in enumerate(location_hits, 1):
        print(f"    {i}. {hit['object_id']} (distance: {hit['distance']:.4f})")
        print(f"       Location: {hit['object_xyz']}")
    print()

    print("4. Retrieve object by ID")
    retrieved = object_db.get_object("mug_01")
    if retrieved:
        print(f"  Found: {retrieved['object_id']}")
        print(f"    Location: {retrieved['object_xyz']}")
        print(f"    Image: {retrieved['object_image_ref']}")
        if retrieved.get('scene_id'):
            print(f"    Scene: {retrieved['scene_id']}")
    print()
    
    print("4b. Query: Get all objects in a scene")
    objects_in_scene = object_db.get_objects_by_scene("kitchen_01")
    print(f"  Found {len(objects_in_scene)} objects in scene 'kitchen_01':")
    for i, obj in enumerate(objects_in_scene, 1):
        print(f"    {i}. {obj['object_id']} at {obj['object_xyz']}")
    print()

    print("=" * 60)
    print("PART 2: SCENE STORAGE & QUERY")
    print("=" * 60)
    print()

    print("5. Adding scenes to database")
    scene1 = SceneRecord(
        scene_id="kitchen_01",
        scene_xyz=(2.5, 3.1, 0.0),
        scene_image_ref="images/kitchen_01.png",
        scene_embedding=random_vector(256),
    )
    scene2 = SceneRecord(
        scene_id="living_room_01",
        scene_xyz=(5.0, 2.0, 0.0),
        scene_image_ref="images/living_room_01.png",
        scene_embedding=random_vector(256),
    )
    scene3 = SceneRecord(
        scene_id="bedroom_01",
        scene_xyz=(7.2, 1.5, 0.0),
        scene_image_ref="images/bedroom_01.png",
        scene_embedding=random_vector(256),
    )

    object_db.upsert_scene(scene1)
    object_db.upsert_scene(scene2)
    object_db.upsert_scene(scene3)
    print(f"  Added scenes: {scene1.scene_id}, {scene2.scene_id}, {scene3.scene_id}\n")

    print("6. Query: Find scenes similar to a query scene")
    query_scene_emb = random_vector(256)
    scene_hits = object_db.query_by_scene_embedding(query_scene_emb, n_results=3)
    print(f"  Found {len(scene_hits)} similar scenes:")
    for i, hit in enumerate(scene_hits, 1):
        print(f"    {i}. {hit['scene_id']} (distance: {hit['distance']:.4f})")
        print(f"       SLAM Coordinates: {hit['scene_xyz']}")
        print(f"       Image: {hit['scene_image_ref']}")
    print()

    print("7. Query: Find scenes near SLAM coordinates")
    query_coords = (2.4, 3.0, 0.0)
    nearby_scenes = object_db.find_scenes_by_slam_coords(
        query_xyz=query_coords,
        radius=2.0,
        n_results=3
    )
    print(f"  Searching near {query_coords} (radius: 2.0m)")
    print(f"  Found {len(nearby_scenes)} nearby scenes:")
    for i, hit in enumerate(nearby_scenes, 1):
        print(f"    {i}. {hit['scene_id']} (distance: {hit['distance']:.4f}m)")
        print(f"       SLAM Coordinates: {hit['scene_xyz']}")
    print()

    print("8. Retrieve scene by ID")
    retrieved_scene = object_db.get_scene("kitchen_01")
    if retrieved_scene:
        print(f"  Found: {retrieved_scene['scene_id']}")
        print(f"    SLAM Coordinates: {retrieved_scene['scene_xyz']}")
        print(f"    Image: {retrieved_scene['scene_image_ref']}")
    print()

    print("=" * 60)
    print("PART 3: OBJECT & SCENE FINDING")
    print("=" * 60)
    print()

    print("9. Simulating Object&SceneFinding component:")
    print("   Finding objects by image similarity")
    obj_results = object_db.query_by_image_embedding(random_vector(128), n_results=2)
    print(f"   Found {len(obj_results)} objects")
    for result in obj_results:
        print(f"     {result['object_id']} at {result['object_xyz']}")

    print("\n   Finding scenes by scene similarity")
    scene_results = object_db.query_by_scene_embedding(random_vector(256), n_results=2)
    print(f"   Found {len(scene_results)} scenes")
    for result in scene_results:
        print(f"     {result['scene_id']} at {result['scene_xyz']}")
    print()

    print("=" * 60)
    print("PART 4: PEOPLE STORAGE & QUERY")
    print("=" * 60)
    print()

    print("10. Adding people to database")
    person1 = PersonRecord(
        person_id="person_01",
        people_xyz=(4.0, 5.0, 0.0),
        face_embedding=random_vector(512),
        pose_embedding=random_vector(256),
        timeframe="2024-01-01T10:30:00",
        chat_history_ref="chat/person_01.json",
    )
    person2 = PersonRecord(
        person_id="person_02",
        people_xyz=(6.0, 3.0, 0.0),
        face_embedding=random_vector(512),
        pose_embedding=random_vector(256),
        timeframe="2024-01-01T10:35:00",
    )
    person3 = PersonRecord(
        person_id="person_03",
        people_xyz=(8.0, 2.0, 0.0),
        face_embedding=random_vector(512),
        timeframe="2024-01-01T10:40:00",
    )

    people_db.upsert(person1)
    people_db.upsert(person2)
    people_db.upsert(person3)
    print(f"  Added people: {person1.person_id}, {person2.person_id}, {person3.person_id}\n")

    print("11. Query: Find people by face embedding")
    query_face_emb = random_vector(512)
    face_hits = people_db.query_by_face_embedding(query_face_emb, n_results=3)
    print(f"  Found {len(face_hits)} similar people by face:")
    for i, hit in enumerate(face_hits, 1):
        print(f"    {i}. {hit['person_id']} (distance: {hit['distance']:.4f})")
        print(f"       Location: {hit['people_xyz']}")
        if hit.get('timeframe'):
            print(f"       Timeframe: {hit['timeframe']}")
    print()

    print("12. Query: Find people by pose embedding")
    query_pose_emb = random_vector(256)
    pose_hits = people_db.query_by_pose_embedding(query_pose_emb, n_results=3)
    print(f"  Found {len(pose_hits)} people with similar poses:")
    for i, hit in enumerate(pose_hits, 1):
        print(f"    {i}. {hit['person_id']} (distance: {hit['distance']:.4f})")
        print(f"       Location: {hit['people_xyz']}")
    print()

    print("13. Retrieve person by FaceID")
    retrieved_person = people_db.get_person_by_face_id("person_01")
    if retrieved_person:
        print(f"  Found: {retrieved_person['person_id']}")
        print(f"    Location: {retrieved_person['people_xyz']}")
        if retrieved_person.get('chat_history_ref'):
            print(f"    Chat History: {retrieved_person['chat_history_ref']}")
    print()

    print("=" * 60)
    print("PART 5: PEOPLE FINDING")
    print("=" * 60)
    print()

    print("14. Simulating PeopleFinding component:")
    print("   Finding people by face")
    people_by_face = people_db.query_by_face_embedding(random_vector(512), n_results=2)
    print(f"   Found {len(people_by_face)} people")
    for result in people_by_face:
        print(f"     {result['person_id']} at {result['people_xyz']}")

    print("\n   Finding people by pose")
    people_by_pose = people_db.query_by_pose_embedding(random_vector(256), n_results=2)
    print(f"   Found {len(people_by_pose)} people")
    for result in people_by_pose:
        print(f"     {result['person_id']} at {result['people_xyz']}")
    print()

    print("=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

