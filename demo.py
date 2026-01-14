from __future__ import annotations

import numpy as np
from vector_db import ObjectRecord, ObjectVectorDB


def random_vector(dim: int) -> list[float]:
    # define random normalized vector
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v.tolist()


def main() -> None:
    print("Vector Database Demo - Object Storage & Similarity Search")
    print()

    # define init db
    db = ObjectVectorDB(persist_directory="chroma_db")
    print("Database initialized\n")

    # define example objects
    print("1. Adding objects to database")
    obj1 = ObjectRecord(
        object_id="mug_01",
        object_xyz=(1.0, 2.0, 0.0),
        object_image_ref="images/mug_01.png",
        object_embedding=random_vector(128),
        location_embedding=random_vector(64),
    )
    obj2 = ObjectRecord(
        object_id="bottle_01",
        object_xyz=(3.5, -1.2, 0.0),
        object_image_ref="images/bottle_01.png",
        object_embedding=random_vector(128),
        location_embedding=random_vector(64),
    )
    obj3 = ObjectRecord(
        object_id="cup_01",
        object_xyz=(0.5, 1.8, 0.0),
        object_image_ref="images/cup_01.png",
        object_embedding=random_vector(128),
        location_embedding=random_vector(64),
    )

    db.upsert(obj1)
    db.upsert(obj2)
    db.upsert(obj3)
    print(f"  Added: {obj1.object_id}, {obj2.object_id}, {obj3.object_id}\n")

    # define query by image similarity
    print("2. Query: Find objects similar to a query image")
    query_img_emb = random_vector(128)
    image_hits = db.query_by_image_embedding(query_img_emb, n_results=3)
    print(f"  Found {len(image_hits)} similar objects:")
    for i, hit in enumerate(image_hits, 1):
        print(f"    {i}. {hit['object_id']} (distance: {hit['distance']:.4f})")
        print(f"       Location: {hit['object_xyz']}")
        print(f"       Image: {hit['object_image_ref']}")
    print()

    # define query by location similarity
    print("3. Query: Find objects in similar locations")
    query_loc_emb = random_vector(64)
    location_hits = db.query_by_location_embedding(query_loc_emb, n_results=3)
    print(f"  Found {len(location_hits)} objects in similar locations:")
    for i, hit in enumerate(location_hits, 1):
        print(f"    {i}. {hit['object_id']} (distance: {hit['distance']:.4f})")
        print(f"       Location: {hit['object_xyz']}")
    print()

    # define retrieve by id
    print("4. Retrieve object by ID")
    retrieved = db.get_object("mug_01")
    if retrieved:
        print(f"  Found: {retrieved['object_id']}")
        print(f"    Location: {retrieved['object_xyz']}")
        print(f"    Image: {retrieved['object_image_ref']}")
    print()

    print("Demo Done")


if __name__ == "__main__":
    main()

