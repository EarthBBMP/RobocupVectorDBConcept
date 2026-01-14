from __future__ import annotations

import numpy as np

from vector_db import ObjectRecord, ObjectVectorDB


def random_vector(dim: int) -> list[float]:
    # define random normalized vector
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v.tolist()


def main() -> None:
    db = ObjectVectorDB(persist_directory="chroma_db")

    # define example obj1
    obj1 = ObjectRecord(
        object_id="mug_01",
        object_xyz=(1.0, 2.0, 0.0),
        object_image_ref="images/mug_01.png",
        object_embedding=random_vector(128),
        location_embedding=random_vector(64),
    )

    # define example obj2
    obj2 = ObjectRecord(
        object_id="bottle_01",
        object_xyz=(3.5, -1.2, 0.0),
        object_image_ref="images/bottle_01.png",
        object_embedding=random_vector(128),
        location_embedding=random_vector(64),
    )

    # define insert objs
    db.upsert(obj1)
    db.upsert(obj2)

    # define query by image emb
    query_img_emb = random_vector(128)
    image_hits = db.query_by_image_embedding(query_img_emb, n_results=2)
    print("Nearest objects by IMAGE embedding:")
    for hit in image_hits:
        print(hit)

    # define query by location emb
    query_loc_emb = random_vector(64)
    location_hits = db.query_by_location_embedding(query_loc_emb, n_results=2)
    print("\nNearest objects by LOCATION embedding:")
    for hit in location_hits:
        print(hit)


if __name__ == "__main__":
    main()


