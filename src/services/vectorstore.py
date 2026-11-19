"""Vector store operations using Zilliz Cloud (managed Milvus)."""
import os
import logging
import time
from typing import List, Dict, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

logger = logging.getLogger(__name__)

class ZillizVectorStore:
    """Zilliz Cloud vector store for learning content."""
    
    def __init__(self):
        """Initialize connection to Zilliz Cloud."""
        self.uri = os.getenv("ZILLIZ_CLOUD_URI")
        self.token = os.getenv("ZILLIZ_CLOUD_API_KEY")
        self.collection_name = os.getenv("ZILLIZ_COLLECTION_NAME", "learning_vec")
        self.dim = int(os.getenv("VECTOR_DIM", "1536"))
        self.collection = None
        
        self._connect()
        self._init_collection()
    
    def _connect(self):
        """Connect to Zilliz Cloud."""
        connections.connect(
            alias="default",
            uri=self.uri,
            token=self.token
        )
        print("✓ Connected to Zilliz Cloud")
    
    def _init_collection(self):
        """Initialize or load the collection with schema."""
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"✓ Loaded existing collection: {self.collection_name}")
        else:
            # Create new collection with schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
                FieldSchema(name="txt_emb", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="course_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="module_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="from_src", dtype=DataType.VARCHAR, max_length=512),
            ]
            
            schema = CollectionSchema(fields=fields, description="Learning content embeddings")
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create IVF_FLAT index for vector search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="txt_emb", index_params=index_params)
            self.collection.load()
            print(f"✓ Created new collection: {self.collection_name}")
    
    def insert(self, data: List[Dict]):
        """Insert data into the collection.
        
        Args:
            data: List of dictionaries containing:
                - id: unique identifier
                - txt_emb: vector embedding
                - text: original text content
                - course_id: course identifier (can be empty string)
                - module_id: module identifier
                - from_src: source filename
        """
        if not data:
            return
        
        logger.info(f"⏳ Preparing to insert {len(data)} vectors into Zilliz...")
        insert_start = time.time()
        
        # Debug: Log first item to verify field mapping
        if data:
            sample = data[0]
            logger.info(f"   🔍 Sample data - course_id: '{sample.get('course_id', 'N/A')}', module_id: '{sample.get('module_id', 'N/A')}', from_src: '{sample.get('from_src', 'N/A')}'")
            logger.info(f"   🔍 Sample text preview: '{sample.get('text', '')[:50]}...'")
        
        # IMPORTANT: Order must match the schema field order in Zilliz!
        # Schema order: id, txt_emb, course_id, module_id, text, from_src
        entities = [
            [item["id"] for item in data],
            [item["txt_emb"] for item in data],
            [item["text"] for item in data],
            [item.get("course_id", "").strip() for item in data],
            [item["module_id"].strip() for item in data],
            [item.get("from_src", "").strip() for item in data],
        ]
        
        logger.info(f"   ⏳ Inserting entities into collection '{self.collection_name}'...")
        logger.info(f"   📋 Field order: id, txt_emb, course_id, module_id, text, from_src")
        self.collection.insert(entities)
        
        logger.info("   ⏳ Flushing data to ensure persistence...")
        self.collection.flush()
        
        insert_time = time.time() - insert_start
        logger.info(f"✓ Inserted {len(data)} vectors into {self.collection_name} in {insert_time:.2f}s")
        print(f"✓ Inserted {len(data)} vectors into {self.collection_name}")
    
    def search(
        self,
        query_embedding: List[float],
        course_id: Optional[str] = None,
        module_id: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """Search for similar vectors with optional filtering.
        
        Args:
            query_embedding: Query vector embedding.
            course_id: Filter by course ID (optional, can be None).
            module_id: Filter by module ID (optional).
            top_k: Number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        if top_k is None:
            top_k = int(os.getenv("TOP_K", "5"))
        
        # Strip whitespace from filter values to handle trailing spaces in stored data
        if course_id:
            course_id = course_id.strip()
        if module_id:
            module_id = module_id.strip()
        
        # Log the filter values for debugging
        logger.info(f"🔍 Searching with filters - course_id: '{course_id}', module_id: '{module_id}'")
        
        # Build filter expression
        # NOTE: Use 'like' operator to handle potential trailing/leading spaces in stored data
        filter_expr = []
        if course_id:
            # Try exact match first, but stored data might have trailing spaces
            filter_expr.append(f'course_id like "{course_id}%"')
        if module_id:
            filter_expr.append(f'module_id like "{module_id}%"')
        
        expr = " && ".join(filter_expr) if filter_expr else None
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="txt_emb",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "course_id", "module_id", "from_src"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text"),
                    "course_id": hit.entity.get("course_id"),
                    "module_id": hit.entity.get("module_id"),
                    "from_src": hit.entity.get("from_src"),
                })
        
        return formatted_results
    
    def delete_by_filter(self, course_id: str = None, module_id: str = None):
        """Delete vectors by filter.
        
        Args:
            course_id: Course ID to delete.
            module_id: Module ID to delete.
        """
        filter_expr = []
        if course_id:
            filter_expr.append(f'course_id == "{course_id}"')
        if module_id:
            filter_expr.append(f'module_id == "{module_id}"')
        
        if not filter_expr:
            raise ValueError("At least one filter must be provided")
        
        expr = " && ".join(filter_expr)
        self.collection.delete(expr)
        print(f"✓ Deleted vectors matching: {expr}")
    
    def close(self):
        """Close the connection."""
        connections.disconnect("default")
