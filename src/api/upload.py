"""Upload API endpoints for document ingestion."""
import logging
import time
import tempfile
import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.models.schemas import UploadResponse, FileUploadResult, AzureUploadRequest
from src.services import EmbeddingService, ZillizVectorStore
from src.tools import TextExtractor, chunk_text, AzureBlobStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])


def process_file(
    file_path: str,
    filename: str,
    course_id: str,
    module_id: str,
    from_src: Optional[str],
    embedding_service: EmbeddingService,
    vector_store: ZillizVectorStore
) -> FileUploadResult:
    """Process a single file: extract, chunk, embed, and store.
    
    Args:
        file_path: Path to the file.
        filename: Original filename.
        course_id: Course identifier.
        module_id: Module identifier.
        from_src: Source name (uses filename if None).
        embedding_service: Embedding service instance.
        vector_store: Vector store instance.
        
    Returns:
        FileUploadResult with processing details.
    """
    try:
        source_name = from_src if from_src else filename
        logger.info(f"\n{'='*80}")
        logger.info(f"📄 Processing file: {filename}")
        logger.info(f"{'='*80}")
        
        # Step 1: Extract text
        logger.info("⏳ [STEP 1/4] Extracting text...")
        extract_start = time.time()
        
        text_content = TextExtractor.extract_text(file_path)
        
        if not text_content.strip():
            return FileUploadResult(
                filename=filename,
                success=False,
                error="Could not extract text from file. File might be empty or corrupted."
            )
        
        extract_time = time.time() - extract_start
        text_length = len(text_content)
        logger.info(f"✓ [STEP 1/4] Text extraction completed in {extract_time:.2f}s ({text_length:,} characters)")
        
        # Step 2: Chunk text
        logger.info("⏳ [STEP 2/4] Chunking text (500 chars, 50 overlap)...")
        chunk_start = time.time()
        
        chunks = chunk_text(text_content, chunk_size=500, overlap=50)
        
        if not chunks:
            return FileUploadResult(
                filename=filename,
                success=False,
                error="No valid text chunks created from file"
            )
        
        chunk_time = time.time() - chunk_start
        logger.info(f"✓ [STEP 2/4] Created {len(chunks)} chunks in {chunk_time:.2f}s")
        
        # Step 3: Generate embeddings and store
        logger.info("⏳ [STEP 3/4] Generating embeddings...")
        batch_size = 20
        total_inserted = 0
        embed_start = time.time()
        
        for i in range(0, len(chunks), batch_size):
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            batch_chunks = chunks[i:i+batch_size]
            
            logger.info(f"   ⏳ Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
            
            # Generate embeddings
            embeddings = embedding_service.embed_texts(batch_chunks)
            
            # Prepare data for insertion
            data = []
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                chunk_id = f"{course_id}::{module_id}::{source_name}::chunk_{i+j}"
                data.append({
                    "id": chunk_id,
                    "txt_emb": embedding,
                    "course_id": course_id,
                    "module_id": module_id,
                    "text": chunk,
                    "from_src": source_name
                })
                
            # Insert into vector store
            vector_store.insert(data)
            total_inserted += len(data)
        
        embed_time = time.time() - embed_start
        logger.info(f"✓ [STEP 3/4] Embeddings completed in {embed_time:.2f}s")
        logger.info(f"✓ [STEP 4/4] Stored {total_inserted} chunks")
        logger.info(f"🎉 File processed successfully: {filename}")
        
        return FileUploadResult(
            filename=filename,
            success=True,
            chunks_count=total_inserted
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing {filename}: {str(e)}")
        return FileUploadResult(
            filename=filename,
            success=False,
            error=str(e)
        )


@router.post("/files", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(..., description="Multiple files to upload (PDF, DOCX, TXT)"),
    course_id: str = Form(..., description="Course ID"),
    module_id: str = Form(..., description="Module ID"),
    from_src: Optional[str] = Form(None, description="Source name (optional, defaults to filename)")
):
    """
    Upload and embed multiple files into the vector database.
    
    - **files**: Multiple files (PDF, DOCX, TXT) - required
    - **course_id**: Course identifier - required
    - **module_id**: Module identifier - required
    - **from_src**: Source name (optional, defaults to filename)
    
    Supported file types: PDF, DOCX, TXT
    
    Returns:
    - **total_files**: Number of files processed
    - **successful**: Number of successful uploads
    - **failed**: Number of failed uploads
    - **total_chunks**: Total chunks created across all files
    - **results**: Individual result for each file
    """
    start_time = time.time()
    logger.info(f"\n{'='*80}")
    logger.info(f"📥 [UPLOAD START] Received {len(files)} files")
    logger.info(f"   Course ID: {course_id}")
    logger.info(f"   Module ID: {module_id}")
    logger.info(f"{'='*80}")
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = ZillizVectorStore()
    
    results = []
    temp_files = []
    
    try:
        # Process each file
        for file in files:
            logger.info(f"\n📄 File: {file.filename}")
            
            # Validate file type
            file_ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
            if file_ext not in ['pdf', 'docx', 'doc', 'txt']:
                results.append(FileUploadResult(
                    filename=file.filename,
                    success=False,
                    error=f"Unsupported file type: {file_ext}. Supported: pdf, docx, txt"
                ))
                continue
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
                temp_files.append(tmp_file_path)
            
            # Process file
            result = process_file(
                file_path=tmp_file_path,
                filename=file.filename,
                course_id=course_id,
                module_id=module_id,
                from_src=from_src,
                embedding_service=embedding_service,
                vector_store=vector_store
            )
            results.append(result)
        
        # Calculate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_chunks = sum(r.chunks_count for r in results if r.success)
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 [UPLOAD COMPLETE]")
        logger.info(f"   Total files: {len(results)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"{'='*80}")
        
        return UploadResponse(
            total_files=len(results),
            successful=successful,
            failed=failed,
            total_chunks=total_chunks,
            course_id=course_id,
            module_id=module_id,
            results=results
        )
        
    finally:
        # Clean up temp files
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {tmp_path}: {e}")


@router.post("/from-azure", response_model=UploadResponse)
async def upload_from_azure(request: AzureUploadRequest):
    """
    Upload and embed files from Azure Blob Storage URLs.
    
    - **urls**: List of Azure blob storage URLs - required
    - **course_id**: Course identifier - required
    - **module_id**: Module identifier - required
    - **from_src**: Source name (optional, defaults to filename from URL)
    
    Supported file types: PDF, DOCX, TXT
    URLs must be publicly accessible or include SAS token.
    
    Returns:
    - **total_files**: Number of files processed
    - **successful**: Number of successful uploads
    - **failed**: Number of failed uploads
    - **total_chunks**: Total chunks created across all files
    - **results**: Individual result for each file
    """
    start_time = time.time()
    logger.info(f"\n{'='*80}")
    logger.info(f"📥 [AZURE UPLOAD START] Received {len(request.urls)} URLs")
    logger.info(f"   Course ID: {request.course_id}")
    logger.info(f"   Module ID: {request.module_id}")
    logger.info(f"{'='*80}")
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = ZillizVectorStore()
    azure_storage = AzureBlobStorage()
    
    results = []
    temp_files = []
    
    try:
        # Process each URL
        for url in request.urls:
            logger.info(f"\n🔗 URL: {url[:80]}...")
            
            try:
                # Download from Azure
                file_path, filename = azure_storage.download_from_url(url)
                temp_files.append(file_path)
                
                # Validate file type
                file_type = azure_storage.detect_file_type(url, file_path)
                if file_type not in ['pdf', 'docx', 'txt']:
                    results.append(FileUploadResult(
                        filename=filename,
                        success=False,
                        error=f"Unsupported file type: {file_type}. Supported: pdf, docx, txt"
                    ))
                    continue
                
                # Process file
                result = process_file(
                    file_path=file_path,
                    filename=filename,
                    course_id=request.course_id,
                    module_id=request.module_id,
                    from_src=request.from_src,
                    embedding_service=embedding_service,
                    vector_store=vector_store
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Failed to process URL: {str(e)}")
                results.append(FileUploadResult(
                    filename=url.split('/')[-1] if '/' in url else url,
                    success=False,
                    error=f"Download or processing failed: {str(e)}"
                ))
        
        # Calculate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_chunks = sum(r.chunks_count for r in results if r.success)
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 [AZURE UPLOAD COMPLETE]")
        logger.info(f"   Total files: {len(results)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"{'='*80}")
        
        return UploadResponse(
            total_files=len(results),
            successful=successful,
            failed=failed,
            total_chunks=total_chunks,
            course_id=request.course_id,
            module_id=request.module_id,
            results=results
        )
        
    finally:
        # Clean up temp files
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {tmp_path}: {e}")


# Keep old /pdf endpoint for backward compatibility
@router.post("/pdf")
async def upload_pdf_single(
    file: UploadFile = File(..., description="PDF file to upload"),
    course_id: str = Form(..., description="Course ID"),
    module_id: str = Form(..., description="Module ID"),
    from_src: Optional[str] = Form(None, description="Source name (optional, defaults to filename)")
):
    """
    Upload a single PDF file (backward compatibility endpoint).
    
    For new implementations, use /upload/files for multiple file support.
    """
    # Redirect to the new multi-file endpoint
    result = await upload_files([file], course_id, module_id, from_src)
    
    # Return old format for backward compatibility
    if result.results:
        first_result = result.results[0]
        return {
            "success": first_result.success,
            "message": first_result.error if not first_result.success else f"PDF uploaded successfully. Created {first_result.chunks_count} chunks.",
            "chunks_count": first_result.chunks_count,
            "course_id": course_id,
            "module_id": module_id,
            "from_src": from_src if from_src else file.filename
        }
    else:
        raise HTTPException(status_code=500, detail="Processing failed")
