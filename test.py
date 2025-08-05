#!/usr/bin/env python3
"""
Test script using the FIXED add_single_document method
"""

import os
import sys
import logging
import tempfile

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_fixed_chunking():
    """Test the FIXED chunking method"""
    print("🔍 TESTING FIXED PDF CHUNKING METHOD")
    print("=" * 60)
    
    # Your PDF path
    pdf_path = r"C:\Users\Chutchanan.Ma\Desktop\Project\Chatbot\PIM Documents\PIM General.pdf"
    
    # Alternative paths to try
    alternative_paths = [
        "PIM Documents/PIM General.pdf",
        "./PIM Documents/PIM General.pdf", 
        "PIM Documents\\PIM General.pdf",
        ".\\PIM Documents\\PIM General.pdf"
    ]
    
    # Find the PDF
    found_path = None
    if os.path.exists(pdf_path):
        found_path = pdf_path
        print(f"✅ Found PDF at original path: {pdf_path}")
    else:
        print(f"❌ Original path not found: {pdf_path}")
        print("🔍 Trying alternative paths...")
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                found_path = alt_path
                print(f"✅ Found PDF at: {alt_path}")
                break
        
        if not found_path:
            print("❌ Could not find PDF file!")
            print("Please make sure the PDF exists and run this script from the Chatbot folder")
            return
    
    # Test the FIXED method
    print(f"\n🏭 TESTING FIXED add_single_document METHOD")
    try:
        from services.embedding_service import EmbeddingService
        
        # Create service with your admin settings
        service = EmbeddingService(
            faculty="test_fixed",
            custom_chunk_size=30000,
            custom_chunk_overlap=3000
        )
        
        print(f"   ✅ EmbeddingService created with admin settings:")
        print(f"      - chunk_size: {service.chunk_size}")
        print(f"      - chunk_overlap: {service.chunk_overlap}")
        
        # Clear any existing test documents
        try:
            service.clear_all_documents()
            print(f"   🧹 Cleared existing test documents")
        except:
            pass
        
        # Use the FIXED add_single_document method
        print(f"\n   🔄 Processing PDF with FIXED method...")
        success = service.add_single_document(found_path, "PIM_General_TEST.pdf")
        
        if success:
            print(f"   ✅ Successfully processed PDF with fixed method!")
            
            # Get the processed chunks
            chunks = service.get_document_chunks("PIM_General_TEST.pdf", limit=20)
            
            print(f"\n   📊 RESULTS FROM FIXED METHOD:")
            print(f"      - Total chunks created: {len(chunks)}")
            
            if chunks:
                chunk_lengths = [chunk['length'] for chunk in chunks]
                avg_len = sum(chunk_lengths) / len(chunk_lengths)
                min_len = min(chunk_lengths)
                max_len = max(chunk_lengths)
                total_content = sum(chunk_lengths)
                
                print(f"      - Average chunk size: {avg_len:,.0f} characters")
                print(f"      - Min/Max chunk size: {min_len:,}/{max_len:,} characters")
                print(f"      - Total chunk content: {total_content:,} characters")
                
                print(f"\n   📋 Individual chunk analysis:")
                for i, chunk in enumerate(chunks):
                    efficiency = (chunk['length'] / 30000) * 100
                    print(f"      - Chunk {i+1}: {chunk['length']:,} chars ({efficiency:.1f}% of target)")
                
                # Show sample content
                print(f"\n   📄 SAMPLE CHUNK CONTENT:")
                if chunks:
                    first_chunk = chunks[0]
                    print(f"      - First chunk length: {first_chunk['length']:,} characters")
                    print(f"      - Content preview: {first_chunk['content']}")
                
                # Diagnosis
                if avg_len > 25000:
                    print(f"\n   ✅ SUCCESS: Fixed method creates large chunks!")
                    print(f"      Average chunk size ({avg_len:,.0f}) is close to target (30,000)")
                elif avg_len > 10000:
                    print(f"\n   ⚠️  IMPROVED: Chunks are larger but not at target")
                    print(f"      Average chunk size: {avg_len:,.0f} (target: 30,000)")
                elif avg_len > 5000:
                    print(f"\n   📈 BETTER: Chunks are larger than before")
                    print(f"      Average chunk size: {avg_len:,.0f} (was ~1,540 before)")
                else:
                    print(f"\n   ❌ STILL SMALL: Chunks are still small")
                    print(f"      Average chunk size: {avg_len:,.0f} (target: 30,000)")
            
            # Get collection statistics
            try:
                stats = service.get_chunk_statistics()
                print(f"\n   📈 COLLECTION STATISTICS:")
                print(f"      - Total chunks in collection: {stats.get('total_chunks', 'N/A')}")
                print(f"      - Configured chunk size: {stats.get('configured_chunk_size', 'N/A')}")
                print(f"      - Average actual length: {stats.get('average_actual_length', 'N/A')}")
            except Exception as e:
                print(f"      - Could not get statistics: {e}")
            
        else:
            print(f"   ❌ Failed to process PDF with fixed method")
        
    except Exception as e:
        print(f"   ❌ Error testing fixed method: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare with old method
    print(f"\n🔍 COMPARISON TEST: Testing old vs new method")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from services.embedding_service import PureSizeTextSplitter
        
        # Load PDF pages
        loader = PyPDFLoader(found_path)
        documents = loader.load()
        
        print(f"   📄 Loaded {len(documents)} pages")
        
        # Test OLD method (split each page separately)
        splitter = PureSizeTextSplitter(chunk_size=30000, chunk_overlap=3000)
        old_chunks = splitter.split_documents(documents)  # This splits each page separately
        
        # Test NEW method (combine pages first)
        combined_text = "\n\n".join(doc.page_content.strip() for doc in documents if doc.page_content.strip())
        new_chunks = splitter.split_text(combined_text)  # This splits combined text
        
        print(f"\n   📊 COMPARISON RESULTS:")
        print(f"      OLD METHOD (page-by-page):")
        print(f"        - Chunks: {len(old_chunks)}")
        print(f"        - Avg size: {sum(len(c.page_content) for c in old_chunks) / len(old_chunks):.0f} chars")
        
        print(f"      NEW METHOD (combined text):")
        print(f"        - Chunks: {len(new_chunks)}")
        print(f"        - Avg size: {sum(len(c) for c in new_chunks) / len(new_chunks):.0f} chars")
        print(f"        - Combined text length: {len(combined_text):,} chars")
        
        if len(new_chunks) < len(old_chunks):
            print(f"   ✅ NEW METHOD: Creates fewer, larger chunks!")
        else:
            print(f"   ⚠️  Something might still be wrong...")
        
    except Exception as e:
        print(f"   ❌ Error in comparison test: {e}")
    
    print(f"\n💡 FINAL DIAGNOSIS")
    print("=" * 60)
    
    if 'chunks' in locals() and chunks:
        avg_size = sum(chunk['length'] for chunk in chunks) / len(chunks)
        
        if avg_size > 15000:
            print("✅ THE FIX WORKED!")
            print(f"   - Your PDF now creates {len(chunks)} chunk(s)")
            print(f"   - Average size: {avg_size:,.0f} characters")
            print("   - This is much better than the previous 10 small chunks")
            print("\n🎯 Next steps:")
            print("   1. Use this fixed version in your admin console")
            print("   2. The chunks should now be properly sized for your use case")
        else:
            print("❌ The fix didn't work as expected")
            print(f"   - Still getting small chunks: {avg_size:.0f} chars")
            print("   - Need to investigate further")

if __name__ == "__main__":
    test_fixed_chunking()