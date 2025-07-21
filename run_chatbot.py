#!/usr/bin/env python3
"""
Simple Streamlit chatbot launcher
"""
import subprocess
import sys
import os

def main():
    print("🚀 Launching Company Chatbot...")
    print("This will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Run streamlit with the fixed frontend
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/chatbot_test.py",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped")
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")
        print("\nTry running manually:")
        print("streamlit run frontend/chatbot_test.py")

if __name__ == "__main__":
    main()