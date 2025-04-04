"""
Simple wrapper script to run the Streamlit application
"""
import os
import sys
import subprocess

def main():
    """Run the Streamlit application"""
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Make sure the current directory is in the path
    sys.path.append(current_dir)
    
    # Set the working directory to the current directory
    os.chdir(current_dir)
    
    print("Starting SEC Filing Q&A Application...")
    print(f"Working directory: {current_dir}")
    
    # Run streamlit from the command line
    streamlit_path = "streamlit"
    main_script = os.path.join(current_dir, "main.py")
    
    try:
        subprocess.run([streamlit_path, "run", main_script], check=True)
    except Exception as e:
        print(f"Error running Streamlit application: {e}")
        print("If you're having trouble, try running 'streamlit run main.py' directly from the project directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
