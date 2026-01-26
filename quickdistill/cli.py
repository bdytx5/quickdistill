"""
CLI interface for QuickDistill
"""
import argparse
import webbrowser
import time
import os
from pathlib import Path
from quickdistill.server import app


def launch_command(args):
    """Launch the QuickDistill UI"""
    port = args.port
    data_dir = Path.home() / '.cache' / 'quickdistill'

    print(f"🚀 Launching QuickDistill UI...")
    print(f"💾 Data directory: {data_dir}")
    print(f"🌐 Server running on http://localhost:{port}")
    print(f"\n💡 Access the UI at:")
    print(f"   - Trace Viewer: http://localhost:{port}/")
    print(f"   - Judge Manager: http://localhost:{port}/judge")
    print(f"\n⏹  Press Ctrl+C to stop the server\n")

    # Open browser after a short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}/")

        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=args.debug)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="QuickDistill - Fast and easy AI model distillation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quickdistill launch              Launch the UI on default port 5001
  quickdistill launch --port 8080  Launch on custom port
  quickdistill launch --no-browser Launch without opening browser
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Launch command
    launch_parser = subparsers.add_parser('launch', help='Launch the QuickDistill UI')
    launch_parser.add_argument('--port', type=int, default=5001, help='Port to run the server on (default: 5001)')
    launch_parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    launch_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    launch_parser.set_defaults(func=launch_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
