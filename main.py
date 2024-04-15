#!/usr/bin/env python
from sd_model_manager.server_app import create_server

def main() -> None:
    create_server(start_type="loop", need_db=False, try_debug=False)

if __name__ == "__main__":
    main()
